from functools import cache
import logging
import os
from pathlib import Path
from tempfile import tempdir
import tempfile
from typing import NoReturn, Optional
import uuid
import filelock
import hydra.types
import joblib
import mlflow
import omegaconf
import pandas as pd
import sklearn.preprocessing
import lightning
import lightning.pytorch.loggers
import hydra
import hydra.core.hydra_config
import lightning.pytorch.callbacks
import lightning.fabric.utilities.logger
import sklearn.model_selection
import hashlib


from superconduct_tc_reg.utils import (
    dicts_mean,
    seed_everything,
    untensor_dict,
    remove_in_keys,
)
from superconduct_tc_reg.data.target_scaler import TargetScaler
from superconduct_tc_reg.pipeline.abstract import SuperconductPipeline
from superconduct_tc_reg.data.process import DataProcessor

_logger = logging.getLogger(__name__)

# Register the function with OmegaConf
omegaconf.OmegaConf.register_new_resolver(
    "uuid.uuid4", lambda _: str(uuid.uuid4()), use_cache=True
)


def log_job_num(tracking_logger):
    try:
        hydra_conf = hydra.core.hydra_config.HydraConfig.get()
        if hydra_conf.mode == hydra.types.RunMode.MULTIRUN:
            tracking_logger.log_hyperparams({"hydra/job/num": hydra_conf.job.num})
    except ValueError:
        pass


def process_data_cached(config) -> tuple[pd.DataFrame, DataProcessor]:
    """Process-safe data preprocessing with caching"""
    cache_path = Path(config.dir.cache)
    cache_path.mkdir(exist_ok=True, parents=True)

    hash_key = hashlib.md5(str(config["preprocessing"]).encode()).hexdigest()
    dataset_cache_file = cache_path / f"processed_data_{hash_key}.parquet"
    processor_cache_file = cache_path / f"data_processor_{hash_key}.joblib"

    with filelock.FileLock(f"{str(dataset_cache_file)}.lock"):

        if os.path.exists(dataset_cache_file) and os.path.exists(processor_cache_file):
            _logger.info("Loading cached data...")
            dp = joblib.load(processor_cache_file)
            return pd.read_parquet(dataset_cache_file), dp

        _logger.info("Processing data...")
        data_processor = DataProcessor(config)
        processed_data: pd.DataFrame = data_processor.read_fit_transform()
        processed_data.to_parquet(dataset_cache_file)
        joblib.dump(data_processor, processor_cache_file)

        return processed_data, data_processor


def log_data_processor(dp: DataProcessor, config, run_id):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, config["data_processor_artifact"]["name"])
        joblib.dump(dp, local_path)
        mlflow.log_artifact(
            local_path=local_path,
            artifact_path=config["data_processor_artifact"]["path"],
            run_id=run_id,
        )


@hydra.main(
    config_path="../../configs",
    config_name="train",
    version_base="1.3",
)
def train(config) -> float | NoReturn:
    # Setup tracking
    # We can access mlflow client through lightning logger
    tracking_logger: lightning.pytorch.loggers.MLFlowLogger = hydra.utils.instantiate(
        config["tracking"]
    )(run_id=None)

    try:
        # Set seed
        if config["seed"] is not None:
            _logger.info(f"Set seed={config.seed}")
            seed_everything(
                config["seed"],
                enable_deterministic=(
                    "trainer" in config["pipeline"]
                    and config["pipeline"]["trainer"]["deterministic"]
                ),
            )

        # Log config
        tracking_logger.log_hyperparams(config)

        # Read & process data
        df, data_processor = process_data_cached(config)
        
        if config.data_processor_artifact.log:
            log_data_processor(data_processor, config, tracking_logger.run_id)

        # Train+val - test split
        df_train_val, df_test = sklearn.model_selection.train_test_split(
            df,
            test_size=config["test_sample"]["size"],
            random_state=config["test_sample"]["seed"],
        )

        # Train - val split
        spliter = hydra.utils.instantiate(config["split"])

        fold_val_metrics = []

        for i, (train_idx, val_idx) in enumerate(spliter(elements=df_train_val.index)):
            fold_i = None

            fold_i = i if spliter.n_folds > 1 else None

            pipeline: SuperconductPipeline = hydra.utils.instantiate(
                config["pipeline"], _recursive_=False
            )(
                target_scaler=data_processor.target_scaler,
                tracking_logger=tracking_logger,
                fold_i=fold_i,
            )

            metrics_val = pipeline.fit(
                df_train=df.loc[train_idx], df_val=df.loc[val_idx]
            )
            metrics_val = remove_in_keys(metrics_val, pipeline.fold_postfix)

            fold_val_metrics.append(metrics_val)

            # Test
            # There is no sense to do the test in the multiple-folds settings. So it is assumed
            # that do_test is set only with single fold splits.
            if config["do_test"]:
                pipeline.test(df_test)

        mean_metrics = dicts_mean(fold_val_metrics)

        _logger.info(mean_metrics)

        if config.do_log_model:
            pipeline.log_model(export_onnx=config.do_export_onnx)

        # We want to log the mean metrics over folds if in multiple folds settings
        if spliter.n_folds > 1:
            tracking_logger.log_metrics(
                untensor_dict(mean_metrics), step=pipeline.global_step  # type: ignore
            )

        tracking_logger.finalize("success")

        return mean_metrics[config["target_metric"]]

    except Exception as e:
        _logger.error("Error in train pipeline", exc_info=True)
        tracking_logger.finalize("failed")
        raise Exception("Train pipilene terminated with exception.") from e


if __name__ == "__main__":
    train()
