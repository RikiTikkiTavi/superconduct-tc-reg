import logging
import os
import random
from typing import Optional, Sequence
import uuid
import hydra.types
import mlflow
import mlflow.data
import mlflow.entities
import numpy as np
import omegaconf
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch
import torch.utils.data.sampler
import torchmetrics.regression.mse
import tqdm
import torchmetrics.regression
import lightning
import lightning.pytorch.loggers
import hydra
import hydra.core.hydra_config
import lightning.pytorch.callbacks
import lightning.fabric.utilities.logger
import sklearn.model_selection

from superconduct_tc_reg.utils import dicts_mean, seed_everything, untensor_dict
from superconduct_tc_reg.model_module import DNNModel, ModelModule

_logger = logging.getLogger(__name__)


def log_job_num(tracking_logger):
    try:
        hydra_conf = hydra.core.hydra_config.HydraConfig.get()
        if hydra_conf.mode == hydra.types.RunMode.MULTIRUN:
            tracking_logger.log_hyperparams({"hydra/job/num": hydra_conf.job.num})
    except ValueError:
        pass


def log_dataset(tracking_logger, df: pd.DataFrame, config):
    # TODO: modify .experiment.
    tracking_logger.experiment.log_inputs(
        run_id=tracking_logger.run_id,
        datasets=[
            mlflow.data.DatasetInput(
                dataset=mlflow.data.from_pandas(  # type: ignore
                    df, source=config["dataset"]["dir"], targets=config["target"]
                )._to_mlflow_entity(),
                tags=[
                    mlflow.entities.InputTag(
                        key="mlflow.data.context", value="train+val+test"
                    )
                ],
            )
        ],
    )


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base="1.3",
)
def train(config):

    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(config["tracking"]["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config["tracking"]["experiment_name"],
            artifact_location=config["tracking"]["artifact_location"],
        )
    else:
        experiment_id = experiment.experiment_id

    parent_run = mlflow.start_run(experiment_id=experiment_id)

    # Setup tracking
    tracking_logger = hydra.utils.instantiate(config["tracking"])(
        run_id=parent_run.info.run_uuid
    )

    # Set seed
    if config["seed"] is not None:
        _logger.info(f"Set seed={config.seed}")
        seed_everything(
            config["seed"], enable_deterministic=config["trainer"]["deterministic"]
        )

    # Read dataset
    df = pd.read_csv(config["dataset"]["dir"])
    target = config["target"]

    # Log config
    mlflow.log_params(lightning.fabric.utilities.logger._flatten_dict(config))
    # Log job number if doing hopt with hydra
    # log_job_num(tracking_logger)

    # Log dataset
    mlflow.log_input(
        dataset=mlflow.data.from_pandas(df, targets=target),  # type: ignore
        context="train+val+test",
    )

    # Train+val - test split
    df_train_val, df_test = sklearn.model_selection.train_test_split(
        df,
        test_size=config["test_sample"]["size"],
        random_state=config["test_sample"]["seed"],
    )

    # Train - val split
    spliter = hydra.utils.instantiate(config["split"])

    fold_metrics = []
    fold_losses = []

    features = [c for c in df.columns if c != target]

    for i, (train_idx, val_idx) in enumerate(spliter(elements=df_train_val.index)):

        if spliter.n_folds > 1:
            child_run = mlflow.start_run(
                run_name=f"fold-{i}",
                experiment_id=experiment_id,
                parent_run_id=parent_run.info.run_id,
                nested=True,
            )

        metrics_last_epoch = {}

        dataset_train = torch.utils.data.TensorDataset()

        # Configure train and val datasets
        df_train = df.loc[train_idx]
        dataset_train = torch.utils.data.TensorDataset(
            torch.tensor(df_train[features].to_numpy(), dtype=torch.float32),
            torch.tensor(df_train[[target]].to_numpy(), dtype=torch.float32),
        )

        df_val = df.loc[val_idx]
        dataset_val = torch.utils.data.TensorDataset(
            torch.tensor(df_val[features].to_numpy(), dtype=torch.float32),
            torch.tensor(df_val[[target]].to_numpy(), dtype=torch.float32),
        )

        # Loss
        loss = hydra.utils.instantiate(config["loss"])

        # Model
        model = DNNModel(
            input_size=len(features),
            hidden_size=config["model"]["hidden_size"],
            n_hidden=config["model"]["n_hidden"],
            output_size=1,
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
        )

        model_module = ModelModule(
            model=model,
            target=target,
            loss=loss,
            config=config,
            loss_fold_postfix="",
            log_metrics=True,
            log_loss=True,
        )

        cb_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=f"checkpoints/fold-{i}",
            monitor="val_loss",
            save_top_k=2,
            filename="{epoch}-{val_loss:.2f}",
            mode="min",
            save_weights_only=True,
        )

        # Trainer
        trainer = lightning.Trainer(
            callbacks=[
                lightning.pytorch.callbacks.ModelSummary(max_depth=2),
                lightning.pytorch.callbacks.TQDMProgressBar(),
                cb_checkpoint,
            ],
            max_epochs=config["trainer"]["max_epochs"],
            log_every_n_steps=1,
            gradient_clip_val=config["trainer"]["gradient_clip_val"],
            enable_checkpointing=True,
            accelerator=config["trainer"]["accelerator"],
            deterministic=config["trainer"]["deterministic"],
            devices=config["trainer"]["devices"],
            num_sanity_val_steps=0,
            logger=tracking_logger,
        )

        # Train
        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=config["trainer"]["batch_size"],
            shuffle=False,
        )
        trainer.fit(
            model_module,
            train_dataloaders=torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=config["trainer"]["batch_size"],
            ),
            val_dataloaders=dataloader_val,
        )

        # Save fold metrics
        metrics_last_epoch.update(model_module.last_val_metric_values)

        fold_metrics.append(model_module.last_val_metric_values)
        fold_losses.append(model_module.val_loss)

        best_model = ModelModule.load_from_checkpoint(
            cb_checkpoint.best_model_path
        )._model
        best_model.eval()
        example_batch_input = next(iter(dataloader_val))[0]
        with torch.no_grad():
            signature = mlflow.models.infer_signature(
                model_input=example_batch_input.numpy(),
                model_output=best_model(example_batch_input).numpy(),
            )
        mlflow.pytorch.log_model(best_model, "model", signature=signature)

        # Test
        if config["do_test"]:
            mlflow.log_input(
                dataset=mlflow.data.from_pandas(  # type: ignore
                    df_test, targets=target
                ),
                context="test",
            )

            dataset_test = torch.utils.data.TensorDataset(
                torch.tensor(df_test[features].to_numpy(), dtype=torch.float32),
                torch.tensor(df_test[[target]].to_numpy(), dtype=torch.float32),
            )

            trainer.test(
                model_module,
                dataloaders=torch.utils.data.DataLoader(
                    dataset=dataset_test,
                    batch_size=config["trainer"]["batch_size"],
                    shuffle=False,
                ),
            )

        if spliter.n_folds > 1:
            mlflow.end_run()

    _logger.info(dicts_mean(fold_metrics))

    if spliter.n_folds > 1:
        mlflow.log_metrics(
            untensor_dict(dicts_mean(fold_metrics)), step=trainer.global_step
        )

    mlflow.end_run()

    return np.mean(fold_losses)


if __name__ == "__main__":
    train()
