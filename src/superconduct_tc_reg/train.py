import logging
import os
import random
from typing import Optional, Sequence
import uuid
import hydra.types
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
import sklearn.model_selection

from superconduct_tc_reg.utils import dicts_mean, seed_everything
from superconduct_tc_reg.model_module import DNNModel, ModelModule

_logger = logging.getLogger(__name__)

def log_job_num(tracking_logger):
    try:
        hydra_conf = hydra.core.hydra_config.HydraConfig.get()
        if hydra_conf.mode == hydra.types.RunMode.MULTIRUN:
            tracking_logger.log_hyperparams({"hydra/job/num": hydra_conf.job.num})
    except ValueError:
        pass

@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base="1.3",
)
def train(config):
    # Set seed
    if config["seed"] is not None:
        _logger.info(f"Set seed={config.seed}")
        seed_everything(config["seed"], enable_deterministic=config["trainer"]["deterministic"])

    # Read dataset
    df = pd.read_csv(config["dataset"]["dir"])
    target = config["target"]

    # Setup tracking
    tracking_logger = hydra.utils.instantiate(config["tracking"])

    # Log config
    tracking_logger.log_hyperparams(config)
    # Log job number if doing hopt with hydra
    log_job_num(tracking_logger)

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

    for i, (train_idx, val_idx) in enumerate(
        spliter(elements=df_train_val.index)
    ):

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
        loss = torch.nn.MSELoss()

        # Model
        model = DNNModel(
            input_size=len(features),
            hidden_size=config["model"]["hidden_size"],
            n_hidden=config["model"]["n_hidden"],
            output_size=1,
            dropout=config["model"]["dropout"],
            activation=config["model"]["activation"],
        )

        loss_fold_prefix = "" if spliter.n_folds == 1 else f"-{target}-fold-{i}"
        log_metrics = spliter.n_folds == 1
        model_module = ModelModule(
            model=model,
            target=target,
            loss=loss,
            config=config,
            loss_fold_postfix=loss_fold_prefix,
            log_metrics=log_metrics,
            log_loss=True,
        )

        # Trainer
        trainer = lightning.Trainer(
            callbacks=[
                lightning.pytorch.callbacks.ModelSummary(max_depth=2),
                lightning.pytorch.callbacks.TQDMProgressBar(),
            ],
            max_epochs=config["trainer"]["max_epochs"],
            log_every_n_steps=1,
            logger=tracking_logger,
            gradient_clip_val=config["trainer"]["gradient_clip_val"],
            enable_checkpointing=False,
            accelerator=config["trainer"]["accelerator"],
            deterministic=config["trainer"]["deterministic"],
            devices=config["trainer"]["devices"],
            num_sanity_val_steps=0,
        )

        # Train
        # Set data shuffle seed
        data_loader_generator = torch.default_generator
        if config["data_loader"]["seed"] is not None:
            data_loader_generator = torch.Generator()
            data_loader_generator.manual_seed(config["data_loader"]["seed"])

        # Set dropout seed
        if config["model"]["dropout_seed"] is not None:
            _logger.info(
                f'Setting torch seed (to affect dropout) to dropout seed: {config["model"]["dropout_seed"]}'
            )
            torch.cuda.manual_seed_all(config["model"]["dropout_seed"])
            torch.manual_seed(config["model"]["dropout_seed"])

        trainer.fit(
            model_module,
            train_dataloaders=torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=config["trainer"]["batch_size"],
                generator=data_loader_generator,
            ),
            val_dataloaders=torch.utils.data.DataLoader(
                dataset=dataset_val,
                batch_size=config["trainer"]["batch_size"],
                shuffle=False,
            ),
        )

        if config["model"]["dropout_seed"] is not None and config["seed"] is not None:
            _logger.info(
                f"Setting torch seed (affected dropout) back to default seed: {config.seed}"
            )
            torch.cuda.manual_seed_all(config.seed)  # type: ignore
            torch.manual_seed(config.seed)

        # Save fold metrics
        metrics_last_epoch.update(model_module.last_val_metric_values)

        fold_metrics.append(model_module.last_val_metric_values)
        fold_losses.append(model_module.val_loss)

    _logger.info(dicts_mean(fold_metrics))

    if spliter.n_folds > 1:
        tracking_logger.log_metrics(
            dicts_mean(fold_metrics), step=trainer.global_step
        )

    # Test
    if config["do_test"]:
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

    return np.mean(fold_losses)


if __name__ == "__main__":
    train()
