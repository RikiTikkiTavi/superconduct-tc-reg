from dataclasses import dataclass
import logging
from typing import Optional
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch
import torchmetrics.regression.mse
import tqdm
import torchmetrics.regression
import lightning
import lightning.pytorch.loggers
import hydra
import lightning.pytorch.callbacks

_logger = logging.getLogger(__name__)


class DNNModel(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        output_size: int,
        activation: str,
        dropout: float = 0.5,
    ):
        if hidden_size > 0 and n_hidden > 0:
            activation_class = getattr(torch.nn, activation)
            layers = []
            for i in range(n_hidden):
                size = input_size if i == 0 else hidden_size
                layers.append(nn.Linear(size, hidden_size))
                layers.append(activation_class())
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_size, output_size))
        else:
            layers = [nn.Linear(input_size, output_size)]

        super().__init__(*layers)


class ModelModule(lightning.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        target: str,
        loss: nn.Module,
        config,
        loss_fold_postfix="",
        metrics_fold_postfix="",
        log_metrics: bool = False,
        log_loss: bool = True,
    ):
        super().__init__()

        self._model = model
        self._loss = loss

        self._target = target

        self._config = config

        metrics = torchmetrics.MetricCollection(
            metrics={
                "MSE": torchmetrics.regression.MeanSquaredError(
                    num_outputs=1
                ),
                "R2": torchmetrics.regression.R2Score(num_outputs=1),
            }
        )

        self._stage_to_metrics = torch.nn.ModuleDict(
            {
                "val": metrics.clone(prefix=f"val_"),
                "test": metrics.clone(prefix=f"test_"),
            }
        )

        self._stage_to_losses = {
            "val": [],
            "test": [],
        }

        self._loss_fold_postfix = loss_fold_postfix
        self._metrics_fold_postfix = metrics_fold_postfix

        self._log_metrics = log_metrics
        self._log_loss = log_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)
        self.log(f"train_loss{self._loss_fold_postfix}", loss)
        return loss

    def _eval_step(self, batch, batch_idx, dataloader_idx=0, stage="val"):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y).item()

        # Save batch loss
        self._stage_to_losses[stage].append(loss)
        # Update metrics
        self._stage_to_metrics[stage].update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, dataloader_idx, stage="val")

    def _metric_name(self, target: str, metric: str):
        return f"{metric}/{target}{self._metrics_fold_postfix}"

    def _calculate_metrics(self, stage: str) -> dict[str, float]:

        metrics: dict[str, torch.Tensor] = self._stage_to_metrics[stage].compute()

        self._stage_to_metrics[stage].reset()

        return metrics

    def _calculate_eval_loss(self, stage: str):
        loss_mean = torch.tensor(self._stage_to_losses[stage]).mean().item()

        self._stage_to_losses[stage].clear()

        return loss_mean

    def on_validation_epoch_end(self):

        metrics_processed = self._calculate_metrics(stage="val")

        if self._log_metrics:
            self.log_dict(
                metrics_processed,
                add_dataloader_idx=False,
                on_epoch=True,
                on_step=False,
            )

        self.last_val_metric_values = metrics_processed
        self.val_loss = self._calculate_eval_loss(stage="val")

        if self._log_loss:
            self.log(f"val_loss{self._loss_fold_postfix}", self.val_loss)

        return metrics_processed

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        metrics_processed = self._calculate_metrics(stage="test")

        self.log_dict(
            metrics_processed,
            add_dataloader_idx=False,
            on_epoch=True,
            on_step=False,
        )

        test_loss = self._calculate_eval_loss(stage="test")

        self.log(f"test_loss{self._loss_fold_postfix}", test_loss)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self._config["optimizer"])(
            params=self._model.parameters()
        )
        lr_scheduler = hydra.utils.instantiate(self._config["lr_scheduler"])(
            optimizer=optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
