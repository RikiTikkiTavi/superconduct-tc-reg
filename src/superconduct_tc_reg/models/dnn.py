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

from superconduct_tc_reg.data.target_scaler import TargetScaler

_logger = logging.getLogger(__name__)


class DNNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        output_size: int,
        activation: str,
        dropout: float = 0.5,
        batch_norm: bool = False,
        skip_conn: bool = False,
        softplus: bool = False,
        dropout_seed: Optional[int] = None,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.n_hidden = n_hidden
        self.skip_conn = skip_conn

        if hidden_size > 0 and n_hidden > 0:
            activation_class = getattr(torch.nn, activation, None)
            for i in range(n_hidden):
                size = input_size if i == 0 else hidden_size
                self.layers.append(nn.Linear(size, hidden_size))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_size))
                if activation_class is not None:
                    self.layers.append(activation_class())
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(hidden_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, output_size))
        if softplus:
            self.layers.append(nn.Softplus())

    def init_weights(self, seed: Optional[int] = None):

        if seed is not None:
            # Save current state
            _logger.info(f"Capture rng state before weights init ...")
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_states = torch.cuda.get_rng_state_all()

            # Set weights seed
            _logger.info(f"Set weights init seed={seed}")
            torch.manual_seed(seed)

        # init weights
        _logger.info(f"Init weights ...")
        gain = torch.nn.init.calculate_gain("tanh")
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.0)

        if seed is not None:
            # Set initial state back
            _logger.info(f"Restore rng state after weights init ...")
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(gpu_rng_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = None
        for i, m in enumerate(self.layers):

            # After input layer
            if i == 1 and self.n_hidden > 1 and self.skip_conn:
                x1 = x

            if i == len(self.layers) - 3 and x1 is not None:
                x = x + x1

            x = m(x)

        return x


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
        target_scaler: Optional[TargetScaler] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self._model = model
        self._loss = loss

        self._target = target

        self._config = config

        self._target_scaler = target_scaler

        metrics = torchmetrics.MetricCollection(
            metrics={
                "MSE": torchmetrics.regression.MeanSquaredError(num_outputs=1),
                "MAE": torchmetrics.regression.MeanAbsoluteError(),
                "RMSE": torchmetrics.regression.MeanSquaredError(
                    num_outputs=1, squared=False
                ),
                "R2": torchmetrics.regression.R2Score(num_outputs=1),
            }
        )

        self._stage_to_metrics = torch.nn.ModuleDict(
            {
                "val": metrics.clone(prefix=f"val_", postfix=metrics_fold_postfix),
                "test": metrics.clone(prefix=f"test_", postfix=metrics_fold_postfix),
            }
        )

        self._stage_to_losses = {
            "val": [],
            "test": [],
        }

        self._stage_to_outputs = {
            "val": [],
            "test": [],
        }

        self._loss_fold_postfix = loss_fold_postfix
        self._metrics_fold_postfix = metrics_fold_postfix

        self._log_metrics = log_metrics
        self._log_loss = log_loss

        self.stage_to_last_metrics = {"val": {}, "test": {}}

    def outputs(self, stage: str) -> torch.Tensor:
        return torch.concat(self._stage_to_outputs[stage], dim=0).flatten()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)
        self.log(
            f"train_loss{self._loss_fold_postfix}", loss, on_step=False, on_epoch=True
        )
        return loss

    def _eval_step(self, batch, batch_idx, dataloader_idx=0, stage="val"):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y).item()

        if self._target_scaler is not None:
            y = self._target_scaler.inverse_transform(y)
            y_hat = self._target_scaler.inverse_transform(y_hat)

        # Save batch loss
        self._stage_to_losses[stage].append(loss)
        # Update metrics
        self._stage_to_metrics[stage].update(y_hat, y)

        return y_hat, loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_batch_end(
        self, outputs: tuple[torch.Tensor, torch.Tensor], batch, batch_idx, *_
    ):
        y_hat_batch, _ = outputs
        if batch_idx == 0:
            self._stage_to_outputs["val"].clear()

        self._stage_to_outputs["val"].append(y_hat_batch.detach().cpu())

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

        self.stage_to_last_metrics["val"] = {
            f"val_loss{self._loss_fold_postfix}": self.val_loss,
            **self.last_val_metric_values,
        }

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

        self.stage_to_last_metrics["test"] = {
            f"test_loss{self._loss_fold_postfix}": self.val_loss,
            **metrics_processed,
        }

        self.log(f"test_loss{self._loss_fold_postfix}", test_loss)

    def configure_optimizers(self):
        result = {
            "optimizer": hydra.utils.instantiate(self._config["optimizer"])(
                params=self._model.parameters()
            )
        }

        if self._config["lr_scheduler"] is not None:
            result["lr_scheduler"] = {
                "scheduler": hydra.utils.instantiate(self._config["lr_scheduler"])(
                    optimizer=result["optimizer"]
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": f"val_loss{self._loss_fold_postfix}",
            }

        return result

    def on_fit_start(self) -> None:
        super().on_fit_start()

        dropout_seed = self._config["model"]["dropout_seed"]

        if dropout_seed is not None:
            self.__initial_state_cpu = torch.random.get_rng_state()
            self.__initial_states_gpu = torch.cuda.get_rng_state_all()
            _logger.info(
                f"Setting torch seed (to affect dropout) to dropout seed: {dropout_seed} on all devices:"
            )
            torch.manual_seed(dropout_seed)

    def on_fit_end(self) -> None:
        super().on_fit_end()

        dropout_seed = self._config["model"]["dropout_seed"]

        if dropout_seed is not None:

            _logger.info(
                f"Restoring torch rng states (affected dropout) back to initial"
            )

            torch.set_rng_state(self.__initial_state_cpu)
            torch.cuda.set_rng_state_all(self.__initial_states_gpu)
