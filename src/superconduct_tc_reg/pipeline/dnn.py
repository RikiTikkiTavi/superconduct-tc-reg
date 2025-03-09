import io
import logging
from typing import Optional, Sequence

import hydra.types
import mlflow
import mlflow.data
import mlflow.entities
import mlflow.models
import mlflow.models
import mlflow.models.signature
import mlflow.types
import numpy as np
import omegaconf
import onnx
import pandas as pd

import torch.nn as nn
import torch.utils.data
import torch
import torch.utils.data.sampler
import lightning
import lightning.pytorch.loggers
import hydra
import hydra.core.hydra_config
import lightning.pytorch.callbacks
import lightning.fabric.utilities.logger
import torch.onnx

from superconduct_tc_reg.models.dnn import DNNModel, ModelModule
from superconduct_tc_reg.pipeline.abstract import SuperconductPipeline
from superconduct_tc_reg.utils import untensor_dict

_logger = logging.getLogger(__name__)


class DNNPipeline(SuperconductPipeline):

    def log_model(self, export_onnx: bool = False):

        # Do not use context here to exclude mlflow finishes run.
        # Just set global mlflow context to current run
        # NOTE: This is a workaround for mlflow client still does not have log_model...
        # https://github.com/mlflow/mlflow/issues/7392
        mlflow.start_run(run_id=self.tracking_logger.run_id)

        input_schema = mlflow.types.Schema(
            [mlflow.types.ColSpec("float", f) for f in self.features]
        )

        torch_output_schema = mlflow.types.Schema(
            [
                mlflow.types.TensorSpec(
                    type=np.dtype(np.float32),
                    shape=(-1, 1),
                )
            ]
        )

        signature = mlflow.models.signature.ModelSignature(
            inputs=input_schema, outputs=torch_output_schema
        )

        example_input = self._df_train_head[self.features]

        mlflow.pytorch.log_model(
            self.model,
            artifact_path="models/superconduct-dnn:torch",
            input_example=example_input,
            signature=signature,
            run_id=self.tracking_logger.run_id,
        )

        if export_onnx:
            buff = io.BytesIO()
            torch.onnx.export(
                model=self.model,
                args=(torch.tensor(example_input.to_numpy(), dtype=torch.float32),),
                f=buff,
                input_names=["input"],
                output_names=["critical_temp"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "critical_temp": {0: "batch_size"},
                },
                opset_version=20,
            )
            onnx_model = onnx.load_from_string(buff.getvalue())
            mlflow.onnx.log_model(
                onnx_model,
                input_example=example_input,
                signature=mlflow.models.signature.ModelSignature(
                    inputs=input_schema,
                    outputs=mlflow.types.Schema(
                        [mlflow.types.ColSpec(type="float", name="critical_temp")]
                    ),
                ),
                artifact_path="models/superconduct-dnn:onnx",
            )

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        config = self.config

        torch.set_float32_matmul_precision(config["matmul_precision"])

        target = config["target"]
        self.target = target
        features = [c for c in df_train.columns if c != target]
        self.features = features

        # Configure train and val datasets
        dataset_train = torch.utils.data.TensorDataset(
            torch.tensor(df_train[features].to_numpy(), dtype=torch.float32),
            torch.tensor(df_train[[target]].to_numpy(), dtype=torch.float32),
        )

        dataset_val = torch.utils.data.TensorDataset(
            torch.tensor(df_val[features].to_numpy(), dtype=torch.float32),
            torch.tensor(df_val[[target]].to_numpy(), dtype=torch.float32),
        )

        self._df_train_head = df_train.head(3)

        # Loss
        loss = hydra.utils.instantiate(config["loss"])

        # Model
        model: DNNModel = hydra.utils.instantiate(config["model"])(
            input_size=len(features)
        )
        model.init_weights(seed=config["weights_init"]["seed"])

        model_module = ModelModule(
            model=model,
            target=target,
            loss=loss,
            config=config,
            loss_fold_postfix=self.fold_postfix,
            metrics_fold_postfix=self.fold_postfix,
            log_metrics=True,
            log_loss=True,
            target_scaler=self.target_scaler,
        )

        cbs = [
            lightning.pytorch.callbacks.ModelSummary(max_depth=3),
            lightning.pytorch.callbacks.TQDMProgressBar(),
            lightning.pytorch.callbacks.LearningRateMonitor(
                logging_interval="step", log_momentum=True
            ),
        ]

        cb_checkpoint = None
        if config["do_checkpointing"]:
            cb_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
                dirpath=f"checkpoints/{self.fold_postfix}",
                monitor=f"val_loss{self.fold_postfix}",
                save_top_k=2,
                filename="{epoch}-{val_loss:.2f}",
                mode="min",
                save_weights_only=True,
            )
            cbs.append(cb_checkpoint)

        # Trainer
        trainer = lightning.Trainer(
            callbacks=cbs,
            max_epochs=config["trainer"]["max_epochs"],
            log_every_n_steps=1,
            gradient_clip_val=config["trainer"]["gradient_clip_val"],
            enable_checkpointing=config["do_checkpointing"],
            accelerator=config["trainer"]["accelerator"],
            deterministic=config["trainer"]["deterministic"],
            devices=config["trainer"]["devices"],
            num_sanity_val_steps=0,
            logger=self.tracking_logger,
            precision=config["trainer"]["precision"],
        )

        # Create rng for train data loader
        data_loader_generator = torch.default_generator
        if config["data_loader"]["seed"] is not None:
            data_loader_generator = torch.Generator()
            data_loader_generator.manual_seed(config["data_loader"]["seed"])

        # Val data loader
        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=config["trainer"]["batch_size"],
            num_workers=config["data_loader"]["num_workers"],
            persistent_workers=config["data_loader"]["persistent_workers"],
            shuffle=False,
        )

        # Fit
        # NOTE: Dropout seed will be set before fit in model module
        # NOTE After the fit rng states will be restored
        trainer.fit(
            model_module,
            train_dataloaders=torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=config["trainer"]["batch_size"],
                num_workers=config["data_loader"]["num_workers"],
                persistent_workers=config["data_loader"]["persistent_workers"],
                drop_last=config["data_loader"]["drop_last"],
                pin_memory=config["data_loader"]["pin_memory"],
                shuffle=True,
                generator=data_loader_generator,
            ),
            val_dataloaders=dataloader_val,
        )

        _logger.info(model_module.stage_to_last_metrics["val"])

        self.trainer = trainer
        self.model_module = model_module
        self.model = model
        self.cb_checkpoint = cb_checkpoint

        return untensor_dict(model_module.stage_to_last_metrics["val"])

    def test(self, df: pd.DataFrame):
        dataset_test = torch.utils.data.TensorDataset(
            torch.tensor(df[self.features].to_numpy(), dtype=torch.float32),
            torch.tensor(df[[self.target]].to_numpy(), dtype=torch.float32),
        )

        self.trainer.test(
            self.model_module,
            dataloaders=torch.utils.data.DataLoader(
                dataset=dataset_test,
                batch_size=self.config["trainer"]["batch_size"],
                shuffle=False,
            ),
        )

        _logger.info(self.model_module.stage_to_last_metrics["test"])

        return untensor_dict(self.model_module.stage_to_last_metrics["test"])

    @property
    def global_step(self):
        return self.trainer.global_step - 1
