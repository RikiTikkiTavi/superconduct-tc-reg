from copy import deepcopy
import logging
import math
from typing import Union
import hydra
import mlflow.models
import onnxmltools
import pandas as pd
from superconduct_tc_reg.data.target_scaler import TargetScaler
from superconduct_tc_reg.pipeline.abstract import SuperconductPipeline
import xgboost as xgb
import sklearn.metrics
import numpy as np
import lightning.pytorch.loggers
import mlflow
import mlflow.xgboost
import mlflow.models
import mlflow.models.signature
import mlflow.types

_logger = logging.getLogger(__name__)


class TrackingCB(xgb.callback.TrainingCallback):  # type: ignore
    def __init__(
        self,
        metric_to_key: dict[str, str],
        subset_to_prefix: dict[str, str],
        tracking_logger: lightning.pytorch.loggers.MLFlowLogger,
        fold_postfix: str = "",
    ):
        super().__init__()
        self.metric_to_key = metric_to_key
        self.subset_to_suffix = subset_to_prefix
        self.tracking_logger = tracking_logger
        self.fold_postfix = fold_postfix

    def build_metric_key(self, val_subset_name: str, metric_name: str):
        if (
            val_subset_name in self.subset_to_suffix
            and metric_name in self.metric_to_key
        ):
            return f"{self.subset_to_suffix[val_subset_name]}_{self.metric_to_key[metric_name]}{self.fold_postfix}"
        else:
            return f"{val_subset_name}-{metric_name}"

    def extract_step_metrics(self, evals_log: dict[str, dict], step: int):
        metrics_to_log = {}
        for val_subset_name, val_subset_metrics in evals_log.items():
            for metric_name in val_subset_metrics.keys():
                metrics_to_log[self.build_metric_key(val_subset_name, metric_name)] = (
                    val_subset_metrics[metric_name][step]
                )
        return metrics_to_log

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ):

        self.tracking_logger.log_metrics(
            self.extract_step_metrics(evals_log, epoch), step=epoch
        )

    def after_training(self, model):
        self.tracking_logger.save()
        return model


class XGBPipeline(SuperconductPipeline):
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        config = self.config
        target = config["target"]
        features = [c for c in df_train.columns if c != target]
        self.features = features

        #  Tracking
        cb_tracking = TrackingCB(
            metric_to_key={"rmse": "RMSE"},
            subset_to_prefix={"validation_0": "train", "validation_1": "val"},
            tracking_logger=self.tracking_logger,
            fold_postfix=self.fold_postfix,
        )
        self.cb_tracking_ = cb_tracking
        cbs = [cb_tracking]

        # Early stopping
        self.do_early_stopping_ = (
            "early_stopping" in config and config["early_stopping"] is not None
        )
        if self.do_early_stopping_:
            cbs.append(hydra.utils.instantiate(config["early_stopping"]))

        # Model
        model = hydra.utils.instantiate(config["model"])(callbacks=cbs)
        self.model_ = model

        self._df_example = df_train.head(n=3)

        # Shuffle data
        df_train = df_train.sample(
            frac=1, replace=False, random_state=config["train_shuffle_seed"]
        )

        # Train and validate
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            verbose=self.config["verbose"],
        )

        val_results = cb_tracking.extract_step_metrics(
            model.evals_result(), self.global_step
        )

        _logger.info(f"Epoch: {self.global_step}")
        _logger.info(val_results)

        return val_results

    def test(self, df: pd.DataFrame):
        assert getattr(self, "model_", None) is not None
        y_pred = self.model_.predict(df[self.model_.feature_names_in_])
        y_true = df[self.config["target"]]
        metrics = {
            "test_RMSE": math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
        }
        self.tracking_logger.log_metrics(metrics, self.global_step)  # type: ignore
        self.tracking_logger.save()
        _logger.info(metrics)
        return metrics

    @property
    def global_step(self):
        if self.do_early_stopping_:
            return self.model_.best_iteration
        return self.model_.get_num_boosting_rounds() - 1

    def log_model(self, export_onnx: bool = True):
        input_data = self._df_example[self.features]
        signature = mlflow.models.infer_signature(
            input_data, self.model_.predict(input_data)
        )
        mlflow.xgboost.log_model(
            xgb_model=self.model_,
            artifact_path="models/superconduct-gbdt:xgb",
            signature=signature,
            input_example=input_data,
            run_id=self.tracking_logger.run_id,
        )

        if export_onnx:
            from xgboost import XGBRegressor
            from skl2onnx.common.data_types import FloatTensorType
            from skl2onnx import convert_sklearn, update_registered_converter
            from skl2onnx.common.shape_calculator import (
                calculate_linear_regressor_output_shapes,
            )
            from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
            xgb_model_copy = deepcopy(self.model_)
            xgb_model_copy.get_booster().feature_names = [f"f{i}" for i in range(input_data.shape[1])]

            update_registered_converter(
                XGBRegressor,
                "XGBoostXGBRegressor",
                calculate_linear_regressor_output_shapes,
                convert_xgboost,
            )
            model_onnx = convert_sklearn(
                xgb_model_copy,
                "xgb_model",
                initial_types=[("input", FloatTensorType([None, input_data.shape[1]]))],
                final_types=[("critical_temp", FloatTensorType([None, 1]))],
                target_opset={"": 12, "ai.onnx.ml": 2},
            )
            model_onnx.graph.output[0].name = "critical_temp"
            mlflow.start_run(self.tracking_logger.run_id)
            mlflow.onnx.log_model(
                model_onnx,
                input_example=input_data,
                signature=mlflow.models.signature.ModelSignature(
                    inputs=mlflow.types.Schema(
                        [mlflow.types.ColSpec(type="float", name=col) for col in self.features]
                    ),
                    outputs=mlflow.types.Schema(
                        [mlflow.types.TensorSpec(np.dtype(np.float32), shape=(-1, 1))]
                    ),
                ),
                artifact_path="models/superconduct-gbdt:onnx",
            )