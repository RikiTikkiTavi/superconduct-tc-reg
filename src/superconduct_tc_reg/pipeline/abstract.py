from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Optional, Union

import pandas as pd
import lightning.pytorch.loggers

from superconduct_tc_reg.data.target_scaler import TargetScaler


class SuperconductPipeline(ABC):
    def __init__(
        self,
        tracking_logger: lightning.pytorch.loggers.MLFlowLogger,
        target_scaler: TargetScaler,
        fold_i: Optional[int] = None,
        **pipeline_config,
    ) -> None:
        self.config = pipeline_config
        self.tracking_logger = tracking_logger
        self.target_scaler = target_scaler
        self.fold_i = fold_i

    @property
    def fold_postfix(self):
        return f"-{self.fold_i}" if self.fold_i is not None else ""

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict[str, Any]:
        """
        Train and evaluate
        """
        ...

    @abstractmethod
    def test(self, df: pd.DataFrame) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def global_step(self): ...

    @abstractmethod
    def log_model(self, export_onnx: bool): ...