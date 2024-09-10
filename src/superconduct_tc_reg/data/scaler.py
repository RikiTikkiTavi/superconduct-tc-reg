from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from pydantic import InstanceOf
from sklearn.preprocessing import (
    MinMaxScaler as _MinMaxScaler,
    StandardScaler as _StandardScaler,
)
import pandas as pd
import torch

_XT = TypeVar("_XT", pd.DataFrame, torch.Tensor)


class Scaler(ABC):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def inverse_transform(self, x: _XT, name: Optional[str] = None) -> _XT: ...