from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler as _MinMaxScaler,
    StandardScaler as _StandardScaler,
)
import pandas as pd
import torch

_XT = TypeVar("_XT", np.ndarray, torch.Tensor)


class TargetScaler(ABC):
    @abstractmethod
    def fit_transform(self, x: _XT) -> _XT: ...

    @abstractmethod
    def inverse_transform(self, x: _XT) -> _XT: ...


class StandardTargetScaler(TargetScaler):
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mean = np.mean(x)
        self.var = np.var(x)
        return (x - self.mean) / self.var

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.var + self.mean


class MinMaxTargetScaler(TargetScaler):
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.min = np.min(x)
        self.max = np.max(x)
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.max - self.min) + self.min


class ExpTargetScaler(TargetScaler):
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x * 0.1515) - 1.5543

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + 1.5543) / 0.1515
