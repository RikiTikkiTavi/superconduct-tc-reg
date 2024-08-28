from abc import ABC, abstractmethod
import logging
from re import split
from typing import Any, Iterator, Optional, Sequence

import sklearn.model_selection
import sklearn.utils
import numpy.random

_logger = logging.getLogger(__name__)


class FoldsGeneratorAlgorithm(ABC):

    def __init__(self, split_algorithm_seed: Optional[int] = None) -> None:
        self._split_algorithm_seed = split_algorithm_seed

    @abstractmethod
    def __call__(
        self,
        index: Sequence[int],
        targets: Optional[Sequence[int]] = None,
    ) -> Iterator[tuple[Sequence[int], Sequence[int]]]: ...

    @property
    @abstractmethod
    def n_folds(self) -> int: ...


class TrainValRandomSplitSingleFold(FoldsGeneratorAlgorithm):
    def __init__(self, val_size: float, stratified: bool, **kwargs):
        super().__init__(**kwargs)
        self._val_size = val_size
        self._stratified = stratified

    @property
    def n_folds(self):
        return 1

    def __call__(
        self, elements: Sequence[int], targets: Optional[Sequence[int]] = None
    ) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
        if self._stratified:
            assert targets is not None

        elements_train, elements_val = sklearn.model_selection.train_test_split(
            elements,
            test_size=self._val_size,
            random_state=self._split_algorithm_seed,
            stratify=targets,
        )  # type: ignore

        yield elements_train, elements_val
