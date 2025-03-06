import os
import random
from typing import Any, Union

import numpy as np
import torch

from collections import defaultdict


def seed_everything(seed=42, enable_deterministic: bool = True) -> None:
    """
    Set given seed in every random number generator available.
    :param seed: seed
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if enable_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def dicts_mean(dicts: list[dict[str, float]]) -> dict[str, float]:
    assert len(dicts) > 0
    r: dict[str, float] = defaultdict(float)  # type: ignore

    for d in dicts:
        for k in dicts[0].keys():
            r[k] += d[k]

    for k in dicts[0].keys():
        r[k] /= len(dicts)

    return r


def untensor_dict(dict: dict[str, Union[torch.Tensor, float]]) -> dict[str, float]:
    r = {}
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            r[k] = v.item()
        else:
            r[k] = v
    return r


def add_postfix(d: dict[str, Any], postfix) -> dict[str, Any]:
    r = {}
    for k, v in d.items():
        d[f"{k}{postfix}"] = v
    return r


def remove_in_keys(d: dict[str, Any], s: str) -> dict[str, Any]:
    r = {}
    for k, v in d.items():
        r[k.replace(s, "")] = v
    return r
