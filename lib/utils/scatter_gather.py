# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor
from torch.nn.parallel._functions import Scatter as OrigScatter

from .data_container import DataContainer

ScatterInputs = Union[Tensor, DataContainer, tuple, list, dict]


def scatter(inputs: ScatterInputs,
            target_gpus: List[int],
            dim: int = 0) -> list:
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, Tensor):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            return obj.data
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore


def scatter_kwargs(inputs: ScatterInputs,
                   target_gpus: List[int],
                   dim: int = 0) -> Tuple[tuple, tuple]:
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    inputs = tuple(inputs)
    return inputs
