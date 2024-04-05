from functools import singledispatch
from typing import Union, Callable
import numpy as np
import torch


NDType = Union[torch.Tensor, np.ndarray]


@singledispatch
def nptify(x: np.ndarray) -> Callable:
    """ Num-Py-Torch-i-FYing
    Make a type convertor based on the type of x

    Example:
    a = torch.Tensor([1])
    b = np.array([])
    To convert b to the type of a:
    b_out = nptify(a)(b)

    Returns:
        A Callable that converts its input to x's type
    """
    return lambda a: np.asarray(a)
@nptify.register
def _nptify(x: torch.Tensor):
    return lambda a: torch.as_tensor(a, dtype=x.dtype, device=x.device)


@singledispatch
def numpize(tensor: np.ndarray) -> np.ndarray:
    return tensor
@numpize.register
def _numpize(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

