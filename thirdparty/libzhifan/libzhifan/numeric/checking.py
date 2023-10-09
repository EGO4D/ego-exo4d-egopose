import torch
import numpy as np


def check_shape(tensor, shape, name=""):
    """ From libyana

    Args:
        tensor ([torch.Tensor,np.ndarray]): array to verify
        shape (tuple): allowed shapes, -1 encodes any size
            (-1, 3) is valid for any 2 dimensional array with
            second dimension of size 3
        name (str): name of the tensor
    """
    if isinstance(tensor, torch.Tensor):
        tens_size = tensor.dim()
        type_name = "torch tensor"
    elif isinstance(tensor, np.ndarray):
        tens_size = tensor.ndim
        type_name = "np array"
    else:
        raise ValueError(
            f"Expected {name} to be torch or numpy array, "
            f"got {type(tensor)}"
        )
    if tens_size != len(shape):
        raise ValueError(
            f"Expected {type_name} {name} of shape {shape}"
            f", got {tensor.shape}"
        )
    for dim_idx, dim_shape in enumerate(shape):
        if dim_shape not in (-1, tensor.shape[dim_idx]):
            raise ValueError(
                f"Expected {type_name} {name} of shape {shape}"
                f", got {tensor.shape}"
            )


def check_shape_like(tensor, other):
    """
    Args:
        tensor ([torch.Tensor,np.ndarray]): array to verify
        other ([torch.Tensor,np.ndarray]): array to verify
    """
    shape = tuple(other.shape)
    check_shape(tensor, shape)
