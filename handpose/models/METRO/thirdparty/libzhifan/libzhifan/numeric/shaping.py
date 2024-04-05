import torch

def gather_ext(src: torch.Tensor, 
               index: torch.Tensor, 
               dim: int) -> torch.Tensor:
    """
    Args:
        src: shape (dim0, ..., dim_d_1, ..., dim_k, ..., dim_n)
        index: shape (dim0, ..., dim_d_2, ..., dim_k)
        dim: int
    Returns:
        gathered: (dim0, dim1, ..., dim_d_2, ... dim_n)
    """
    view_shape = index.shape + (1,) * (src.ndim - index.ndim)
    expand_shape = index.shape + src.shape[index.ndim:]
    index = index.view(view_shape).expand(expand_shape)
    return src.gather(dim=dim, index=index)