from multiprocessing.sharedctypes import Value
import numpy as np
import torch


""" Notations:

xyxy: x_min, y_min, x_max, y_max
xywh: x_min, y_min, width, height
xcycwh: x_center, y_center, width, height
yxyx: y_min, x_min, y_max, x_max
"""


def __concat_last(arr_list):
    if isinstance(arr_list[0], np.ndarray):
        return np.concatenate(arr_list, axis=-1)
    elif isinstance(arr_list[0], torch.Tensor):
        return torch.cat(arr_list, dim=-1)
    else:
        raise ValueError(f"type of arr_list not understood.")


def xyxy_to_xywh(boxes):
    """ 
    Args:
        boxes: (..., 4) in xyxy
    Returns:
        boxes: (..., 4) in xywh
    """
    x0 = boxes[..., [0]]
    y0 = boxes[..., [1]]
    x1 = boxes[..., [2]]
    y1 = boxes[..., [3]]
    return __concat_last([x0, y0, x1-x0, y1-y0])


def xywh_to_xyxy(boxes):
    """ 
    Args:
        boxes: (..., 4) in xywh
    Returns:
        boxes: (..., 4) in xyxy
    """
    x0 = boxes[..., [0]]
    y0 = boxes[..., [1]]
    w = boxes[..., [2]]
    h = boxes[..., [3]]
    return __concat_last([x0, y0, x0+w, y0+h])


def xyxy_to_yxyx(boxes):
    """ 
    Args:
        boxes: (..., 4) in xyxy
    Returns:
        boxes: (..., 4) in yxyx
    """
    x0 = boxes[..., [0]]
    y0 = boxes[..., [1]]
    x1 = boxes[..., [2]]
    y1 = boxes[..., [3]]
    return __concat_last([y0, x0, y1, x1])

def yxyx_to_xyxy(boxes):
    """ 
    Args:
        boxes: (..., 4) in xyxy
    Returns:
        boxes: (..., 4) in yxyx
    """
    # This is just flipping x and y
    return xyxy_to_yxyx(boxes)


def xyxy_to_xcycwh(boxes):
    """ 
    Args:
        boxes: (..., 4) in xyxy
    Returns:
        boxes: (..., 4) in xcycwh
    """
    x0 = boxes[..., [0]]
    y0 = boxes[..., [1]]
    x1 = boxes[..., [2]]
    y1 = boxes[..., [3]]
    return __concat_last([(x1+x0)/2, (y1+y0)/2, x1-x0, y1-y0])

def xcycwh_to_xyxy(boxes):
    """
    Args:
        boxes: (..., 4) in xcycwh
    Returns:
        boxes: (..., 4) in xyxy
    """
    # xcycwh -> xywh -> xyxy
    return xywh_to_xyxy(xcycwh_to_xywh(boxes))

def xcycwh_to_xywh(boxes):
    """
    Args:
        boxes: (..., 4) in xcycwh
    Returns:
        boxes: (..., 4) in xyxy
    """
    xc = boxes[..., [0]]
    yc = boxes[..., [1]]
    w = boxes[..., [2]]
    h = boxes[..., [3]]
    return __concat_last([xc-w/2, yc-h/2, w, h])

def xywh_to_xcycwh(boxes):
    """
    Args:
        boxes: (..., 4) in xywh
    Returns:
        boxes: (..., 4) in xcycwh
    """
    # xywh -> xywh -> xcycwh
    return xyxy_to_xcycwh(xywh_to_xyxy(boxes))

