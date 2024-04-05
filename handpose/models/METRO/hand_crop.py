""" Initialize Hand crop regions for METRO
"""

import argparse
import numpy as np
import os
import os.path as osp
import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import json
import cv2

from libzhifan.odlib import xyxy_to_xywh, xywh_to_xyxy
from libzhifan import odlib
odlib.setup(order='xywh', norm=True)

HOCROP_SIZE = 256

def arg_parse():
    parser = argparse.ArgumentParser("Ego-pose baseline model dataset preparation")

    parser.add_argument(
        "--storage_dir",
        type=str,
        nargs="+",
        default="/media/shan/Volume2/egoexo-challenge-03/metro_output",
        help="directory to save the output results",
    )

    parser.add_argument(
        "--gt_output_dir",
        type=str,
        default=None,
        help="Directory to store preprocessed ground truth annotation JSON file",
        required=True,
    )

    parser.add_argument("--hand_crop_size", type=int, default=256)
    args = parser.parse_args()

    return args


def square_bbox(bbox, pad=0):
    """
    Args:
        bbox: (N, 4) xyxy
        pad: args = parser.parse_args()pad ratio

    Returns:
        bbox: (N, 4) xyxy
    """
    if not torch.is_tensor(bbox):
        is_numpy = True
        bbox = torch.FloatTensor(bbox)
    else:
        is_numpy = False

    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    center = (x1y1 + x2y2) / 2
    half_w = torch.max((x2y2 - x1y1) / 2, dim=-1).values
    half_w = half_w * (1 + 2 * pad)
    half_w = half_w[:, None]
    bbox = torch.cat([center - half_w, center + half_w], dim=-1)
    if is_numpy:
        bbox = bbox.cpu().detach().numpy()
    return bbox


def square_bbox_xywh(bbox, pad_ratio=0.0):
    obj_bbox_xyxy = xywh_to_xyxy(bbox)
    obj_bbox_squared_xyxy = square_bbox(
        obj_bbox_xyxy, pad_ratio)
    obj_bbox_squared = xyxy_to_xywh(obj_bbox_squared_xyxy)
    return obj_bbox_squared


def square_expand_box(box1, box2):
    """
    Compute the max-bound for box1 and box2,
    First Square and then expand
    one of box1 or box2 can be None

    Args:
        box1 / box2: np.ndarray (4), x0y0wh
    Returns:
        squared_bbox: (4,)
    """
    if box1 is None and box2 is not None:
        bound_box = box2
    elif box1 is not None and box2 is None:
        bound_box = box1
    elif box1 is not None and box2 is not None:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1+w1, x2+w2)
        ymax = max(y1+h1, y2+h2)
        bound_box = np.array([xmin, ymin, xmax-xmin, ymax-ymin])
    else:
        raise ValueError("box1 and box2 cannot be both None")

    HOCROP_EXPAND_RATIO = 0.0
    box_squared = square_bbox_xywh(bound_box[None], pad_ratio=HOCROP_EXPAND_RATIO)[0]
    return box_squared


def crop_resize(image: PIL.Image,
                box: np.ndarray,
                out_size: int) -> PIL.Image:
    """ Pad 0's if box exceeds boundary.

    Args:
        image: (..., H, W)
        box: (4) xywh
        out_size: int

    Returns:
        img_crop: (..., crop_h, crop_w) in [0, 1]
    """
    x, y, w, h = box
    img_w, img_h = image.size
    pad_x = max(max(-x ,0), max(x+w-img_w, 0))
    pad_y = max(max(-y ,0), max(y+h-img_h, 0))
    pad_x = int(pad_x)
    pad_y = int(pad_y)
    transform = transforms.Compose([
        transforms.Pad((pad_x, pad_y))
    ])
        # [transforms.Pad([pad_x, pad_y])])
    x += pad_x
    y += pad_y

    image_pad = transform(image)
    crop_pil = F.resized_crop(
        image_pad,
        int(y), int(x), int(h), int(w), size=[out_size, out_size],
        # interpolation=transforms.InterpolationMode.NEAREST
        interpolation=Image.NEAREST
    )
    return crop_pil


def update_box(crop_box, box, out_size=HOCROP_SIZE):
    """
    Returns:
        new_box: (4,) the position of box w.r.t squared_bbox
    """
    if box is None:
        return None
    box[..., 0] = box[..., 0] - crop_box[0]  # x
    box[..., 1] = box[..., 1] - crop_box[1]  # y
    scale_w = out_size / crop_box[2]
    scale_h = out_size / crop_box[3]
    box[..., 0] = box[..., 0] * scale_w
    box[..., 1] = box[..., 1] * scale_h
    box[..., 2] = box[..., 2] * scale_w
    box[..., 3] = box[..., 3] * scale_h
    return box


def row2xyxy(r):
    return np.float32([r['x0'], r['y0'], r['x1'], r['y1']])

def run_hand_crops(bbox, args):

    storage_dir = args.storage_dir
    for bbox_take in bbox.keys():
        take_uid = bbox_take
        handcrop_seqs_dir = f'{storage_dir}/handcrops_gt_bbox/{take_uid}'
        os.makedirs(handcrop_seqs_dir, exist_ok=True)

        for bbox_frame in bbox[bbox_take].keys():
            take_name = bbox[bbox_take][bbox_frame]["metadata"]["take_name"]
            for hand_name in ["left", "right"]:
                if len(bbox[bbox_take][bbox_frame][hand_name+'_hand_bbox']) == 0:
                    continue
                hand_box = bbox[bbox_take][bbox_frame][hand_name+'_hand_bbox']
                # change to xywh
                hand_box = np.array(hand_box).astype(int)
                hand_box[2] = hand_box[2] - hand_box[0]
                hand_box[3] = hand_box[3] - hand_box[1]
                box_squared = square_expand_box(hand_box, box2=None)
                img_path = os.path.join(args.gt_output_dir, "image/undistorted/test", take_name, '{:06d}.jpg'.format(int(bbox_frame)))
                image = cv2.imread(img_path)
                # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image = Image.fromarray(np.uint8(image)).convert('RGB')
                handcrop = crop_resize(image, box_squared, HOCROP_SIZE)
                frame_name = '{}_{:06d}.jpg'.format(hand_name, int(bbox_frame))
                # handcrop = cv2.rotate(np.array(handcrop), cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(osp.join(handcrop_seqs_dir, frame_name), np.array(handcrop))


if __name__ == '__main__':

    args = arg_parse()

    # read bbox from annotation file
    bbox_annot_file = os.path.join(args.gt_output_dir, "annotation/manual/ego_pose_gt_anno_test_public.json")
    assert os.path.exists(bbox_annot_file), f"no file found {bbox_annot_file}"
    with open(bbox_annot_file) as f:
        bbox = json.load(f)

    run_hand_crops(bbox, args)
