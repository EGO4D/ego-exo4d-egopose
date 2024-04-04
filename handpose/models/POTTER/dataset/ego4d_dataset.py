import copy
import glob
import json
import os

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class ego4dDataset(Dataset):
    """
    Load Ego4D dataset with only Ego(Aria) images for 3D hand pose estimation
    Reference: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    """

    def __init__(self, args, cfg, split, transform=None):
        self.split = split
        self.num_joints = cfg.MODEL.NUM_JOINTS  # Number of joints for one single hand
        self.pixel_std = 200  # Pixel std to define scale factor for image resizing
        self.undist_img_dim = np.array(
            cfg.DATASET.ORIGINAL_IMAGE_SIZE
        )  # Size of undistorted aria image
        self.image_size = np.array(
            cfg.MODEL.IMAGE_SIZE
        )  # Size of input image to the model
        gt_anno_path = os.path.join(
            args.gt_anno_dir, f"ego_pose_gt_anno_{self.split}_public.json"
        )
        self.img_dir = os.path.join(args.aria_img_dir, split)
        self.db = self.load_all_data(gt_anno_path)
        self.pred_temp = self.generate_pred_temp(gt_anno_path)
        self.transform = transform
        self.joint_mean = np.array(
            [
                [0.0000000e00, 0.0000000e00, 0.0000000e00],
                [-3.9501650e00, -8.6685377e-01, 2.4517984e01],
                [-1.3187613e01, 1.2967486e00, 4.7673504e01],
                [-2.2936522e01, 1.5275195e00, 7.2566208e01],
                [-3.1109295e01, 1.9404153e00, 9.5952751e01],
                [-4.8375599e01, 4.6012049e00, 6.7085617e01],
                [-5.9843365e01, 5.9568534e00, 9.3948418e01],
                [-5.7148232e01, 5.7935758e00, 1.1097713e02],
                [-5.1052166e01, 4.9937048e00, 1.2502338e02],
                [-5.1586624e01, 2.5471370e00, 7.2120811e01],
                [-6.5926834e01, 3.0671554e00, 9.8404510e01],
                [-6.1979191e01, 2.8341565e00, 1.1610429e02],
                [-5.4618130e01, 2.5274558e00, 1.2917862e02],
                [-4.6503471e01, 3.3559692e-01, 7.3062035e01],
                [-5.9186893e01, 2.6649246e-02, 9.6192421e01],
                [-5.6693432e01, -8.4625520e-02, 1.1205978e02],
                [-5.1260197e01, 3.4378145e-02, 1.2381713e02],
                [-3.5775276e01, -1.0368422e00, 7.0583588e01],
                [-4.3695080e01, -1.9620019e00, 8.8694397e01],
                [-4.4897186e01, -2.6101866e00, 1.0119468e02],
                [-4.4571526e01, -3.3564034e00, 1.1180748e02],
            ]
        )
        self.joint_std = np.array(
            [
                [0.0, 0.0, 0.0],
                [17.266953, 44.075836, 14.078445],
                [24.261362, 65.793236, 18.580193],
                [25.479671, 74.18796, 19.767653],
                [30.458921, 80.729996, 23.553158],
                [21.826715, 45.61571, 18.80888],
                [26.570208, 54.434124, 19.955523],
                [30.757236, 60.084938, 23.375763],
                [35.174015, 64.042404, 31.206692],
                [21.586899, 28.31489, 16.090088],
                [29.26384, 35.83172, 18.48644],
                [35.396465, 40.93173, 26.987226],
                [40.40074, 45.358475, 37.419308],
                [20.73408, 21.591717, 14.190551],
                [28.290194, 27.946808, 18.350618],
                [34.42277, 31.388414, 28.024563],
                [39.819054, 35.205494, 38.80897],
                [19.79841, 29.38799, 14.820373],
                [26.476702, 34.7448, 20.027615],
                [31.811651, 37.06962, 27.742807],
                [36.893555, 38.98199, 36.001797],
            ]
        )

    def __getitem__(self, idx):
        """
        Return transformed images, normalized & offset 3D hand GT pose, valid hand joint flag and metadata.
        """
        curr_db = copy.deepcopy(self.db[idx])

        # Define parameters for affine transformation of hand image
        c, s = xyxy2cs(*curr_db["bbox"], self.undist_img_dim, self.pixel_std)
        r = 0
        trans = get_affine_transform(c, s, r, self.image_size)
        # Load image
        metadata = curr_db["metadata"]
        img_path = os.path.join(
            self.img_dir,
            f"{metadata['take_name']}",
            f"{metadata['frame_number']:06d}.jpg",
        )
        img = imageio.imread(img_path, pilmode="RGB")
        # Get affine transformed hand image
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR,
        )
        # Apply Pytorch transform if needed
        if self.transform:
            input = self.transform(input)

        # Return only input if split is test
        if self.split == "test":
            return input, curr_db["metadata"]

        # Load ground truth 3D hand joints and valid flag info for train and val, omit for test
        curr_3d_kpts_cam = curr_db["joints_3d"]
        curr_3d_kpts_cam = curr_3d_kpts_cam * 1000  # m to mm
        curr_3d_kpts_cam_offset = curr_3d_kpts_cam - curr_3d_kpts_cam[0]
        # Normalization
        curr_3d_kpts_cam_offset = (curr_3d_kpts_cam_offset - self.joint_mean) / (
            self.joint_std + 1e-8
        )
        curr_3d_kpts_cam_offset[~curr_db["valid_flag"]] = None
        curr_3d_kpts_cam_offset = torch.from_numpy(
            curr_3d_kpts_cam_offset.astype(np.float32)
        )
        # Generate valid joints flag
        vis_flag = torch.from_numpy(curr_db["valid_flag"])

        # Record meta info for later reprojection
        meta = curr_db["metadata"]

        return input, curr_3d_kpts_cam_offset, vis_flag, meta

    def __len__(self):
        return len(self.db)

    def load_all_data(self, gt_anno_path):
        """
        Store each valid hand's annotation per frame separately, with
        dict key based on split:
        Train & val:
            - joints_3d
            - valid_flag
            - bbox
            - metadata
        Test:
            - bbox
            - metadata
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))

        # Load gt annotation for train & val
        if self.split in ["train", "val"]:
            all_frame_anno = []
            for _, curr_take_anno in gt_anno.items():
                for _, curr_f_anno in curr_take_anno.items():
                    # check image existence
                    image_path = os.path.join(self.img_dir,
                                              curr_f_anno['metadata']['take_name'],
                                              '{:06d}.jpg'.format(curr_f_anno['metadata']['frame_number']))
                    if not os.path.exists(image_path):
                        continue
                    for hand_order in ["right", "left"]:
                        single_hand_anno = {}
                        if len(curr_f_anno[f"{hand_order}_hand_3d"]) != 0:
                            single_hand_anno["joints_3d"] = np.array(
                                curr_f_anno[f"{hand_order}_hand_3d"]
                            )
                            single_hand_anno["valid_flag"] = np.array(
                                curr_f_anno[f"{hand_order}_hand_valid_3d"]
                            )
                            single_hand_anno["bbox"] = np.array(
                                curr_f_anno[f"{hand_order}_hand_bbox"]
                            )
                            single_hand_anno["metadata"] = curr_f_anno["metadata"]
                            all_frame_anno.append(single_hand_anno)
        # Load un-annotated test JSON file for evaluation
        else:
            all_frame_anno = []
            for _, curr_take_anno in gt_anno.items():
                for _, curr_f_anno in curr_take_anno.items():
                    for hand_order in ["right", "left"]:
                        single_hand_anno = {}
                        if len(curr_f_anno[f"{hand_order}_hand_bbox"]) != 0:
                            # Load bbox regardless of whether it's empty or not
                            single_hand_anno["bbox"] = np.array(
                                curr_f_anno[f"{hand_order}_hand_bbox"]
                            )
                            single_hand_anno["metadata"] = copy.deepcopy(
                                curr_f_anno["metadata"]
                            )
                            single_hand_anno["metadata"]["hand_order"] = hand_order
                            all_frame_anno.append(single_hand_anno)
        return all_frame_anno

    def generate_pred_temp(self, gt_anno_path):
        """
        Generate empty prediction template with specicifed JSON format:
        {
            "<take_uid>": {
                "frame_number": {
                        "left_hand_3d":  [],
                        "right_hand_3d": []
                }
            }
        }
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))
        # Create empty template for each frame in each take
        pred_temp = {}
        for take_uid, take_anno in gt_anno.items():
            curr_take_pred_temp = {}
            for frame_number in take_anno.keys():
                curr_take_pred_temp[frame_number] = {"left_hand_3d": [], "right_hand_3d": []}
            pred_temp[take_uid] = curr_take_pred_temp
        return pred_temp

    def init_split(self):
        # Get tain/val/test df
        train_df = self.takes_df[self.takes_df["split"] == "TRAIN"]
        val_df = self.takes_df[self.takes_df["split"] == "VAL"]
        test_df = self.takes_df[self.takes_df["split"] == "TEST"]
        # Get train/val/test uid
        all_train_uid = list(train_df["take_uid"])
        all_val_uid = list(val_df["take_uid"])
        all_test_uid = list(test_df["take_uid"])
        return {"train": all_train_uid, "val": all_val_uid, "test": all_test_uid}


def xyxy2cs(x1, y1, x2, y2, img_shape, pixel_std):
    aspect_ratio = img_shape[1] * 1.0 / img_shape[0]

    center = np.zeros((2), dtype=np.float32)
    center[0] = (x1 + x2) / 2
    center[1] = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
