import copy
import json
import os

import cv2
import numpy as np
from dataloader import ego_pose_anno_loader
from PIL import Image
from projectaria_tools.core import calibration
from tqdm import tqdm
from utils.config import create_arg_parse
from utils.reader import PyAvReader
from utils.utils import get_aria_camera_models, get_ego_aria_cam_name


def undistort_aria_img(args):
    # Load all takes metadata
    takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

    for anno_type in args.anno_types:
        for split in args.splits:
            # Load GT annotation
            if split in ["train", "val"]:
                gt_anno_path = os.path.join(
                    args.gt_output_dir,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}.json",
                )
            else:
                gt_anno_path = os.path.join(
                    args.gt_output_dir,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}_private.json",
                )
            assert os.path.exists(
                gt_anno_path
            ), f"Extraction of aria raw image fails for split={split}. Invalid path: {gt_anno_path}"
            gt_anno = json.load(open(gt_anno_path))
            # Input and output root path
            vrs_root = os.path.join(args.ego4d_data_dir, "captures")
            dist_img_root = os.path.join(
                args.gt_output_dir, "image", "distorted", split
            )
            undist_img_root = os.path.join(
                args.gt_output_dir, "image", "undistorted", split
            )
            # Extract frames with annotations for all takes
            for take_uid, take_anno in gt_anno.items():
                # Get current take's metadata
                take = [t for t in takes if t["take_uid"] == take_uid]
                assert len(take) == 1, f"Take: {take_uid} does not exist"
                take = take[0]
                # Get current take's name and aria camera name
                take_name = take["root_dir"]
                ego_aria_cam_name = get_ego_aria_cam_name(take)
                # Get aria calibration model and pinhole camera model
                capture_name = "_".join(take_name.split("_")[:-1])
                vrs_path = os.path.join(
                    vrs_root, capture_name, f"videos/{ego_aria_cam_name}.vrs"
                )
                aria_rgb_calib = get_aria_camera_models(vrs_path)["214-1"]
                pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
                # Input and output directory
                curr_dist_img_dir = os.path.join(dist_img_root, take_name)
                assert os.path.exists(
                    curr_dist_img_dir
                ), f"{take_name} doesn't have extracted raw aria images yet."
                curr_undist_img_dir = os.path.join(undist_img_root, take_name)
                os.makedirs(curr_undist_img_dir, exist_ok=True)
                # Extract undistorted aria images
                for frame_number in tqdm(take_anno.keys(), total=len(take_anno.keys())):
                    f_idx = int(frame_number)
                    curr_undist_img_path = os.path.join(
                        curr_undist_img_dir, f"{f_idx:06d}.jpg"
                    )
                    if not os.path.exists(curr_undist_img_path):
                        # Load in distorted images
                        curr_dist_img_path = os.path.join(
                            curr_dist_img_dir, f"{f_idx:06d}.jpg"
                        )
                        curr_dist_image = np.array(
                            Image.open(curr_dist_img_path).rotate(90)
                        )
                        # Undistortion
                        undistorted_image = calibration.distort_by_calibration(
                            curr_dist_image, pinhole, aria_rgb_calib
                        )
                        undistorted_image = cv2.rotate(
                            undistorted_image, cv2.ROTATE_90_CLOCKWISE
                        )[:, :, ::-1]
                        # Save undistorted image
                        assert cv2.imwrite(
                            curr_undist_img_path, undistorted_image
                        ), curr_undist_img_path


def extract_aria_img(args):
    # Load all takes metadata
    takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

    for anno_type in args.anno_types:
        for split in args.splits:
            # Load GT annotation
            if split in ["train", "val"]:
                gt_anno_path = os.path.join(
                    args.gt_output_dir,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}.json",
                )
            else:
                gt_anno_path = os.path.join(
                    args.gt_output_dir,
                    "annotation",
                    anno_type,
                    f"ego_pose_gt_anno_{split}_private.json",
                )
            assert os.path.exists(
                gt_anno_path
            ), f"Extraction of aria raw image fails for split={split}. Invalid path: {gt_anno_path}"
            gt_anno = json.load(open(gt_anno_path))
            # Input and output root path
            take_video_dir = os.path.join(args.ego4d_data_dir, "takes")
            img_output_root = os.path.join(
                args.gt_output_dir, "image", "distorted", split
            )
            os.makedirs(img_output_root, exist_ok=True)
            # Extract frames with annotations for all takes
            for take_uid, take_anno in gt_anno.items():
                # Get current take's metadata
                take = [t for t in takes if t["take_uid"] == take_uid]
                assert len(take) == 1, f"Take: {take_uid} does not exist"
                take = take[0]
                # Get current take's name and aria camera name
                take_name = take["root_dir"]
                ego_aria_cam_name = get_ego_aria_cam_name(take)
                # Load current take's aria video
                curr_take_video_path = os.path.join(
                    take_video_dir,
                    take_name,
                    "frame_aligned_videos",
                    f"{ego_aria_cam_name}_214-1.mp4",
                )
                curr_take_img_output_path = os.path.join(img_output_root, take_name)
                os.makedirs(curr_take_img_output_path, exist_ok=True)
                reader = PyAvReader(
                    path=curr_take_video_path,
                    resize=None,
                    mean=None,
                    frame_window_size=1,
                    stride=1,
                    gpu_idx=-1,
                )
                # Extract frames
                for frame_number in tqdm(take_anno.keys(), total=len(take_anno.keys())):
                    f_idx = int(frame_number)
                    out_path = os.path.join(
                        curr_take_img_output_path, f"{f_idx:06d}.jpg"
                    )
                    if not os.path.exists(out_path):
                        frame = reader[f_idx][0].cpu().numpy()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        assert cv2.imwrite(out_path, frame), out_path


def save_test_gt_anno(output_dir, gt_anno_private):
    # 1. Save private annotated test JSON file
    with open(
        os.path.join(output_dir, f"ego_pose_gt_anno_test_private.json"), "w"
    ) as f:
        json.dump(gt_anno_private, f, indent=4)
    # 2. Exclude GT 3D joints and valid flag information for public un-annotated test file
    gt_anno_public = copy.deepcopy(gt_anno_private)
    for _, take_anno in gt_anno_public.items():
        for _, frame_anno in take_anno.items():
            for k in ["left_hand", "right_hand", "left_hand_valid", "right_hand_valid"]:
                frame_anno.pop(k)
    # 3. Save public un-annotated test JSON file
    with open(os.path.join(output_dir, f"ego_pose_gt_anno_test_public.json"), "w") as f:
        json.dump(gt_anno_public, f, indent=4)


def create_gt_anno(args):
    """
    Creates ground truth annotation file for train, val and test split. For
    test split creates two versions:
    - public: doesn't have GT 3D joints and valid flag information, used for
    public to do local inference
    - private: has GT 3D joints and valid flag information, used for server
    to evaluate model performance
    """
    for anno_type in args.anno_types:
        for split in args.splits:
            # Get ground truth annotation
            gt_anno = ego_pose_anno_loader(args, split, anno_type)
            gt_anno_output_dir = os.path.join(
                args.gt_output_dir, "annotation", anno_type
            )
            os.makedirs(gt_anno_output_dir, exist_ok=True)
            # Save ground truth JSON file
            if split in ["train", "val"]:
                with open(
                    os.path.join(gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json"),
                    "w",
                ) as f:
                    json.dump(gt_anno.db, f, indent=4)
            # For test split, create two versions of GT-anno
            else:
                save_test_gt_anno(gt_anno_output_dir, gt_anno.db)


def main(args):
    for step in args.steps:
        if step == "gt_anno":
            create_gt_anno(args)
        elif step == "raw_image":
            extract_aria_img(args)
        elif step == "undistorted_image":
            undistort_aria_img(args)
        else:
            raise Exception(f"Invalid step: {step}")


if __name__ == "__main__":
    args = create_arg_parse()
    main(args)
