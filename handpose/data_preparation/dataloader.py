import json
import os

import cv2
import numpy as np
import pandas as pd
from projectaria_tools.core import calibration
from utils.utils import (
    aria_original_to_extracted,
    cam_to_img,
    get_aria_camera_models,
    get_interested_take,
    HAND_ORDER,
    hand_pad_bbox_from_kpts,
    hand_rand_bbox_from_kpts,
    joint_dist_angle_check,
    world_to_cam,
    xywh2xyxy,
)


class ego_pose_anno_loader:
    """
    Load Ego4D data and create ground truth annotation JSON file for Ego-pose baseline model
    """

    def __init__(self, args, split, anno_type):
        # Set dataloader parameters
        self.dataset_root = args.ego4d_data_dir
        self.anno_type = anno_type
        self.split = split
        self.num_joints = 21  # Number of joints for single hand
        self.undist_img_dim = (512, 512)  # Dimension of undistorted aria image [H, W]
        self.valid_kpts_threshold = (
            args.valid_kpts_num_thresh
        )  # Threshold of minimum number of valid kpts in single hand
        self.bbox_padding = (
            args.bbox_padding
        )  # Amount of pixels to pad around kpts to find bbox
        self.takes = json.load(open(os.path.join(self.dataset_root, "takes.json")))

        # Determine annotation sub-directory
        anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}
        self.hand_anno_dir = os.path.join(
            self.dataset_root,
            "annotations/ego_pose/hand",
            anno_type_dir_dict[self.anno_type],
        )
        self.cam_pose_dir = os.path.join(
            self.dataset_root, "annotations/ego_pose/hand/camera_pose"
        )
        self.all_take_uid = [k[:-5] for k in os.listdir(self.hand_anno_dir)]
        self.take_to_uid = {
            t["root_dir"]: t["take_uid"]
            for t in self.takes
            if t["take_uid"] in self.all_take_uid
        }
        self.uid_to_take = {uid: take for take, uid in self.take_to_uid.items()}
        self.takes_df = pd.read_csv(
            os.path.join(
                self.dataset_root, "annotations/egoexo_split_latest_train_val_test.csv"
            )
        )
        # Whether use extracted view (Default is False)
        self.extracted_view = args.extracted_view
        # TODO: Modify as needed with updated public egoexo data
        self.split_take_dict = self.init_split()
        self.db = self.load_raw_data()

    def load_raw_data(self):
        gt_db = {}

        # Get all valid local take uids that are used in current split
        curr_split_uid = self.split_take_dict[self.split]
        common_take_uid = list(set(self.all_take_uid) & set(self.takes_df["take_uid"]))
        available_cam_pose_uid = [k[:-5] for k in os.listdir(self.cam_pose_dir)]
        comm_take_w_cam_pose = list(set(common_take_uid) & set(available_cam_pose_uid))
        all_interested_scenario_uid, _ = get_interested_take(
            comm_take_w_cam_pose, self.takes_df
        )
        available_curr_split_uid = list(
            set(curr_split_uid) & set(all_interested_scenario_uid)
        )
        print(
            f"Trying to use {len(available_curr_split_uid)} takes in {self.split} dataset"
        )

        # Iterate through all takes from annotation directory and check
        for curr_take_uid in available_curr_split_uid:
            curr_take_name = self.uid_to_take[curr_take_uid]
            # Load annotation, camera pose JSON and image directory
            curr_take_anno_path = os.path.join(
                self.hand_anno_dir, f"{curr_take_uid}.json"
            )
            curr_take_cam_pose_path = os.path.join(
                self.cam_pose_dir, f"{curr_take_uid}.json"
            )
            # Load in annotation JSON and image directory
            curr_take_anno = json.load(open(curr_take_anno_path))
            curr_take_cam_pose = json.load(open(curr_take_cam_pose_path))

            # Get valid takes info for all frames
            if len(curr_take_anno) > 0:
                aria_mask, aria_cam_name = self.load_aria_calib(curr_take_name)
                curr_take_data = self.load_take_raw_data(
                    curr_take_name,
                    curr_take_uid,
                    curr_take_anno,
                    curr_take_cam_pose,
                    aria_cam_name,
                    aria_mask,
                )
                # Append into dataset if has at least valid annotation
                if len(curr_take_data) > 0:
                    gt_db[curr_take_uid] = curr_take_data
        return gt_db

    def load_take_raw_data(
        self,
        take_name,
        take_uid,
        anno,
        cam_pose,
        aria_cam_name,
        aria_mask,
    ):
        curr_take_db = {}

        for frame_idx, curr_frame_anno in anno.items():
            # Load in current frame's 2D & 3D annotation and camera parameter
            curr_hand_2d_kpts, curr_hand_3d_kpts, _ = self.load_frame_hand_2d_3d_kpts(
                curr_frame_anno, aria_cam_name
            )
            curr_intri, curr_extri = self.load_frame_cam_pose(
                frame_idx, cam_pose, aria_cam_name
            )
            # Skip this frame if missing valid data
            if curr_hand_3d_kpts is None or curr_intri is None or curr_extri is None:
                continue
            # Look at each hand in current frame
            curr_frame_anno = {}
            at_least_one_hands_valid = False
            for hand_idx, hand_name in enumerate(HAND_ORDER):
                # Get current hand's 2D kpts and 3D world kpts
                start_idx, end_idx = self.num_joints * hand_idx, self.num_joints * (
                    hand_idx + 1
                )
                one_hand_2d_kpts = curr_hand_2d_kpts[start_idx:end_idx]
                if self.extracted_view:
                    one_hand_2d_kpts = aria_original_to_extracted(
                        one_hand_2d_kpts, self.undist_img_dim
                    )
                one_hand_3d_kpts_world = curr_hand_3d_kpts[start_idx:end_idx]
                # Skip this hand if the hand wrist (root) is None
                if np.any(np.isnan(one_hand_3d_kpts_world[0])):
                    one_hand_3d_kpts_world[:, :] = None

                # Hand biomechanical structure check for train and val
                if self.split != "test":
                    one_hand_3d_kpts_world = joint_dist_angle_check(
                        one_hand_3d_kpts_world
                    )
                # 3D world to camera (original view)
                one_hand_3d_kpts_cam = world_to_cam(one_hand_3d_kpts_world, curr_extri)
                # Camera original to original aria image plane
                one_hand_proj_2d_kpts = cam_to_img(one_hand_3d_kpts_cam, curr_intri)
                if self.extracted_view:
                    one_hand_proj_2d_kpts = aria_original_to_extracted(
                        one_hand_proj_2d_kpts, self.undist_img_dim
                    )

                # Filter projected 2D kpts
                (
                    one_hand_filtered_proj_2d_kpts,
                    valid_proj_2d_flag,
                ) = self.one_hand_kpts_valid_check(one_hand_proj_2d_kpts, aria_mask)
                # Get filtered 3D kpts in camera original view
                one_hand_filtered_3d_kpts_cam = one_hand_3d_kpts_cam.copy()
                one_hand_filtered_3d_kpts_cam[~valid_proj_2d_flag] = None

                # Filter 2D annotation kpts
                one_hand_filtered_anno_2d_kpts, _ = self.one_hand_kpts_valid_check(
                    one_hand_2d_kpts, aria_mask
                )

                # Prepare 2d kpts, 3d kpts, bbox and flag data based on number of valid 3D kpts
                if sum(valid_proj_2d_flag) >= self.valid_kpts_threshold:
                    at_least_one_hands_valid = True
                    # Assign original hand wrist 3d kpts back (needed for offset hand wrist)
                    one_hand_filtered_3d_kpts_cam[0] = one_hand_3d_kpts_cam[0]
                    # Generate hand bbox based on 2D GT kpts
                    if self.split == "test":
                        one_hand_bbox = hand_rand_bbox_from_kpts(
                            one_hand_filtered_proj_2d_kpts[valid_proj_2d_flag],
                            self.undist_img_dim,
                        )
                    else:
                        # For train and val, generate hand bbox with padding
                        one_hand_bbox = hand_pad_bbox_from_kpts(
                            one_hand_filtered_proj_2d_kpts[valid_proj_2d_flag],
                            self.undist_img_dim,
                            self.bbox_padding,
                        )
                # If no valid annotation for current hand, assign empty bbox, anno and valid flag
                else:
                    one_hand_bbox = np.array([])
                    one_hand_filtered_3d_kpts_cam = np.array([])
                    one_hand_filtered_anno_2d_kpts = np.array([])
                    valid_proj_2d_flag = np.array([])

                # Compose current hand GT info in current frame
                curr_frame_anno[
                    f"{hand_name}_hand_3d"
                ] = one_hand_filtered_3d_kpts_cam.tolist()
                curr_frame_anno[
                    f"{hand_name}_hand_2d"
                ] = one_hand_filtered_anno_2d_kpts.tolist()
                curr_frame_anno[f"{hand_name}_hand_bbox"] = one_hand_bbox.tolist()
                curr_frame_anno[f"{hand_name}_hand_valid"] = valid_proj_2d_flag.tolist()

            # Append current frame into GT JSON if at least one valid hand exists
            if at_least_one_hands_valid:
                metadata = {
                    "take_uid": take_uid,
                    "take_name": take_name,
                    "frame_number": int(frame_idx),
                }
                curr_frame_anno["metadata"] = metadata
                curr_take_db[frame_idx] = curr_frame_anno

        return curr_take_db

    # TODO: Correct as needed after egoexo public ego-pose split is updated
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

    def load_aria_calib(self, curr_take_name):
        # Load aria calibration model
        capture_name = "_".join(curr_take_name.split("_")[:-1])
        # Find aria names
        take = [t for t in self.takes if t["root_dir"] == curr_take_name]
        take = take[0]
        ego_cam_names = [
            x["cam_id"]
            for x in take["capture"]["cameras"]
            if str(x["is_ego"]).lower() == "true"
        ]
        assert len(ego_cam_names) > 0, "No ego cameras found!"
        if len(ego_cam_names) > 1:
            ego_cam_names = [
                cam
                for cam in ego_cam_names
                if cam in take["frame_aligned_videos"].keys()
            ]
            assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
            if len(ego_cam_names) > 1:
                ego_cam_names_filtered = [
                    cam for cam in ego_cam_names if "aria" in cam.lower()
                ]
                if len(ego_cam_names_filtered) == 1:
                    ego_cam_names = ego_cam_names_filtered
            assert (
                len(ego_cam_names) == 1
            ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"
        ego_cam_names = ego_cam_names[0]
        # Load aria calibration model
        vrs_path = os.path.join(
            self.dataset_root, "captures", capture_name, f"videos/{ego_cam_names}.vrs"
        )
        aria_rgb_calib = get_aria_camera_models(vrs_path)["214-1"]
        dst_cam_calib = calibration.get_linear_camera_calibration(512, 512, 150)
        # Generate mask in undistorted aria view
        mask = np.full((1408, 1408), 255, dtype=np.uint8)
        undistorted_mask = calibration.distort_by_calibration(
            mask, dst_cam_calib, aria_rgb_calib
        )
        undistorted_mask = (
            cv2.rotate(undistorted_mask, cv2.ROTATE_90_CLOCKWISE)
            if self.extracted_view
            else undistorted_mask
        )
        undistorted_mask = undistorted_mask / 255
        return undistorted_mask, ego_cam_names

    def load_frame_hand_2d_3d_kpts(self, frame_anno, aria_cam_name):
        """
        Input:
            frame_anno: annotation for current frame
            aria_cam_name: aria camera name
        Output:
            curr_frame_2d_kpts: (42,2) 2D hand keypoints in original frame
            curr_frame_3d_kpts: (42,3) 3D hand keypoints in world coordinate system
            joints_view_stat: (42,) Number of triangulation views for each 3D hand keypoints
        """
        # Finger dict to load annotation
        finger_dict = {
            "wrist": None,
            "thumb": [1, 2, 3, 4],
            "index": [1, 2, 3, 4],
            "middle": [1, 2, 3, 4],
            "ring": [1, 2, 3, 4],
            "pinky": [1, 2, 3, 4],
        }

        ### Load 2D GT hand kpts ###
        # Return NaN if no annotation exists
        if (
            "annotation2D" not in frame_anno[0].keys()
            or aria_cam_name not in frame_anno[0]["annotation2D"].keys()
            or len(frame_anno[0]["annotation2D"][aria_cam_name]) == 0
        ):
            curr_frame_2d_kpts = [[None, None] for _ in range(42)]
        else:
            curr_frame_2d_anno = frame_anno[0]["annotation2D"][aria_cam_name]
            curr_frame_2d_kpts = []
            # Load 3D annotation for both hands
            for hand in HAND_ORDER:
                for finger, finger_joint_order in finger_dict.items():
                    if finger_joint_order:
                        for finger_joint_idx in finger_joint_order:
                            finger_k_json = f"{hand}_{finger}_{finger_joint_idx}"
                            # Load 3D if exist annotation, and check for minimum number of visible views
                            if finger_k_json in curr_frame_2d_anno.keys():
                                curr_frame_2d_kpts.append(
                                    [
                                        curr_frame_2d_anno[finger_k_json]["x"],
                                        curr_frame_2d_anno[finger_k_json]["y"],
                                    ]
                                )
                            else:
                                curr_frame_2d_kpts.append([None, None])
                    else:
                        finger_k_json = f"{hand}_{finger}"
                        # Load 3D if exist annotation, and check for minimum number of visible views
                        if finger_k_json in curr_frame_2d_anno.keys():
                            curr_frame_2d_kpts.append(
                                [
                                    curr_frame_2d_anno[finger_k_json]["x"],
                                    curr_frame_2d_anno[finger_k_json]["y"],
                                ]
                            )
                        else:
                            curr_frame_2d_kpts.append([None, None])

        ### Load 3D GT hand kpts ###
        # Return NaN if no annotation exists
        if (
            "annotation3D" not in frame_anno[0].keys()
            or len(frame_anno[0]["annotation3D"]) == 0
        ):
            return None, None, None
        else:
            curr_frame_3d_anno = frame_anno[0]["annotation3D"]
            curr_frame_3d_kpts = []
            joints_view_stat = []
            # Load 3D annotation for both hands
            for hand in HAND_ORDER:
                for finger, finger_joint_order in finger_dict.items():
                    if finger_joint_order:
                        for finger_joint_idx in finger_joint_order:
                            finger_k_json = f"{hand}_{finger}_{finger_joint_idx}"
                            # Load 3D if exist annotation, and check for minimum number of visible views
                            if (
                                finger_k_json in curr_frame_3d_anno.keys()
                                and curr_frame_3d_anno[finger_k_json][
                                    "num_views_for_3d"
                                ]
                                >= 3
                            ):
                                curr_frame_3d_kpts.append(
                                    [
                                        curr_frame_3d_anno[finger_k_json]["x"],
                                        curr_frame_3d_anno[finger_k_json]["y"],
                                        curr_frame_3d_anno[finger_k_json]["z"],
                                    ]
                                )
                                joints_view_stat.append(
                                    curr_frame_3d_anno[finger_k_json][
                                        "num_views_for_3d"
                                    ]
                                )
                            else:
                                curr_frame_3d_kpts.append([None, None, None])
                                joints_view_stat.append(None)
                    else:
                        finger_k_json = f"{hand}_{finger}"
                        # Load 3D if exist annotation, and check for minimum number of visible views
                        if (
                            finger_k_json in curr_frame_3d_anno.keys()
                            and curr_frame_3d_anno[finger_k_json]["num_views_for_3d"]
                            >= 3
                        ):
                            curr_frame_3d_kpts.append(
                                [
                                    curr_frame_3d_anno[finger_k_json]["x"],
                                    curr_frame_3d_anno[finger_k_json]["y"],
                                    curr_frame_3d_anno[finger_k_json]["z"],
                                ]
                            )
                            joints_view_stat.append(
                                curr_frame_3d_anno[finger_k_json]["num_views_for_3d"]
                            )
                        else:
                            curr_frame_3d_kpts.append([None, None, None])
                            joints_view_stat.append(None)
        return (
            np.array(curr_frame_2d_kpts).astype(np.float32),
            np.array(curr_frame_3d_kpts).astype(np.float32),
            np.array(joints_view_stat).astype(np.float32),
        )

    def load_frame_cam_pose(self, frame_idx, cam_pose, aria_cam_name):
        # Check if current frame has corresponding camera pose
        if (
            aria_cam_name not in cam_pose.keys()
            or "camera_intrinsics" not in cam_pose[aria_cam_name].keys()
            or "camera_extrinsics" not in cam_pose[aria_cam_name].keys()
            or frame_idx not in cam_pose[aria_cam_name]["camera_extrinsics"].keys()
        ):
            return None, None
        # Build camera projection matrix
        curr_cam_intrinsic = np.array(
            cam_pose[aria_cam_name]["camera_intrinsics"]
        ).astype(np.float32)
        curr_cam_extrinsics = np.array(
            cam_pose[aria_cam_name]["camera_extrinsics"][frame_idx]
        ).astype(np.float32)
        return curr_cam_intrinsic, curr_cam_extrinsics

    def one_hand_kpts_valid_check(self, kpts, aria_mask):
        """
        Return valid kpts with three checks:
            - Has valid kpts
            - Within image bound
            - Visible within aria mask
        Input:
            kpts: (21,2) raw single 2D hand kpts
            aria_mask: (H,W) binary mask that has same shape as undistorted aria image
        Output:
            new_kpts: (21,2)
            flag: (21,)
        """
        new_kpts = kpts.copy()
        # 1. Check missing annotation kpts
        miss_anno_flag = np.any(np.isnan(kpts), axis=1)
        new_kpts[miss_anno_flag] = 0
        # 2. Check out-bound annotation kpts
        x_out_bound = np.logical_or(
            new_kpts[:, 0] < 0, new_kpts[:, 0] >= self.undist_img_dim[1]
        )
        y_out_bound = np.logical_or(
            new_kpts[:, 1] < 0, new_kpts[:, 1] >= self.undist_img_dim[0]
        )
        out_bound_flag = np.logical_or(x_out_bound, y_out_bound)
        new_kpts[out_bound_flag] = 0
        # 3. Check in-bound but invisible kpts
        invis_flag = (
            aria_mask[new_kpts[:, 1].astype(np.int64), new_kpts[:, 0].astype(np.int64)]
            == 0
        )
        # 4. Get valid flag
        invalid_flag = miss_anno_flag + out_bound_flag + invis_flag
        valid_flag = ~invalid_flag
        # 5. Assign invalid kpts as None
        new_kpts[invalid_flag] = None

        return new_kpts, valid_flag
