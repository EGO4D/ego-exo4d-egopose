import numpy as np


# Customized order when processing each hand's annotation
HAND_ORDER = ["right", "left"]


def get_aria_camera_models(aria_path):
    try:
        from projectaria_tools.core import data_provider

        vrs_data_provider = data_provider.create_vrs_data_provider(aria_path)
        aria_camera_model = vrs_data_provider.get_device_calibration()
        slam_left = aria_camera_model.get_camera_calib("camera-slam-left")
        slam_right = aria_camera_model.get_camera_calib("camera-slam-right")
        rgb_cam = aria_camera_model.get_camera_calib("camera-rgb")
    except Exception as e:
        print(
            f"[Warning] Hitting exception {e}. Fall back to old projectaria_tools ..."
        )
        import projectaria_tools

        vrs_data_provider = projectaria_tools.dataprovider.AriaVrsDataProvider()
        vrs_data_provider.openFile(aria_path)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(214, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 1)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        aria_stream_id = projectaria_tools.dataprovider.StreamId(1201, 2)
        vrs_data_provider.setStreamPlayer(aria_stream_id)
        vrs_data_provider.readFirstConfigurationRecord(aria_stream_id)

        assert vrs_data_provider.loadDeviceModel()

        aria_camera_model = vrs_data_provider.getDeviceModel()
        slam_left = aria_camera_model.getCameraCalib("camera-slam-left")
        slam_right = aria_camera_model.getCameraCalib("camera-slam-right")
        rgb_cam = aria_camera_model.getCameraCalib("camera-rgb")

    assert slam_left is not None
    assert slam_right is not None
    assert rgb_cam is not None

    return {
        "1201-1": slam_left,
        "1201-2": slam_right,
        "214-1": rgb_cam,
    }


def aria_original_to_extracted(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, _ = img_shape
    none_idx = np.any(np.isnan(kpts), axis=1)
    new_kpts = kpts.copy()
    new_kpts[~none_idx, 0] = H - kpts[~none_idx, 1] - 1
    new_kpts[~none_idx, 1] = kpts[~none_idx, 0]
    return new_kpts


def hand_rand_bbox_from_kpts(kpts, img_shape, expansion_factor=1.5):
    """
    Generate random hand bbox based on hand kpts; for testing purpose.
    """
    img_H, img_W = img_shape[:2]
    # Get proposed hand bounding box from hand keypoints
    xmin, ymin, xmax, ymax = (
        kpts[:, 0].min(),
        kpts[:, 1].min(),
        kpts[:, 0].max(),
        kpts[:, 1].max(),
    )
    # Get x-coordinate for bbox
    x_center = (xmin + xmax) / 2.0
    width = (xmax - xmin) * expansion_factor
    rand_w = width * np.random.uniform(low=0.9, high=1.1)
    rand_x = width * np.random.uniform(low=-0.1, high=0.1)
    xmin = x_center + rand_x - 0.5 * rand_w
    xmax = x_center + rand_x + 0.5 * rand_w
    # Get y-coordinate for bbox
    y_center = (ymin + ymax) / 2.0
    height = (ymax - ymin) * expansion_factor
    rand_h = height * np.random.uniform(low=0.9, high=1.1)
    rand_y = height * np.random.uniform(low=-0.1, high=0.1)
    ymin = y_center + rand_y - 0.5 * rand_h
    ymax = y_center + rand_y + 0.5 * rand_h
    # Clip bbox within image bound
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        np.clip(xmin, 0, img_W - 1),
        np.clip(ymin, 0, img_H - 1),
        np.clip(xmax, 0, img_W - 1),
        np.clip(ymax, 0, img_H - 1),
    )
    bbox = np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2]).astype(np.float32)
    return bbox


def hand_pad_bbox_from_kpts(kpts, img_shape, padding=20):
    """
    Generate hand bbox based on hand kpts with padding; for train and val.
    """
    img_H, img_W = img_shape[:2]
    # Get proposed hand bounding box from hand keypoints
    x1, y1, x2, y2 = (
        kpts[:, 0].min(),
        kpts[:, 1].min(),
        kpts[:, 0].max(),
        kpts[:, 1].max(),
    )

    # Proposed hand bounding box with padding
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        np.clip(x1 - padding, 0, img_W - 1),
        np.clip(y1 - padding, 0, img_H - 1),
        np.clip(x2 + padding, 0, img_W - 1),
        np.clip(y2 + padding, 0, img_H - 1),
    )

    # Return bbox result
    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])


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


def world_to_cam(kpts, extri):
    """
    Transform 3D world kpts to camera coordinate system
    Input:
        kpts: (N,3)
        extri: (3,4) [R|t]
    Output:
        new_kpts: (N,3)
    """
    none_idx = np.any(np.isnan(kpts), axis=1)
    new_kpts = kpts.copy()
    new_kpts[none_idx] = 0
    new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1).T  # (4,N)
    new_kpts = (extri @ new_kpts).T  # (N,3)
    new_kpts[none_idx] = None
    return new_kpts


def cam_to_img(kpts, intri):
    """
    Project points in camera coordinate system to image plane
    Input:
        kpts: (N,3)
    Output:
        new_kpts: (N,2)
    """
    none_idx = np.any(np.isnan(kpts), axis=1)
    new_kpts = kpts.copy()
    new_kpts[none_idx] = -1
    new_kpts = intri @ new_kpts.T  # (3,N)
    new_kpts = new_kpts / new_kpts[2, :]
    new_kpts = new_kpts[:2, :].T
    new_kpts[none_idx] = None
    return new_kpts


def xywh2xyxy(bbox):
    """
    Given bbox in [x1,y1,w,h], return bbox corners [x1, y1, x2, y2]
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return np.array([x1, y1, x2, y2])


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180.0 / np.pi


def joint_dist_angle_check(curr_hand_pose3d):
    """
    Check hand biomechanical info: Distance and angle
    """
    ## Joint distance threshold ##
    long_joint_dist_index = [4, 8, 12, 16]
    joint_dist_min_threshold = np.full((20,), 0.005)
    joint_dist_min_threshold[long_joint_dist_index] = 0.06
    joint_dist_max_threshold = np.full((20,), 0.08)
    joint_dist_max_threshold[long_joint_dist_index] = 0.12
    ##Joint angle threshold ##
    joint_angle_min_threshold = np.array(
        [100, 90, 90, 60, 70, 80, 60, 70, 80, 60, 70, 80, 60, 70, 80]
    )
    joint_angle_max_threshold = np.array([180] * 15)
    ## Misc ##
    wrist_conn_index = [1, 5, 9, 13, 17]
    joint_dist_index = list(range(1, 21))
    joint_angle_index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    ###### Joint distance check #######
    joint_distance = []
    for joint_idx in joint_dist_index:
        if joint_idx in wrist_conn_index:
            joint_distance.append(
                np.linalg.norm(
                    curr_hand_pose3d[joint_idx][:3] - curr_hand_pose3d[0][:3]
                )
            )
        else:
            joint_distance.append(
                np.linalg.norm(
                    curr_hand_pose3d[joint_idx][:3]
                    - curr_hand_pose3d[joint_idx - 1][:3]
                )
            )
    joint_distance = np.array(joint_distance)

    ###### Joint angle check ######
    joint_angle = []
    for joint_idx in joint_angle_index:
        # If current joint has pose3d estimation
        if joint_idx in wrist_conn_index:
            vec1 = curr_hand_pose3d[0, :3] - curr_hand_pose3d[joint_idx, :3]
        else:
            vec1 = curr_hand_pose3d[joint_idx - 1, :3] - curr_hand_pose3d[joint_idx, :3]
        vec2 = curr_hand_pose3d[joint_idx + 1, :3] - curr_hand_pose3d[joint_idx, :3]
        # Compute angle
        joint_angle.append(angle_between(vec1, vec2))
    joint_angle = np.array(joint_angle)

    # Filter invalid joints from valid joints (vis_flag)
    invalid_dist_flag = np.logical_or(
        joint_distance < joint_dist_min_threshold,
        joint_distance > joint_dist_max_threshold,
    )
    invalid_dist_flag_ = np.full((21,), False)
    invalid_dist_flag_[joint_dist_index] = invalid_dist_flag

    invalid_angle_flag = np.logical_or(
        joint_angle < joint_angle_min_threshold, joint_angle > joint_angle_max_threshold
    )
    invalid_angle_flag_ = np.full((21,), False)
    invalid_angle_flag_[joint_angle_index] = invalid_angle_flag

    invalid_flag = np.logical_or(invalid_dist_flag_, invalid_angle_flag_)
    curr_hand_pose3d[invalid_flag] = None
    return curr_hand_pose3d


def get_interested_take(all_uids, takes_df):
    """
    For hand ego-pose baseline model, we are only interested in takes with
    scenario in Health, Bike Repair, Music, Cooking
    """
    interested_scenarios = ["Health", "Bike Repair", "Music", "Cooking"]
    scenario_take_dict = {scenario: [] for scenario in interested_scenarios}
    all_interested_scenario_uid = []
    for curr_local_cam_valid_uid in all_uids:
        curr_scenario = takes_df[takes_df["take_uid"] == curr_local_cam_valid_uid][
            "scenario_name"
        ].item()
        if curr_scenario in interested_scenarios:
            scenario_take_dict[curr_scenario].append(curr_local_cam_valid_uid)
            all_interested_scenario_uid.append(curr_local_cam_valid_uid)
    all_interested_scenario_uid = sorted(all_interested_scenario_uid)
    return all_interested_scenario_uid, scenario_take_dict


def get_ego_aria_cam_name(take):
    ego_cam_names = [
        x["cam_id"]
        for x in take["capture"]["cameras"]
        if str(x["is_ego"]).lower() == "true"
    ]
    assert len(ego_cam_names) > 0, "No ego cameras found!"
    if len(ego_cam_names) > 1:
        ego_cam_names = [
            cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
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
    return ego_cam_names
