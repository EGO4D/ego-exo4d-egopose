import argparse
import os
import json
import numpy as np


def create_arg_parse():
    parser = argparse.ArgumentParser("Ego-pose baseline model evaluation")

    parser.add_argument(
        "--gt_output_dir",
        type=str,
        default=None,
        help="Directory to store preprocessed ground truth annotation JSON file",
        required=True,
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = create_arg_parse()

    gt_file = os.path.join(args.gt_output_dir, "annotation/manual/ego_pose_gt_anno_test_private.json")
    assert os.path.exists(gt_file), "No ground truth annotation file for test split. Note that this is not released for public. "

    gt = json.load(open(gt_file))
    for take_id in gt.keys():
        for frame_id in gt[take_id].keys():
            for hand in ["right", "left"]:
                if len(gt[take_id][frame_id][hand+"_hand_3d"]) > 0:
                    gt[take_id][frame_id][hand + "_hand_3d"] = np.random.uniform(0,1,(21,3)).tolist()
                    gt[take_id][frame_id][hand + "_hand_2d"] = np.random.uniform(0,1,(21,2)).tolist()
                    gt[take_id][frame_id][hand + "_bbox"] = np.random.uniform(0,1,(4,)).tolist()
                    gt[take_id][frame_id][hand + "_valid_3d"] = (np.random.uniform(0,1,(21,))>0.5).tolist()

    save_dummy_gt_path = os.path.join(args.gt_output_dir, "annotation/manual/dummy_test.json")
    json.dump(gt, open(save_dummy_gt_path, 'w'))
