import argparse
import json
import os
import sys


def create_arg_parse():
    parser = argparse.ArgumentParser(
        "Download ego hand pose data for only annotated takes via EgoExo4D Downloader"
    )
    parser.add_argument(
        "--ego4d_data_dir",
        default=None,
        required=True,
        help="Root directory of downloaded EgoExo-4D anntations <ego4d_data_dir>",
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=[],
        help="download filtered takes or take_vrs",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="train/val/test split of the dataset",
    )
    parser.add_argument(
        "--anno_types",
        type=str,
        nargs="+",
        default=["manual"],
        help="Type of annotation: use manual or automatic data",
    )
    args = parser.parse_args()
    # Can only filter takes and take_vrs for now
    for p in args.parts:
        assert p in ["takes", "take_vrs_noimagestream"], f"Invalid parts: {p}"
    # split sanity check
    for split in args.splits:
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
    # anno type sanity check
    for anno_type in args.anno_types:
        assert anno_type in [
            "manual",
            "auto",
        ], f"Invalid annotation type: {anno_type}"

    return args


def find_annotated_takes(ego4d_data_dir, splits, anno_types):
    # Get all annotated takes
    all_local_take_uids = set()
    anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}
    for split in splits:
        # For test split, check existing takes with two options
        if split == "test":
            test_split_anno_dir = os.path.join(
                ego4d_data_dir, "annotations/ego_pose/test/hand/annotation"
            )
            # Option 1: if existing private test annotation, then find test takes from local directory
            if (
                os.path.exists(test_split_anno_dir)
                and len(os.listdir(test_split_anno_dir)) > 0
            ):
                curr_split_take_uids = [
                    k.split(".")[0] for k in os.listdir(test_split_anno_dir)
                ]
            # Option 2: otherwise, load test takes from public test gt-anno
            else:
                test_list_file = "ego_pose_gt_anno_test_public.json"
                test_file = json.load(open(test_list_file))
                curr_split_take_uids = test_file.keys()
            all_local_take_uids.update(curr_split_take_uids)
        else:
            # For all other splits, find available takes from local directory
            for anno_type_ in anno_types:
                anno_type = anno_type_dir_dict[anno_type_]
                curr_split_anno_dir = os.path.join(
                    ego4d_data_dir, f"annotations/ego_pose/{split}/hand", anno_type
                )
                if os.path.exists(curr_split_anno_dir):
                    curr_split_take_uids = [
                        k.split(".")[0] for k in os.listdir(curr_split_anno_dir)
                    ]
                    all_local_take_uids.update(curr_split_take_uids)
    return list(all_local_take_uids)


def main(args):
    all_local_take_uids = find_annotated_takes(
        args.ego4d_data_dir, args.splits, args.anno_types
    )
    assert len(all_local_take_uids) > 0, "No takes find."
    cmd_uids = " ".join(all_local_take_uids)

    for p in args.parts:
        if p == "takes":
            cmd = f"egoexo -o {args.ego4d_data_dir} --parts takes --views ego --uids {cmd_uids}"
            os.system(cmd)
        if p == "take_vrs_noimagestream":
            cmd = f"egoexo -o {args.ego4d_data_dir} --parts take_vrs_noimagestream --uids {cmd_uids}"
            os.system(cmd)


if __name__ == "__main__":
    args = create_arg_parse()
    main(args)
