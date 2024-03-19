import argparse
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
        assert p in ["takes", "take_vrs"], f"Invalid parts: {p}"
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


def main(args):
    # Get all annotated takes
    all_local_take_uids = set()
    anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}

    for split in args.splits:
        for anno_type_ in args.anno_types:
            anno_type = anno_type_dir_dict[anno_type_]
            curr_split_anno_dir = os.path.join(
                args.ego4d_data_dir, f"annotations/ego_pose/{split}/hand", anno_type
            )
            if os.path.exists(curr_split_anno_dir):
                curr_split_take_uids = [
                    k.split(".")[0] for k in os.listdir(curr_split_anno_dir)
                ]
                all_local_take_uids.update(curr_split_take_uids)
    all_local_take_uids = list(all_local_take_uids)
    assert len(all_local_take_uids) > 0, "No takes find."
    cmd_uids = " ".join(all_local_take_uids)

    for p in args.parts:
        if p == "takes":
            cmd = f"egoexo -o {args.ego4d_data_dir} --parts takes --views ego --uids {cmd_uids}"
            os.system(cmd)
        if p == "take_vrs":
            cmd = f"egoexo -o {args.ego4d_data_dir} --parts take_vrs --uids {cmd_uids}"
            os.system(cmd)


if __name__ == "__main__":
    args = create_arg_parse()
    main(args)
