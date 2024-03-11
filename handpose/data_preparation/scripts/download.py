import argparse
import os
import sys


def create_arg_parse():
    parser = argparse.ArgumentParser("Download ego hand pose data for only annotated takes via EgoExo4D Downloader")
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        help="Root directory of downloaded EgoExo-4D anntations <ego4d_data_dir>"
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=[],
        help="download filtered takes or take_vrs"
    )
    args = parser.parse_args()
    # Can only filter takes and take_vrs for now
    for p in args.parts:
        assert p in ["takes", "take_vrs"], f"Invalid parts: {p}"
    return args


def main(args):
    # TODO: Confirm about directory to be used to find all locally available takes
    # Get all annotated takes
    local_anno_take_dir = os.path.join(args.output_dir, "annotations/ego_pose/hand/annotation")
    all_local_take_uids = [k.split(".")[0] for k in os.listdir(local_anno_take_dir)]
    assert len(all_local_take_uids) > 0, "No takes find."
    cmd_uids = " ".join(all_local_take_uids)

    for p in args.parts:
        if p == "takes":
            cmd = f"egoexo -o {args.output_dir} --parts takes --views ego --uids {cmd_uids}"
            os.system(cmd)
        if p == "take_vrs":
            cmd = f"egoexo -o {args.output_dir} --parts take_vrs --uids {cmd_uids}"
            os.system(cmd)


if __name__ == "__main__":
    args = create_arg_parse()
    main(args)