import argparse


def create_arg_parse():
    parser = argparse.ArgumentParser("Ego-pose baseline model dataset preparation")

    # Parameters of data preparation pipeline
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="train/val/test split of the dataset",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["gt_anno", "raw_image", "undistorted_image"],
        help="""
            Determine which step should be executed in data preparation:
            - gt_anno: Extract ground truth annotation file
            - raw_image: Extract raw ego-view (aria) images
            - undistorted_image: Undistort raw aria images
            """,
    )
    parser.add_argument(
        "--anno_types",
        type=str,
        nargs="+",
        default=["manual"],
        help="Type of annotation: use manual or automatic data",
    )

    # Ego4d data and output directory
    parser.add_argument(
        "--ego4d_data_dir",
        type=str,
        default=None,
        help="Directory of downloaded Ego4D data, including annotations, captures, takes, metadata.",
        required=True,
    )
    parser.add_argument(
        "--gt_output_dir",
        type=str,
        default=None,
        help="Directory to store preprocessed ground truth annotation JSON file",
        required=True,
    )

    # Threshold and parameters in dataloader
    parser.add_argument("--valid_kpts_num_thresh", type=int, default=10)
    parser.add_argument("--bbox_padding", type=int, default=20)

    args = parser.parse_args()

    # Parameter sanity check
    for step in args.steps:
        assert step in [
            "gt_anno",
            "raw_image",
            "undistorted_image",
        ], f"Invalid step: {step}"

    for split in args.splits:
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

    for anno_type in args.anno_types:
        assert anno_type in [
            "manual",
            "auto",
        ], f"Invalid annotation type: {anno_type}"

    return args
