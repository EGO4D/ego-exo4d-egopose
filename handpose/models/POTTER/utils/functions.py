import argparse
import logging
import os
import time
from pathlib import Path

import yaml
from easydict import EasyDict as edict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def create_logger(cfg, cfg_name, phase="train"):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print("=> creating {}".format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    cfg_name = os.path.basename(cfg_name).split(".")[0]
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    final_output_dir = root_output_dir / dataset / time_str

    print("=> creating {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = "{}_{}_{}.log".format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    tensorboard_log_dir = final_output_dir / "tb_log"
    print("=> creating {}".format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def parse_args_function():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_file",
        default="configs/potter_pose_3d_ego4d.yaml",
        help="Config file path of hand-ego-pose-potter",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        help="Pretrained hand-ego-pose-potter checkpoint",
    )
    parser.add_argument(
        "--cls_ckpt",
        default="output/ckpt/cls_s12.pth",
        help="Pretrained potter-cls checkpoint path",
    )
    parser.add_argument(
        "--gpu_number",
        type=int,
        nargs="+",
        default=[0],
        help="Identifies the GPU number to use",
    )
    parser.add_argument(
        "--gt_anno_dir",
        default=None,
        help="Directory of where ground truth annotation JSON files are stored",
        required=True,
    )
    parser.add_argument(
        "--aria_img_dir",
        default=None,
        help="Directory of where undistorted Aria images are stored",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default="output/inference_output",
        help="Output directory where inference JSON result will be stored",
    )

    args = parser.parse_args()
    return args
