import argparse
import copy
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset.ego4d_dataset import ego4dDataset
from easydict import EasyDict as edict
from models.PoolAttnHR_Pose_3D import load_pretrained_weights, PoolAttnHR_Pose_3D
from tqdm import tqdm
from utils.functions import AverageMeter, parse_args_function, update_config
from utils.loss import mpjpe, p_mpjpe, Pose3DLoss

"""
Perform inference on public un-annotated JSON test file, save model output as JSON file
"""


def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    device = torch.device(
        f"cuda:{args.gpu_number[0]}" if torch.cuda.is_available() else "cpu"
    )

    ############ MODEL ###########
    model = PoolAttnHR_Pose_3D(**cfg.MODEL)
    # Load pretrained cls_weight or available hand pose weight
    load_pretrained_weights(
        model, torch.load(args.pretrained_ckpt, map_location=device)
    )
    model = model.to(device)
    model.eval()

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    valid_dataset = ego4dDataset(args, cfg, split="test", transform=transform)
    # Dataloader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )
    print(f"Number of images: {len(valid_dataset)}")

    ######### Inferece #########
    pred_output = copy.deepcopy(valid_dataset.pred_temp)
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, dynamic_ncols=True)
        for input, meta in valid_loader:
            # Pose 3D prediction
            input = input.to(device)
            pose_3d_pred = model(input)

            # Unnormalize predicted 3D hand kpts
            pred_3d_pts = pose_3d_pred.cpu().detach().numpy()
            pred_3d_pts = (
                pred_3d_pts * valid_dataset.joint_std + valid_dataset.joint_mean
            ).squeeze(
                0
            )  # (21,3)
            # mm to m
            pred_3d_pts /= 1000

            # Append into output JSON file
            take_uid, frame_number, hand_order = (
                meta["take_uid"][0],
                str(meta["frame_number"].to(int).item()),
                meta["hand_order"][0],
            )
            pred_output[take_uid][frame_number][
                f"{hand_order}_hand"
            ] = pred_3d_pts.tolist()

    ######### Save output JSON file #########
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_type = re.split("[/_\.]", args.pretrained_ckpt)[-2]
    pred_output_path = os.path.join(args.output_dir, f"ego_pose_pred_{ckpt_type}.json")
    with open(pred_output_path, "w") as f:
        json.dump(pred_output, f, indent=4)
    print(f"Successfully saved inference output at {pred_output_path}")


if __name__ == "__main__":
    args = parse_args_function()
    main(args)
