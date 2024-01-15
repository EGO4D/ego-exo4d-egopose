import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .pool.poolattnformer_HR import (
    GroupNorm,
    load_pretrained_weights,
    PoolAttnFormer_hr,
)


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == "softmax":
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


class PoolAttnHR_Pose_3D(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PoolAttnHR_Pose_3D, self).__init__()
        self.deconv_dim = kwargs["NUM_DECONV_FILTERS"]
        self.num_joints = kwargs["NUM_JOINTS"]
        self.img_H = kwargs["IMAGE_SIZE"][0]
        self.img_W = kwargs["IMAGE_SIZE"][1]

        self.depth_dim = kwargs["EXTRA"]["DEPTH_DIM"]
        self.layers = kwargs["EXTRA"]["LAYERS"]
        self.embed_dims = kwargs["EXTRA"]["EMBED_DIMS"]
        self.mlp_ratios = kwargs["EXTRA"]["MLP_RATIOS"]
        self.drop_rate = kwargs["EXTRA"]["DROP_RATE"]
        self.drop_path_rate = kwargs["EXTRA"]["DROP_PATH_RATE"]

        self.height_dim = kwargs["EXTRA"]["HEATMAP_SIZE"][0]
        self.width_dim = kwargs["EXTRA"]["HEATMAP_SIZE"][1]

        # backbone: POTTER block pretrained with image size of [256,256]
        img_size = [self.img_H, self.img_W]
        self.poolattnformer_pose = PoolAttnFormer_hr(
            img_size,
            layers=self.layers,
            embed_dims=self.embed_dims,
            mlp_ratios=self.mlp_ratios,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
        )

        ######### 2D pose head #########
        self.norm1 = GroupNorm(256)
        self.up_sample = nn.Sequential(
            nn.Conv2d(self.embed_dims[0], 256, 1),
            nn.GELU(),
        )
        self.final_layer = nn.Conv2d(
            256, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0
        )
        self.pose_layer = nn.Sequential(
            nn.Conv3d(self.num_joints, self.num_joints, 1),
            nn.GELU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Conv3d(self.num_joints, self.num_joints, 1),
        )

        ######### 3D pose head #########
        self.pose_3d_head = nn.Sequential(
            nn.Linear(self.depth_dim * 3, 512),
            nn.ReLU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Linear(512, 3),
        )

    def _initialize(self):
        for m in self.up_sample.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward through all branches
        x_feature, _, _ = self.poolattnformer_pose(
            x
        )  # feature_map: [N, 64, H_feat, W_feat]

        # Intermediate layer
        out = self.up_sample(x_feature)  # [N, 256, H_feat, W_feat]
        out = self.norm1(out)
        out = self.final_layer(out)  # [N, num_joints*emb_dim, H_feat, W_feat]
        out = self.pose_layer(
            out.reshape(
                out.shape[0],
                self.num_joints,
                self.depth_dim,
                out.shape[2],
                out.shape[3],
            )
        )  # (N, num_joints, emb_dim, H_feat, W_feat)

        # 3D pose head
        hm_x0 = out.sum((2, 3))
        hm_y0 = out.sum((2, 4))
        hm_z0 = out.sum((3, 4))
        pose_3d_pred = torch.cat((hm_x0, hm_y0, hm_z0), dim=2)
        pose_3d_pred = self.pose_3d_head(pose_3d_pred)

        return pose_3d_pred


def load_pretrained_weights(model, checkpoint):
    import collections

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("backbone."):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(matched_layers)} pretrained parameters")
