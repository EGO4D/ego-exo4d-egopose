import numpy as np
import torch
import torch.nn as nn


class Pose3DLoss(nn.Module):
    def __init__(self):
        super(Pose3DLoss, self).__init__()

    def forward(self, pose_3d_pred, pose_3d_gt, vis_flag):
        # Compute MSE loss between pred and gt 3D hand joints for only visible kpts
        assert (
            pose_3d_pred.shape == pose_3d_gt.shape and len(pose_3d_pred.shape) == 3
        )  # (N, K, dim)
        pose_3d_diff = pose_3d_pred - pose_3d_gt
        pose_3d_loss = torch.mean(pose_3d_diff**2, axis=2) * vis_flag
        pose_3d_loss = torch.sum(pose_3d_loss) / torch.sum(vis_flag)

        return pose_3d_loss


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance) from
    https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py.
    Modified s.t. it could compute MPJPE for only those valid keypoints (where
    # of visible keypoints = num)
    """
    assert predicted.shape == target.shape
    pjpe = torch.norm(predicted - target, dim=len(target.shape) - 1)
    mpjpe = torch.mean(pjpe)
    return mpjpe


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    p_mpjpe = np.mean(pjpe)
    return p_mpjpe
