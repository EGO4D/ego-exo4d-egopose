""" Utility function for coordinate system. """

from typing import Union

import numpy as np
import torch

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Transform3d

from libzhifan.numeric import nptify, numpize


def nptify_wrapper(func):
    def nptified_func(*args):
        type_convertor = nptify(args[0])
        _args = map(numpize, args)
        ret = func(*_args)
        type_convertor(ret)
    return nptified_func


""" Functions dealing with homogenous coordiate transforms. """


# @nptify_wrapper
def to_homo_xn(pts):
    """ assume [x, n], output [x+1, n]"""
    n = pts.shape[1]
    return np.vstack((pts, np.ones([1, n])))


# @nptify_wrapper
def to_homo_nx(pts):
    """ [n,x] -> [n,x+1] """
    return to_homo_xn(pts.T).T


def from_home_xn(pts):
    """ [x+1, n] -> [x, n] """
    return pts[:-1, :]


def from_home_nx(pts):
    """ [n, x+1] -> [n, x] """
    return from_home_xn(pts.T).T


def normalize_homo_xn(x_h):
    return x_h / x_h[-1, :]


def normalize_homo_nx(x_h):
    return x_h / x_h[:, -1]


def normalize_and_drop_homo_xn(x_h):
    """ x_h: (c, n) -> (c-1, n) """
    return x_h[:-1, :] / x_h[-1, :]


def normalize_and_drop_homo_nx(x_h):
    """ x_h: (n, c) -> (n, c-1) """
    return normalize_and_drop_homo_xn(x_h.T).T


def transform_nx3(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (n, 3)

    Returns: (n, 3)

    """
    return transform_3xn(transform_matrix, x.T).T


def transform_3xn(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (3, n)

    Returns: (3, n)

    """
    x2 = transform_matrix @ to_homo_xn(x)
    return from_home_xn(x2)


def concat_rot_transl_3x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (3, 4)

    """
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = transl.squeeze()
    return Rt


def concat_rot_transl_4x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (4, 4)

    """
    typer = nptify(rot)
    Rt = typer(np.zeros([4, 4]))
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = typer(transl.squeeze())
    Rt[-1, -1] = 1.0
    return Rt


def rotation_epfl(alpha, beta, gamma):
    R = np.zeros([3, 3])
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    R[0, 0] = cos_a * cos_g - cos_b * sin_a * sin_g
    R[1, 0] = cos_g * sin_a + cos_a * cos_b * sin_g
    R[2, 0] = sin_b * sin_g

    R[0, 1] = -cos_b * cos_g * sin_a - cos_a * sin_g
    R[1, 1] = cos_a * cos_b * cos_g - sin_a * sin_g
    R[2, 1] = cos_g * sin_b

    R[0, 2] = sin_a * sin_b
    R[1, 2] = -cos_a * sin_b
    R[2, 2] = cos_b

    return R


def lift_rotation_se3(rot_mat):
    """
    Life a (3, 3) rotation matrix into (4, 4) se3 transformation.
    """
    transform = nptify(rot_mat)(np.eye(4))
    transform[..., :3, :3] = rot_mat
    return transform


# def rot6d_to_matrix(rot_6d):
#     """
#     TODO, finalize
#     Convert 6D rotation representation to 3x3 rotation matrix.
#     Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
#     Networks", CVPR 2019

#     Args:
#         rot_6d (B x 6): Batch of 6D Rotation representation.

#     Returns:
#         Rotation matrices (B x 3 x 3).
#     """
#     rot_6d = rot_6d.view(-1, 3, 2)
#     a1 = rot_6d[:, :, 0]
#     a2 = rot_6d[:, :, 1]
#     b1 = F.normalize(a1)
#     b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
#     b3 = torch.cross(b1, b2)
#     return torch.stack((b1, b2, b3), dim=-1)


# def rotation_xyz_from_euler(x_rot, y_rot, z_rot):
#     rz = np.float32([
#         [np.cos(z_rot), np.sin(z_rot), 0],
#         [-np.sin(z_rot), np.cos(z_rot), 0],
#         [0, 0, 1]
#     ])
#     ry = np.float32([
#         [np.cos(y_rot), 0, -np.sin(y_rot)],
#         [0, 1, 0],
#         [np.sin(y_rot), 0, np.cos(y_rot)],
#     ])
#     rx = np.float32([
#         [1, 0, 0],
#         [0, np.cos(x_rot), np.sin(x_rot)],
#         [0, -np.sin(x_rot), np.cos(x_rot)],
#     ])
#     return rz @ ry @ rx


""" Primitive Transforms """


def translate(points, x, y, z):
    """
    Args:
        points: (n, 3)
        x, y, z: scalar
    Returns:
        (n, 3)
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    out_type = nptify(points)
    translation = out_type([[x, y, z]])
    return points + translation


def scale(points, x, y, z):
    """
    Args:
        points: (n, 3)
        x, y, z: scalar
    Returns:
        (n, 3)
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    out_type = nptify(points)
    scale_factor = out_type([[x, y, z]])
    center = points.mean(0)
    return (points - center) * scale_factor + center



""" Camera """


def camera_matrix(fx, fy, cx, cy):
    """

    Returns:
        K: (3, 3) ndarray
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ], dtype=type(fx))
    return K


def project_3d_2d(x3d, A=None, fx=None, fy=None, cx=None, cy=None):
    """

    Args:
        x3d: (3, n)
        A: (3, 3)
        fx, fy, cx, cy: scalar
            either A or (fx, fy, cx , cy) must be supplied.

    Returns: (2, n)
    """
    if A is not None:
        assert A.ndim == 2 and A.shape[-1] == 3
        x2d_h = A @ x3d
    else:
        assert fx and fy and (cx is not None) and (cy is not None)
        K = camera_matrix(fx, fy, cx, cy)
        x2d_h = nptify(x3d)(K) @ x3d
    return extract_pixel_homo_xn(x2d_h)


def extract_pixel_homo_xn(x2d_h):
    """ x2d_h: [3, n] -> [2, n] """
    return normalize_and_drop_homo_xn(x2d_h)


def extract_pixel_homo_nx(x2d_h):
    """ x2d_h: [n, 3] -> [n, 2] """
    return normalize_and_drop_homo_nx(x2d_h.T).T


def world_to_camera_pipeline(x3d, M_intrinsic, M_offset=None, Tmw=None):
    """
    P_2d = M_offset @ divide_by_z() @ M_intrinsic @ P_3d
        where P_3d = (x, y, z)

    Args:
        x3d: (n, 3)
        M_offset: (3, 3)
        M_intrinsic: (3, 3) or (3, 4)
        Tmw: (4, 4)

    Returns:

    """
    pts_h = to_homo_nx(x3d).T  # pts_h: (4, n)
    if M_offset is None:
        M_offset = np.identity(3)
    if Tmw is None:
        Tmw = np.identity(4)
    if M_intrinsic.shape == (3, 3):
        _M_intrinsic = np.zeros([3, 4])
        _M_intrinsic[:3, :3] = M_intrinsic
    elif M_intrinsic.shape == (3, 4):
        _M_intrinsic = np.zeros([3, 4])
        _M_intrinsic[:3, :] = M_intrinsic
    else:
        raise ValueError(f"Unexpected M_intrinsic shape: {M_intrinsic.shape}")

    M_transformations = _M_intrinsic @ Tmw
    pts_h = M_transformations @ pts_h
    pts_h = normalize_homo_xn(pts_h)
    pts_h = M_offset @ pts_h
    pts_2d = from_home_xn(pts_h).transpose()  # (3,n) -> (n,2)

    return pts_2d


def world_to_camera_pipeline_test():
    x3d = np.float32([
        [10, 20, 10],
        [0, -10, 2],
        [-30, 0, 3],
        [-5, -5, 1],
    ])
    fx = 4
    fy = 2
    M_intrinsic = np.float32([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0,  1],
    ])
    x2d_true = np.float32([
        [4, 4],
        [0, -10],
        [-40, 0],
        [-20, -10]
    ])
    x2d_est = world_to_camera_pipeline(x3d, M_intrinsic)
    np.testing.assert_almost_equal(x2d_true, x2d_est, verbose=True)


def reverse_offset(x2d, M_offset):
    """
    A typical pipeline of projecting 3d points to 2d is:
    P_2d = M_offset @ divide_by_z() @ M_intrinsic @ P_3d
         = M_offset @ P_2d'

    This function calculate P_2d' given P_2d and M_offset

    Args:
        x2d: (n, 2), above P_2d
        M_offset: (3, 3)

    Returns: (n, 2)

    """
    M_offset_inv = np.linalg.inv(M_offset)
    x2d_h = to_homo_nx(x2d)
    x2d_prim_h = x2d_h @ M_offset_inv.T
    return from_home_nx(x2d_prim_h)


""" Change of coordinate system. """


def points_opencv_to_opengl(pts):
    """
    Args:
        (n, 3)

    Returns:
        (n, 3)
    """
    pts = pts.copy()
    pts[:, 1] = - pts[:, 1]
    return pts


""" Pytorch3d transforms
"""


def torch3d_get_verts(geom: Union[Meshes, Pointclouds]) -> torch.Tensor:
    if isinstance(geom, Meshes) or isinstance(geom, Meshes):
        view_points = geom.verts_padded()
    elif isinstance(geom, Pointclouds):
        view_points = geom.points_padded()
    elif isinstance(geom, torch.Tensor):
        view_points = geom
    else:
        raise NotImplementedError(type(geom))
    return view_points


def torch3d_apply_transform(
        geom: Union[Meshes, Pointclouds, torch.Tensor],
        trans: Transform3d):
    """
    Returns:
        tranformed geometry object.
    """
    verts = torch3d_get_verts(geom)
    verts = trans.transform_points(verts)
    if hasattr(geom, 'update_padded'):
        geom = geom.update_padded(verts)
    else:
        geom = verts
    return geom


def torch3d_apply_transform_matrix(
        geom: Union[Meshes, Pointclouds, torch.Tensor],
        trans,
        convert_trans_col_to_row=True):
    """
    Note: transformation is implemented as right-multiplication,
    hence geom is row-vector.

    Args:
        trans: transformation. Either
            - matrix of (4, 4)
            - matrix of (1, 4, 4)
            - Transform3d

        convert_trans_col_to_rw: bool, i.e. transpose

    Returns:
        tranformed geometry object.
    """
    if isinstance(trans, Transform3d):
        return torch3d_apply_transform(geom, trans)

    trans = torch.as_tensor(trans).reshape(1, 4, 4)
    if convert_trans_col_to_row:
        trans = Transform3d(
            matrix=trans.transpose(1, 2), device=geom.device)
    else:
        trans = Transform3d(
            matrix=trans, device=geom.device)

    return torch3d_apply_transform(geom, trans)


def torch3d_apply_scale(geom: Union[Meshes, Pointclouds, torch.Tensor],
                        scale: float):
    device = geom.device
    trans = torch.eye(4).reshape(1, 4, 4).to(device)
    trans[..., [0,1,2], [0,1,2]] = scale
    return torch3d_apply_transform_matrix(
        geom,
        Transform3d(matrix=trans, device=device),
        convert_trans_col_to_row=True)


def torch3d_apply_translation(geom: Union[Meshes, Pointclouds, torch.Tensor],
                              translation):
    """
    Args:
        translation: (3,)
    """
    device = geom.device
    trans = torch.eye(4).reshape(1, 4, 4).to(device)
    trans[..., :3, -1] = torch.as_tensor(translation, device=device)
    return torch3d_apply_transform_matrix(
        geom, trans, convert_trans_col_to_row=True)


def torch3d_apply_Rx(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Rx_mat = torch.as_tensor([[
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Rx_mat, convert_trans_col_to_row=True)


def torch3d_apply_Ry(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Ry_mat = torch.as_tensor([[
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Ry_mat, convert_trans_col_to_row=True)


def torch3d_apply_Rz(geom: Union[Meshes, Pointclouds, torch.Tensor],
                     degree: int):
    theta = degree / 180 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Rz_mat = torch.as_tensor([[
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]], dtype=torch.float32, device=geom.device)
    return torch3d_apply_transform_matrix(
        geom, Rz_mat, convert_trans_col_to_row=True)
