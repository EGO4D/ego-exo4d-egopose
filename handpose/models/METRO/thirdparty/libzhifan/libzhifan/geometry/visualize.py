from functools import singledispatch
import numpy as np
from typing import Union
import trimesh
import torch
from trimesh.scene import Scene
from trimesh.transformations import rotation_matrix
from trimesh.points import PointCloud
from pytorch3d.structures import Meshes
from libzhifan.numeric import numpize
from libzhifan.geometry import SimpleMesh


_Rx = rotation_matrix(np.pi, [1, 0, 0])  # rotate pi around x-axis
_Ry = rotation_matrix(np.pi, [0, 1, 0])  # rotate pi around y-axis


@singledispatch
def _to_trimesh(mesh_in: trimesh.Trimesh) -> trimesh.Trimesh:
    return mesh_in
@_to_trimesh.register
def _dummy(mesh_in: Meshes):
    return trimesh.Trimesh(
            vertices=numpize(mesh_in.verts_packed()),
            faces=numpize(mesh_in.faces_packed()))
@_to_trimesh.register
def _dummy(mesh_in: SimpleMesh):
    m = trimesh.Trimesh(
            vertices=mesh_in.vertices,
            faces=mesh_in.faces)
    m.visual = mesh_in.visual
    return m


def color_faces(mesh, face_inds, color):
    """
    Args:
        mesh: SimpleMesh or Trimesh
        face_inds: (N,)
        color: [R, G, B]

    Returns:
        mesh: SimpleMesh or Trimesh
    """
    orig_colors = mesh.visual.face_colors
    new_clr = list(color) + [255]
    orig_colors[face_inds] = new_clr
    mesh.visual.face_colors = orig_colors
    return mesh


def color_verts(mesh, vert_inds, color):
    """
    Args:
        mesh: SimpleMesh or Trimesh
        vert_inds: (N,)
        color: [R, G, B]

    Returns:
        mesh: SimpleMesh or Trimesh
    """
    orig_colors = mesh.visual.vertex_colors
    new_clr = list(color) + [255]
    orig_colors[vert_inds] = new_clr
    mesh.visual.vertex_colors = orig_colors
    return mesh


def add_normals(mesh, normals) -> trimesh.Scene:
    """
    Args:
        mesh: SimpleMesh or Trimesh
        normals: (V, 3) same length as len(mesh.vertices)
    Returns:
        trimesh.Scene
    """
    normals = numpize(normals.squeeze())
    vec = np.column_stack(
        (mesh.vertices, mesh.vertices + (normals * mesh.scale * .05)))
    path = trimesh.load_path(vec.reshape(-1, 2, 3))
    return trimesh.Scene([mesh, path])


def visualize_mesh(mesh_data,
                   show_axis=True,
                   viewpoint='pytorch3d'):
    """
    Args:
        mesh: one of
            - None, which will be skipped
            - SimpleMesh
            - pytorch3d.Meshes
            - list of SimpleMeshes
            - list of pytorch3d.Meshes
        viewpoint: str, one of
            {
                'pytorch3d',
                'opengl',
                'neural_renderer'/'nr'
            }

    Return:
        trimesh.Scene
    """
    s = trimesh.Scene()

    if isinstance(mesh_data, list):
        for m in mesh_data:
            if m is None:
                continue
            s.add_geometry(_to_trimesh(m))
    else:
        s.add_geometry(_to_trimesh(mesh_data))

    if show_axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=0.4)
        s.add_geometry(axis)

    if viewpoint == 'pytorch3d':
        s = s.copy()  # We don't want the in-place transform affecting input data
        s.apply_transform(_Ry)
    elif viewpoint == 'opengl':
        # By default trimesh uses opengl mode
        pass
    elif viewpoint == 'neural_renderer' or viewpoint == 'nr':
        s = s.copy()
        s.apply_transform(_Rx)
    else:
        raise ValueError

    return s


def visualize_mesh_with_point(mesh: trimesh.Trimesh,
                              point,
                              radius=0.01,
                              **kwargs) -> trimesh.Scene:
    """
    Args:
        point: (x, y, z)
    """
    point = numpize(point.squeeze())
    ball = create_spheres(points=point[None], radius=radius)
    return visualize_mesh([mesh, ball], **kwargs)


def visualize_hand_object(hand_verts=None,
                          hand_faces=None,
                          hand_poses=None,
                          hand_betas=None,
                          hand_glb_orient=None,
                          hand_transl=None,
                          use_pca=True,
                          obj_verts=None,
                          obj_faces=None,
                          show_axis=True,
                          viewpoint='pytorch3d'):
    """

    If `hand_verts` are supplied, the rest of hand_* will be ignored.
    Otherwise, will try to construct hand_verts using hand_{poses, betas, glb_orient}

    Args:
        hand_verts: (778, 3)
        hand_faces: (F, 3)
        hand_poses: (45, 3)
        hand_betas: (10,)
        obj_verts: (V_o, 3)
        viewpoint: str, one of {'pytorch3d', 'opengl'}

    Returns:
        trimesh.Scene
    """
    print("DEPRECATED: use visualize_mesh() instead.")
    s = trimesh.Scene()

    if hand_verts is not None:
        assert hand_faces is not None
        hand_mesh = trimesh.Trimesh(hand_verts, hand_faces)
        s.add_geometry(hand_mesh)
    elif hand_poses is not None:
        assert hand_faces is not None

    if obj_verts is not None:
        if obj_faces is not None:
            obj = trimesh.Trimesh(obj_verts, obj_faces)
        else:
            obj = trimesh.points.PointCloud(
                obj_verts, colors=np.tile(np.array([0, 0, 0, 1]), (len(obj_verts), 1))
            )
        s.add_geometry(obj)

    if show_axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=0.4)
        s.add_geometry(axis)

    if viewpoint == 'pytorch3d':
        s.apply_transform(_Ry)

    return s


def visualize_rotation_distribution(rot_mats,
                                    origin_size=0.05,
                                    axis_radius=0.004,
                                    axis_length=0.05) -> trimesh.Scene:
    """ Place rotations on a unit sphere

    Args:
        rot_mats: (N, 3, 3)
    """
    axs = []
    for rot_mat in rot_mats:
        T = np.eye(4)
        T[:3, :3] = rot_mat.cpu().numpy()
        z_vec = rot_mat[:3, -1]
        z_vec = z_vec / np.linalg.norm(z_vec)
        T[:3, -1] = - z_vec
        ax = trimesh.creation.axis(
            transform=T, origin_size=origin_size,
            axis_radius=axis_radius, axis_length=axis_length)
        axs.append(ax)
    return visualize_mesh(axs, show_axis=True)


def create_pcd_scene(points, colors=None, ret_pcd=False):
    """ See also create_spheres()

    Args:
        points: shape (N, 3)
        colors: shape (N, 3), values in [0, 1]

    Returns:
        a Scene
        or
        a PointCloud
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    if colors is not None:
        assert len(colors.shape) == 2 and colors.shape[1] == 3
        alpha = np.ones([len(points)]).reshape(-1, 1)
        colors = np.concatenate([colors, alpha], 1)
    else:
        colors = np.tile(np.array([0, 0, 0, 1]), (len(points), 1))
    pcd = PointCloud(
        vertices=points,
        colors=colors)
    if ret_pcd:
        return pcd
    return Scene([pcd])


def create_spheres(points,
                   colors=None,
                   radius=0.01,
                   subdivisions=0
                   ) -> trimesh.Trimesh:
    """
    Args:
        points: (N, 3)
        colors: (N, 3) in [0, 255]
        subvisions: 0 for speed, 3 for quality
    """
    spheres = []
    if colors is None:
        colors = np.tile(np.array([255, 0, 0, 255]), (len(points), 1))
    else:
        colors = np.hstack([
            np.asarray(colors), np.ones([len(points), 1]) * 255])
    for i, p in enumerate(points):
        s = trimesh.primitives.Sphere(
            radius=radius, center=p, subdivisions=subdivisions)
        s.visual.vertex_colors = colors[i]#  + [255]
        spheres.append(s)
    mesh = trimesh.util.concatenate(spheres)
    return mesh


def create_path_cone(p1: np.ndarray,
                     p2: np.ndarray) -> trimesh.Trimesh:
    """
    Args:
        p1: (3,)
        p2: (3,)
    Return:
        A Trimesh Cone pointing from p1 to p2
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p12 = p2 - p1
    length = np.linalg.norm(p12)
    radius = 0.05 * length
    cone = trimesh.creation.cone(radius=radius, height=length)
    z_axis = p12 / length
    up_axis = np.float32([0, 0, 1.0])
    if (1.0 - z_axis.dot(up_axis)) < 1e3:  # if z_axis is too close to up_axis
        y_axis = np.cross(z_axis, np.float32([1, 0, 0]))
        x_axis = np.cross(y_axis, z_axis)
    else:
        y_axis = np.cross(z_axis, up_axis)
        x_axis = np.cross(z_axis, y_axis)
    transform = np.eye(4)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = p1
    cone.apply_transform(transform)
    return cone
