import unittest

import os
import numpy as np
import torch

from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, PointLights,
    AmbientLights,
    MeshRasterizer, SoftPhongShader, MeshRenderer,
    TexturesUV,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import join_meshes_as_scene

from libzhifan import epylab
from libzhifan.geometry import example_meshes
from libzhifan.geometry import projection
from libzhifan.geometry import SimpleMesh, InstanceIDRenderer


class Shapes:
    def __init__(self):
        pass

    @property
    def cube1(self):
        """ Top-Left """
        fx = 10
        return example_meshes.canonical_cuboids(
            x=1, y=1, z=2*fx + 1,
            w=2, h=2, d=2,
            convention='opengl',
            return_mesh=True,
        )

    @property
    def cube2(self):
        """ Bottom-right, smaller """
        fx = 10
        return example_meshes.canonical_cuboids(
            x=-1, y=-1, z=2*fx + 1,
            w=2, h=2, d=2,
            convention='opengl',
            return_mesh=True,
        )

    @property
    def cube3(self):
        """ Middle """
        return example_meshes.canonical_cuboids(
            x=0, y=0, z=3,
            w=2, h=2, d=2,
            convention='opencv',
            return_mesh=True
        )


_shapes = Shapes()


class TestNaiveProjection(unittest.TestCase):

    def test_one_simple_mesh(self):
        img = projection.perspective_projection(
            _shapes.cube3,
            cam_f=(100, 100),
            cam_p=(100, 100),
            method=dict(name='naive'),
            img_h=200,
            img_w=200,
        )
        self.assertEqual((img != 255).sum(), 5304)


class TestPytorch3dProjection(unittest.TestCase):

    def test_one_mesh(self):
        fx = 10
        img = projection.perspective_projection(
            [_shapes.cube1],
            cam_f=(fx, fx),
            cam_p=(0,0),
            method=dict(
                name='pytorch3d',
                in_ndc=True
            ),
            img_h=200,
            img_w=200,
        )

    def test_two_pytorch3d_meshes(self):
        fx = 10
        device = 'cuda'
        verts, faces = example_meshes.canonical_cuboids(
            x=1, y=1, z=2*fx + 1,
            w=2, h=2, d=2,
            convention='opengl',
            return_mesh=False,
        )
        verts = torch.as_tensor(verts, device=device, dtype=torch.float32)
        faces = torch.as_tensor(faces, device=device)
        verts_rgb = torch.ones_like(
            verts) * torch.as_tensor([0.65, 0.74, 0.86], device=device)
        textures = TexturesVertex(verts_features=verts_rgb[None].to(device))
        _m = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures)
        img = projection.perspective_projection(
            [_m, _m],
            cam_f=(fx, fx),
            cam_p=(0, 0),
            method=dict(
                name='pytorch3d',
                in_ndc=True
            ),
            img_h=200,
            img_w=200,
        )


class TestNeuralRendererProjection(unittest.TestCase):

    def test_one_simple_mesh(self):
        fx = fy = cx = cy = 0.5
        img = projection.perspective_projection(
            _shapes.cube3,
            cam_f=(fx, fy),
            cam_p=(cx, cy),
            method=dict(name='neural_renderer'),
            img_h=200,
            img_w=200)


def visualize_cube_with_unit_camera():
    """
    Note:
    In pytorch3d, `in_ndc=False` means the units are defined in screen space,
    which means the units are pixels;
    However, we normally define the units in world coordinates,
    which hardly can have pixels units, therefore `in_ndc` should be set to True.
    """
    IN_NDC = True  # Setting in_ndc=True is very important
    image_size = (200, 200)
    mesh = example_meshes.canonical_cuboids(
        x=0, y=0, z=3,
        w=2, h=2, d=2,
        convention='opencv'
    )
    verts, faces = map(torch.from_numpy, (mesh.vertices, mesh.faces))
    verts = verts.float()

    device = 'cuda'
    # R, T = pytorch3d.renderer.look_at_view_transform(-1, 0, 0)
    cameras = PerspectiveCameras(
        focal_length=[(1, 1)],
        principal_point=[(0, 0)],
        in_ndc=IN_NDC,
        # R=R,
        # T=T,
        image_size=[image_size],
    )

    # Equivalently 1:
    # TODO: why their full_projection_matrix differs?
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=90 ,R=R, T=T)


    # Equivalently 2:
    # for K, fx=fy=cx=cy= W/2

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0, faces_per_pixel=1)
    lights = PointLights(location=[[0, 0, 0]])
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=rasterizer, shader=shader).to(device)

    V, F = verts.shape[0], faces.shape[0]
    cube_map = torch.ones([1, 1, 3]) * torch.Tensor([0.65, 0.74, 0.86])
    verts = verts / 1
    cube_faceuv = torch.zeros([F, 3]).long()
    cube_vertuv = torch.zeros([1, 2])
    cube = Meshes(
        verts=[verts], faces=[faces],
        textures=TexturesUV(
            maps=[cube_map], faces_uvs=[cube_faceuv], verts_uvs=[cube_vertuv])
    ).to(device)

    images = renderer(cube)

    epylab.eimshow(images[0, :, :,  :])
    outfile = './tests/outputs/visualize_cube_with_unit_camera.png'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    epylab.savefig(outfile)


def render_instance_id_test():
    """
    Note:
    In pytorch3d, `in_ndc=False` means the units are defined in screen space,
    which means the units are pixels;
    However, we normally define the units in world coordinates,
    which hardly can have pixels units, therefore `in_ndc` should be set to True.
    """
    device = 'cuda'
    IN_NDC = True  # Setting in_ndc=True is very important
    image_size = (200, 200)
    def canonical_cuboids_pytorch3d_mesh(x, y, z, w, h, d, device):
        mesh = example_meshes.canonical_cuboids(x, y, z, w, h, d,
            convention='opencv')
        cube = mesh.synced_mesh.to(device)
        return cube

    cube1 = canonical_cuboids_pytorch3d_mesh(
        x=0, y=0, z=3,
        w=2, h=2, d=2, device=device)
    cube2 = canonical_cuboids_pytorch3d_mesh(
        x=2, y=0, z=4,
        w=2, h=2, d=2, device=device)

    # R, T = pytorch3d.renderer.look_at_view_transform(-1, 0, 0)
    cameras = PerspectiveCameras(
        focal_length=[(1, 1)],
        principal_point=[(0, 0)],
        in_ndc=IN_NDC,
        # R=R, T=T,
        image_size=[image_size],
    )

    renderer = InstanceIDRenderer(cameras=cameras, image_size=image_size).to(device)
    images = renderer([cube1, cube2])

    epylab.eimshow(images)
    epylab.colorbar()
    outfile = './tests/outputs/render_instance_id.png'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    epylab.savefig(outfile)


if __name__ == '__main__':
    visualize_cube_with_unit_camera()
    render_instance_id_test()
    unittest.main()
