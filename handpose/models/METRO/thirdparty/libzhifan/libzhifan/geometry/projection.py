from typing import Union, Tuple, List
import numpy as np
import trimesh
import torch

from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, PointLights,
    MeshRasterizer, SoftPhongShader, MeshRenderer,
    TexturesVertex
)
from pytorch3d.renderer import BlendParams, SoftSilhouetteShader
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_scene

from . import coor_utils
from .mesh import SimpleMesh
from .visualize_2d import draw_dots_image
from .camera_manager import CameraManager
from .instance_id_rendering import InstanceIDRenderer

from libzhifan.numeric import numpize

try:
    import neural_renderer as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False


"""
Dealing with vertices projections, possibly using pytorch3d

Pytorch3D has two Perspective Cameras:

- pytorch3d.render.FovPerspectiveCameras(znear, zfar, ar, fov)

- pytorch3d.render.PerspectiveCameras(focal_length, principal_point)
    - or pytorch3d.render.PerspectiveCameras(K: (4,4))


1. Coordinate System.

Camera always looks at (0, 0, 1).

a. pytorch3d / pytorch3d-NDC

            ^ Y
            |
            |   / Z
            |  /
            | /
    X <------

    - this will be projected into:

            ^ Y
            |
            |
        <----
        X

b. OpenGL, naive_implementation:

            ^ Y              Y ^
            |                  |  / Z
            |                  | /
            |                  |/
            /------> X          ------> X
           /                     (NDC)
          /
       Z /

    - this will be projected into:

            ----> X
            |
            |
            v Y

c. OpenCV, Open3D, neural_renderer:

             / Z
            /
           /
          /
         ----------> X
         |
         |
         |
         v Y


2. Projection transforms.

In pytorch3d, the transforms are as follows:
model -> view -> ndc -> screen
In pinhole camera model, i.e. with simple 3x3 matrix K, the transforms is:
model -> screen


3. Rendering configuration

To render a cube [-1, 1]^3, on a W x W = (200, 200) image

naive method:
    - fx=fy=cx=cy=W/2, image_size=(W, W)

pytorch3d in_ndc=True:
    - fx=fy=1, cx=cy=0, image_size=(W, W)

pytorch3d in_ndc=False:
    - fx=fy=cx=cy=W/2, image_size=(W, W)

neural_renderer.git:
    - fx=fy=cx=cy=W/2, image_size=W, orig_size=W
    or,
    - fx=fy=cx=cy=1/2, image_size=W, orig_size=1

`naive` == `pytorch3d in_ndc=False` == `neural_renderer w/ orig_size=image_size`


Ref:
[1] https://medium.com/maochinn/%E7%AD%86%E8%A8%98-camera-dee562610e71https://medium.com/maochinn/%E7%AD%86%E8%A8%98-camera-dee562610e71

"""

AnyMesh = Union[SimpleMesh, Meshes, List[Meshes], List[SimpleMesh]]

_R = torch.eye(3)
_T = torch.zeros(3)


def _to_th_mesh(m: AnyMesh) -> Meshes:
    if isinstance(m, list):
        l = [v for v in m if v is not None]
        return join_meshes_as_scene(list(map(_to_th_mesh, l)))
    elif isinstance(m, Meshes):
        return m
    elif isinstance(m, SimpleMesh):
        return m.synced_mesh
    else:
        raise ValueError(f"type {type(m)} not understood.")


def perspective_projection(mesh_data: AnyMesh,
                           cam_f: Tuple[float],
                           cam_p: Tuple[float],
                           method=dict(
                               name='pytorch3d',
                               ),
                           image=None,
                           img_h=None,
                           img_w=None) -> np.ndarray:
    """ Project verts/mesh by Perspective camera.

    Args:
        mesh_data: one of
            - SimpleMesh
            - pytorch3d.Meshes
            - list of SimpleMeshes
            - list of pytorch3d.Meshes
        cam_f: focal length (2,)
        cam_p: principal points (2,)
        method: dict
            - name: one of {'naive', 'pytorch3d', 'pytorch3d_instance', 'neural_renderer'}.

            Other fields contains the parameters of that function

            Camera by default located at (0, 0, 0) and looks following z-axis.

        in_ndc: bool
        R: (3, 3) camera extrinsic matrix.
        T: (3,) camera extrinsic matrix.
        image: (H, W, 3), if `image` is None,
            will render a image with size (img_h, img_w).

    Returns:
        (H, W, 3) image
    """
    method_name = method.pop('name')
    if image is None:
        assert img_h is not None and img_w is not None
        image = np.ones([img_h, img_w, 3], dtype=np.uint8) * 255

    if method_name == 'naive':
        return naive_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p, image=image,
            **method,
        )
    elif method_name == 'pytorch3d':
        image = torch.as_tensor(
            image, dtype=torch.float32) / 255.
        img = pytorch3d_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image
        )
        return img
    elif method_name == 'pytorch3d_silhouette':
        image = torch.as_tensor(
            image, dtype=torch.float32) / 255.
        img = pth3d_silhouette_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image)
        return img
    elif method_name == 'pytorch3d_instance':
        blur_radius = method.pop('blur_radius', 1e-7)
        img = pth3d_instance_perspective_projection(
            meshes=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, img_h=img_h, img_w=img_w, blur_radius=blur_radius)
        return img
    elif method_name == 'neural_renderer' or method_name == 'nr':
        assert HAS_NR
        img = neural_renderer_perspective_projection(
            mesh_data=mesh_data, cam_f=cam_f, cam_p=cam_p,
            **method, image=image
        )
        return img
    else:
        raise ValueError(f"method_name: {method_name} not understood.")


def perspective_projection_by_camera(mesh_data: AnyMesh,
                                     camera: CameraManager,
                                     method=dict(
                                         name='pytorch3d',
                                         in_ndc=False,
                                     ),
                                     image=None) -> np.ndarray:
    """
    Similar to perspective_projection() but with CameraManager as argument.
    """
    fx, fy, cx, cy, img_h, img_w = camera.unpack()
    assert method.get('in_ndc', False) == False, "in_ndc Must be False for CamaraManager"
    img = perspective_projection(
        mesh_data,
        cam_f=(fx, fy),
        cam_p=(cx, cy),
        method=method.copy(),  # Avoid being optimized by python
        image=image,
        img_h=int(img_h),
        img_w=int(img_w),
    )
    return img


def naive_perspective_projection(mesh_data: AnyMesh,
                                 cam_f,
                                 cam_p,
                                 image,
                                 color='green',
                                 thickness=4,
                                 **kwargs):
    """
    Given image size, naive calculation of K should be

    fx = cx = img_w/2, fy = cy = img_h/2

    """
    if isinstance(mesh_data, list):
        raise NotImplementedError
    elif isinstance(mesh_data, Meshes):
        raise NotImplementedError

    verts = mesh_data.vertices
    fx, fy = cam_f
    cx, cy = cam_p
    fx, fy, cx, cy = map(float, (fx, fy, cx, cy))
    points = coor_utils.project_3d_2d(
        verts.T, fx=fx, fy=fy, cx=cx, cy=cy).T
    img = draw_dots_image(
        image, points, color=color, thickness=thickness)
    return img


def pytorch3d_perspective_projection(mesh_data: AnyMesh,
                                     cam_f,
                                     cam_p,
                                     in_ndc: bool,
                                     coor_sys='pytorch3d',
                                     R=_R,
                                     T=_T,
                                     image=None,
                                     flip_canvas_xy=False,
                                     **kwargs) -> np.ndarray:
    """
    TODO
    flip issue: https://github.com/facebookresearch/pytorch3d/issues/78

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]
        cam_f: Tuple, (2,)
        cam_p: Tuple, (2,)
        R: (3, 3)
        T: (3,)

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

        flip_canvas_xy: see flip issue. Note the issue doesn't happen
            if coor_sys == 'nr'
    """
    device = 'cuda'
    image_size = image.shape[:2]
    _mesh_data = _to_th_mesh(mesh_data)
    _mesh_data = _mesh_data.to(device)

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        _mesh_data = coor_utils.torch3d_apply_transform_matrix(
            _mesh_data, _Rz_mat)
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0, faces_per_pixel=1)
    lights = PointLights(location=[[0, 0, 0]])
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=rasterizer, shader=shader).to(device)

    rendered = renderer(_mesh_data)

    # Add background image
    if image is not None:
        image = image.to(device)
        frags = renderer.rasterizer(_mesh_data)
        is_bg = frags.pix_to_face[..., 0] < 0
        dst = rendered[..., :3]
        mask = is_bg[..., None].repeat(1, 1, 1, 3)
        out = dst.masked_scatter(
            mask, image[None][mask])
        out = numpize(out.squeeze_(0))
    else:
        out = numpize(rendered.squeeze_(0))[..., :3]
    return out


def pth3d_silhouette_perspective_projection(mesh_data: AnyMesh,
                                            cam_f,
                                            cam_p,
                                            in_ndc: bool,
                                            coor_sys='pytorch3d',
                                            R=_R,
                                            T=_T,
                                            image=None,
                                            **kwargs) -> np.ndarray:
    """

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

    """
    device = 'cuda'
    image_size = image.shape[:2]
    _mesh_data = _to_th_mesh(mesh_data)
    _mesh_data = _mesh_data.to(device)

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        _mesh_data = coor_utils.torch3d_apply_transform_matrix(
            _mesh_data, _Rz_mat)
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )
    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-9, gamma=1e-9)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=1,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    renderer = silhouette_renderer.to(device)

    # raster_settings = RasterizationSettings(
    #     image_size=image_size, blur_radius=0, faces_per_pixel=1)
    # lights = PointLights(location=[[0, 0, 0]])
    # rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # shader = SoftPhongShader(cameras=cameras, lights=lights)
    # renderer = MeshRenderer(
    #     rasterizer=rasterizer, shader=shader).to(device)

    rendered = renderer(_mesh_data)

    # Add background image
    out = numpize(rendered.squeeze())[..., -1]
    return out


def pth3d_instance_perspective_projection(meshes: List[Meshes],
                                          cam_f,
                                          cam_p,
                                          in_ndc: bool,
                                          img_h,
                                          img_w,
                                          coor_sys='pytorch3d',
                                          R=_R,
                                          T=_T,
                                          **kwargs) -> np.ndarray:
    """ Instance ID

    Args:
        image: (H, W, 3) torch.Tensor with values in [0, 1]

        coor_sys: str, one of {'pytorch3d', 'neural_renderer'/'nr'}
            Set the input coordinate sysem.
            - 'pytorch3d': render using pytorch3d coordinate system,
                i.e. X-left, Y-top, Z-in
            - 'neural_renderer'/'nr':
                    X-right, Y-down, Z-in.

    Returns:
        instance_id_mask: (H, W) int32
    """
    device = 'cuda'
    image_size = (img_h, img_w)
    assert type(meshes) == list, "Must be list of meshes"
    meshes = [_to_th_mesh(v).to(device) for v in meshes]

    if coor_sys == 'pytorch3d':
        pass  # Nothing
    elif coor_sys == 'neural_renderer' or coor_sys == 'nr':
        # flip XY is the same as Rotation around Z
        _Rz_mat = torch.as_tensor([[
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
        meshes = [
            coor_utils.torch3d_apply_transform_matrix(v,_Rz_mat)
            for v in meshes]
    else:
        raise ValueError(f"coor_sys '{coor_sys}' not understood.")

    R = torch.unsqueeze(torch.as_tensor(R), 0)
    T = torch.unsqueeze(torch.as_tensor(T), 0)
    cameras = PerspectiveCameras(
        focal_length=[cam_f],
        principal_point=[cam_p],
        in_ndc=in_ndc,
        R=R,
        T=T,
        image_size=[image_size],
    )
    blur_radius = kwargs.pop('blur_radius', 1e-7)
    renderer = InstanceIDRenderer(cameras=cameras, image_size=image_size, blur_radius=blur_radius).to(device)
    rendered = renderer(meshes)

    out = numpize(rendered)
    return out


def neural_renderer_perspective_projection(mesh_data: SimpleMesh,
                                           cam_f,
                                           cam_p,
                                           R=_R,
                                           T=_T,
                                           image=None,
                                           orig_size=None,
                                           **kwargs):
    """
    TODO(low priority): add image support, add texture render support.

    Args:
        orig_size: int or None.
            if None, orig_size will be set to image_size.
            It's recommended to keep it as None.
            See above "3." for explanation.
    """
    device = 'cuda'
    if isinstance(mesh_data, list):
        raise NotImplementedError
    elif isinstance(mesh_data, Meshes):
        raise NotImplementedError

    verts = torch.as_tensor(
        mesh_data.vertices, device=device, dtype=torch.float32).unsqueeze(0)
    faces = torch.as_tensor(mesh_data.faces, device=device).unsqueeze(0)
    image_size = image.shape
    fx, fy = cam_f
    cx, cy = cam_p

    K = torch.as_tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
        ], dtype=torch.float32, device=device)
    K = K[None]
    R = torch.eye(3, device=device)[None]
    t = torch.zeros([1, 3], device=device)

    if orig_size is None:
        orig_size = image_size[0]
    renderer = nr.Renderer(
        image_size=image_size[0],
        K=K,
        R=R,
        t=t,
        orig_size=orig_size
    )

    img = renderer(
        verts,
        faces,
        mode='silhouettes'
    )
    return numpize(img)


def project_standardized(mesh_data: AnyMesh,
                         direction='+z',
                         image_size=200,
                         pad=0.2,
                         method=dict(
                             name='pytorch3d',
                             in_ndc=False,
                             coor_sys='nr'
                         ),
                         centering=True,
                         manual_dmax : float = None,
                         show_axis=False,
                         print_dmax=False) -> np.ndarray:
    """
    Given any mesh(es), this function renders the zoom-in images.
    The meshes are proecessed to be in [-0.5, 0.5]^3 space,
    then a weak-perspective camera is applied.

    Args:
        pad: the fraction to be padded around rendered image.
        manual_dmax: set dmax manually
        print_dmax: This helps determine manual_dmax
        centering: if True, look at (xc, yc, zc); otherwise, look at (0, 0, 0)

    Returns:
        (H, W, 3)
    """
    _mesh_data = _to_th_mesh(mesh_data)
    xmin, ymin, zmin = torch.min(_mesh_data.verts_packed(), 0).values
    xmax, ymax, zmax = torch.max(_mesh_data.verts_packed(), 0).values
    xc, yc, zc = map(lambda x: x/2, (xmin+xmax, ymin+ymax, zmin+zmax))
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
    dmax = max(dx, max(dy, dz)).item()
    if manual_dmax is not None:
        dmax = manual_dmax
    if print_dmax:
        print(f"dmax = {dmax}")

    if show_axis:
        device = _mesh_data.device
        _axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=dmax * 0.6)

        _ax_verts = torch.as_tensor(_axis.vertices, device=device, dtype=torch.float32)
        _ax_faces = torch.as_tensor(_axis.faces, device=device)
        _ax_verts_rgb = torch.as_tensor(_axis.visual.vertex_colors[:, :3], device=device)
        _ax_verts_rgb = _ax_verts_rgb / 255.
        textures = TexturesVertex(verts_features=_ax_verts_rgb[None].to(device))
        _axis = Meshes(
            verts=[_ax_verts],
            faces=[_ax_faces],
            textures=textures)
        _mesh_data = _to_th_mesh([_mesh_data, _axis])

    large_z = 20  # can be arbitrary large value >> 1
    if centering:
        _mesh_data = coor_utils.torch3d_apply_translation(
            _mesh_data, (-xc, -yc, -zc))
    _mesh_data = coor_utils.torch3d_apply_scale(_mesh_data, 1./dmax)
    if direction == '+z':
        pass  # Nothing need to be changed
    elif direction == '-z':
        _mesh_data = coor_utils.torch3d_apply_Ry(_mesh_data, 180)
    elif direction == '+x':
        # I guarantee you it's +90, not -90
        _mesh_data = coor_utils.torch3d_apply_Ry(_mesh_data, +90)
    elif direction == '-x':
        _mesh_data = coor_utils.torch3d_apply_Ry(_mesh_data, -90)
    elif direction == '+y':
        _mesh_data = coor_utils.torch3d_apply_Rx(_mesh_data, +90)
    elif direction == '-y':
        _mesh_data = coor_utils.torch3d_apply_Rx(_mesh_data, -90)
    else:
        raise ValueError("direction not understood.")
    _mesh_data = coor_utils.torch3d_apply_translation(
        _mesh_data, (0, 0, large_z))

    fx = fy = 2*large_z / (1+pad)
    camera = CameraManager(
        fx=fx, fy=fy,
        cx=0, cy=0, img_h=image_size, img_w=image_size,
        in_ndc=True,
    )
    return perspective_projection_by_camera(
        _mesh_data,
        camera,
        method=method)
