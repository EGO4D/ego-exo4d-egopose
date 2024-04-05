import unittest

import numpy as np
import torch
from libzhifan.geometry import example_meshes
from libzhifan.geometry import perspective_projection_by_camera
from libzhifan.geometry import CameraManager, BatchCameraManager


class CameraManagerTest(unittest.TestCase):
    
    def test_crop_and_resize(self):
        H, W = 200, 400
        image_size = (H, W)

        global_cam = CameraManager(
            fx=10, fy=20, cx=0, cy=0, img_h=H, img_w=W,
            in_ndc=True)

        H1, W1 = 200, 200
        local_box_1 = np.asarray([0, 0, H1, W1]) # xywh
        local_cam_1_exp = CameraManager(
            fx=20, fy=20, cx=1, cy=0, img_h=H1, img_w=W1,
            in_ndc=True)

        H2, W2 = 100, 100
        local_box_2 = np.asarray([200, 100, H2, W2]) # xywh
        local_cam_2_exp = CameraManager(
            fx=40, fy=40, cx=-1, cy=-1, img_h=H2, img_w=W2,
            in_ndc=True)

        cube_1 = example_meshes.canonical_cuboids(
            x=0.5, y=0, z=10.25,
            w=0.5, h=0.5, d=0.5,
            convention='pytorch3d'
        )
        cube_2 = example_meshes.canonical_cuboids(
            x=-0.375, y=-0.125, z=10.125,
            w=0.25, h=0.25, d=0.25,
            convention='pytorch3d'
        )

        np.testing.assert_allclose(
            local_cam_1_exp.get_K(),
            global_cam.crop(local_box_1).get_K())
        np.testing.assert_allclose(
            local_cam_2_exp.get_K(),
            global_cam.crop(local_box_2).get_K())
        
        """ image rendered by Local camera 1 
        x=0.5 => x_pix=100
        x=0.75 => x_pix=150 (=50 after flip)

        """

        img_global = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam)
        img_1 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_1))
        img_2 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_2))


    def test_batch_crop_and_resize(self):
        H, W = 200, 400

        global_cam = BatchCameraManager(
            fx=torch.as_tensor([10]), 
            fy=torch.as_tensor([20]), 
            cx=torch.as_tensor([0]), 
            cy=torch.as_tensor([0]), 
            img_h=torch.as_tensor([H]), 
            img_w=torch.as_tensor([W]),
            in_ndc=True)

        H1, W1 = 200, 200
        local_box_1 = torch.as_tensor([[0, 0, H1, W1]]) # xywh
        local_cam_1_exp = BatchCameraManager(
            fx=torch.as_tensor([20]),
            fy=torch.as_tensor([20]),
            cx=torch.as_tensor([1]), 
            cy=torch.as_tensor([0]),
            img_h=torch.as_tensor([H1]), 
            img_w=torch.as_tensor([W1]),
            in_ndc=True)

        H2, W2 = 100, 100
        local_box_2 = torch.as_tensor([[200, 100, H2, W2]]) # xywh
        local_cam_2_exp = BatchCameraManager(
            fx=torch.as_tensor([40]), 
            fy=torch.as_tensor([40]), 
            cx=torch.as_tensor([-1]), 
            cy=torch.as_tensor([-1]), 
            img_h=torch.as_tensor([H2]), 
            img_w=torch.as_tensor([W2]),
            in_ndc=True)

        cube_1 = example_meshes.canonical_cuboids(
            x=0.5, y=0, z=10.25,
            w=0.5, h=0.5, d=0.5,
            convention='pytorch3d'
        )
        cube_2 = example_meshes.canonical_cuboids(
            x=-0.375, y=-0.125, z=10.125,
            w=0.25, h=0.25, d=0.25,
            convention='pytorch3d'
        )

        torch.testing.assert_allclose(
            local_cam_1_exp.get_K(),
            global_cam.crop(local_box_1).get_K())
        torch.testing.assert_allclose(
            local_cam_2_exp.get_K(),
            global_cam.crop(local_box_2).get_K())
        
        """ image rendered by Local camera 1 
        x=0.5 => x_pix=100
        x=0.75 => x_pix=150 (=50 after flip)

        """
        img_global = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam[0])
        img_1 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_1)[0])
        img_2 = perspective_projection_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_2)[0])


if __name__ == '__main__':
    unittest.main()
        