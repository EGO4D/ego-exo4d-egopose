""" A helper class that manages camera parameters

Ref on crop and resize:
https://github.com/BerkeleyAutomation/perception/blob/0.0.1/perception/camera_intrinsics.py#L176-#L236
"""

import numpy as np
import torch

""" CameraManager(Fx, Fy, Cx, Cy) is the composed version of intrinsic parameters:

[[Fx, 0, Cx],
 [0, Fy, Cy],
 [0,  0,  1]] 
 
 = 
 
[[f*sx, 0, ox],
 [0, f*sy, oy],
 [0,    0,  1]]

"""


class CameraManager:

    """
    By default, the parameter is in
    conventional non-NDC representation.

    use

    ```fx, fy, cx, cy = self.to_ndc()```
    or
    ```fx, fy, cx, cy = self.to_nr(orig_size)```

    to convert camera parameters to pytorch3d's NDC or neural_renderer's
    representation.

    """

    def __init__(self,
                 fx,
                 fy,
                 cx,
                 cy,
                 img_h,
                 img_w,
                 in_ndc=False):
        """

        Args:
            in_ndc (bool):
                If True, will assume {fx,fy,cx,cy} are in ndc format.

        """
        if in_ndc:
            half_h = img_h / 2
            half_w = img_w / 2
            fx = fx * half_w
            fy = fy * half_h
            cx = half_w * (cx + 1)  # W/2 * cx + W/2
            cy = half_h * (cy + 1)

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.img_h = int(img_h)
        self.img_w = int(img_w)

    def __repr__(self):
        return f"CameraManager (H, W) = ({self.img_h}, {self.img_w})\n"\
            f"K (non-NDC) = \n {self.get_K()}"

    def get_K(self):
        """ Returns: (3, 3) """
        K = np.float32([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        return K

    def unpack(self):
        return self.fx, self.fy, self.cx, self.cy, self.img_h, self.img_w
    
    def repeat(self, bsize: int, device='cpu'):
        fx = torch.ones([bsize]) * self.fx
        fy = torch.ones([bsize]) * self.fy
        cx = torch.ones([bsize]) * self.cx
        cy = torch.ones([bsize]) * self.cy
        img_h = torch.ones([bsize]) * self.img_h
        img_w = torch.ones([bsize]) * self.img_w
        return BatchCameraManager(
            fx=fx, fy=fy, cx=cx, cy=cy, img_h=img_h, img_w=img_w,
            in_ndc=False, device=device)
        
    @staticmethod
    def from_nr(mat, image_size: int):
        """
        Args:
            mat: (3, 3)
            image_size: H and W of neural_renderer's image
        Returns:
            CameraManager
        """
        _mat = image_size * np.asarray(mat).squeeze()
        fx, fy = _mat[..., 0, 0], _mat[..., 1, 1]
        cx, cy = _mat[..., 0, 2], _mat[..., 1, 2]
        return CameraManager(
            fx=fx, fy=fy, cx=cx, cy=cy, img_h=image_size, img_w=image_size
        )

    def to_ndc(self):
        half_h, half_w = self.img_h / 2, self.img_w / 2
        fx, fy = self.fx / half_w, self.fy / half_h
        cx, cy = self.cx/half_w - 1, self.cy/half_h - 1
        return fx, fy, cx, cy, self.img_h, self.img_w

    def to_nr(self, return_mat=False):
        """ Convert to neural renderer format. """
        fx, fy = self.fx / self.img_w, self.fy / self.img_h
        cx, cy = self.cx / self.img_w, self.cy / self.img_h
        if return_mat:
            return np.asarray([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ])
        else:
            return fx, fy, cx, cy, self.img_h, self.img_w

    def crop(self, crop_bbox):
        """
        Args:
            crop_bbox: (4,) x0y0wh
        """
        x0, y0, w_crop, h_crop = crop_bbox
        crop_center_x = x0 + w_crop/2
        crop_center_y = y0 + h_crop/2
        cx_updated = self.cx + w_crop/2 - crop_center_x
        cy_updated = self.cy + h_crop/2 - crop_center_y
        return CameraManager(
            fx=self.fx, fy=self.fy,
            cx=cx_updated, cy=cy_updated,
            img_h=h_crop, img_w=w_crop,
            in_ndc=False
        )

    def uncrop(self, crop_bbox, orig_h, orig_w):
        """
        The reverse of self.crop()
        """
        local_to_global = [
            - crop_bbox[0], - crop_bbox[1], orig_w, orig_h]
        return self.crop(local_to_global)

    def resize(self, new_h: int, new_w: int):
        scale_x = new_w / self.img_w
        scale_y = new_h / self.img_h
        fx = scale_x * self.fx
        fy = scale_y * self.fy
        cx = scale_x * self.cx
        cy = scale_y * self.cy
        return CameraManager(
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_h=new_h, img_w=new_w,
            in_ndc=False
        )

    def crop_and_resize(self, crop_bbox, output_size):
        """
        Crop a window (changing the center & boundary of scene),
        and resize the output image (implicitly chaning intrinsic matrix)

        Args:
            crop_bbox: (4,) x0y0wh
            output_size: tuple of (new_h, new_w) or int

        Returns:
            CameraManager
        """

        if isinstance(output_size, int):
            new_h = new_w = output_size
        elif len(output_size) == 2:
            new_h, new_w = output_size
        else:
            raise ValueError("output_size not understood.")

        return self.crop(crop_bbox).resize(new_h, new_w)


class BatchCameraManager:

    """
    Batched implementation of CameraManager,
    """

    def __init__(self,
                 fx: torch.Tensor,
                 fy: torch.Tensor,
                 cx: torch.Tensor,
                 cy: torch.Tensor,
                 img_h: torch.Tensor,
                 img_w: torch.Tensor,
                 in_ndc=False,
                 device='cpu'):
        """

        Args:
            fx, fy, cx, cy, img_h, img_w: torch.Tensor (B,)
            in_ndc (bool):
                If True, will assume {fx,fy,cx,cy} are in ndc format.

        """
        if in_ndc:
            half_h = img_h / 2
            half_w = img_w / 2
            fx = fx * half_w
            fy = fy * half_h
            cx = half_w * (cx + 1)  # W/2 * cx + W/2
            cy = half_h * (cy + 1)

        self.bsize = len(fx)  # batch size
        self.device = device
        self.fx = fx.float().to(self.device)
        self.fy = fy.float().to(self.device)
        self.cx = cx.float().to(self.device)
        self.cy = cy.float().to(self.device)
        self.img_h = img_h.int().to(self.device)
        self.img_w = img_w.int().to(self.device)

        self._check_shape()

    def _check_shape(self):
        assert self.fx.dim() == 1
        assert self.fy.dim() == 1
        assert self.cx.dim() == 1
        assert self.cy.dim() == 1
        assert self.img_h.dim() == 1
        assert self.img_w.dim() == 1
        assert self.fx.shape == \
            self.fy.shape == \
            self.cx.shape == \
            self.cy.shape == \
            self.img_h.shape == \
            self.img_w.shape

    def __repr__(self):
        return f"BatchCameraManager (H, W) = ({self.img_h}, {self.img_w})\n"\
            f"K (non-NDC) = \n {self.get_K()}"
    
    def __len__(self) -> int:
        return self.bsize

    def __getitem__(self, index: int) -> CameraManager:
        if index >= self.bsize:
            raise IndexError(f"Trying to access index {index} "
                             f"out of {self.bsize} cameras.")
        return CameraManager(
            fx=self.fx[index].item(), fy=self.fy[index].item(),
            cx=self.cx[index].item(), cy=self.cy[index].item(),
            img_h=self.img_h[index].item(), img_w=self.img_w[index].item(),
            in_ndc=False)

    def get_K(self):
        """ Returns: (B, 3, 3) """
        K = torch.zeros([self.bsize, 3, 3], dtype=torch.float32, device=self.device)
        K[:, 0, 0] = self.fx
        K[:, 0, 2] = self.cx
        K[:, 1, 1] = self.fy
        K[:, 1, 2] = self.cy
        K[:, 2, 2] = 1.0
        return K

    def unpack(self):
        return self.fx, self.fy, self.cx, self.cy, self.img_h, self.img_w

    @staticmethod
    def from_nr(mat, image_size: int, device: str):
        """
        Args:
            mat: (B, 3, 3)
            image_size: H and W of neural_renderer's image
        Returns:
            CameraManager
        """
        _mat = image_size * torch.as_tensor(mat)
        fx, fy = _mat[..., 0, 0], _mat[..., 1, 1]
        cx, cy = _mat[..., 0, 2], _mat[..., 1, 2]
        return BatchCameraManager(
            fx=fx, fy=fy, cx=cx, cy=cy, img_h=image_size, img_w=image_size,
            device=device
        )

    def to_ndc(self):
        half_h, half_w = self.img_h / 2, self.img_w / 2
        fx, fy = self.fx / half_w, self.fy / half_h
        cx, cy = self.cx/half_w - 1, self.cy/half_h - 1
        return fx, fy, cx, cy, self.img_h, self.img_w

    def to_nr(self, return_mat=False):
        """ Convert to neural renderer format. """
        fx, fy = self.fx / self.img_w, self.fy / self.img_h
        cx, cy = self.cx / self.img_w, self.cy / self.img_h
        if return_mat:
            K = torch.zeros([self.bsize, 3, 3], dtype=torch.float32, device=self.device)
            K[:, 0, 0] = fx
            K[:, 0, 2] = cx
            K[:, 1, 1] = fy
            K[:, 1, 2] = cy
            K[:, 2, 2] = 1.0
            return K
        else:
            return fx, fy, cx, cy, self.img_h, self.img_w

    def crop(self, crop_bbox):
        """
        Args:
            crop_bbox: (B, 4) x0y0wh corresponds to each camera
        """
        x0, y0, w_crop, h_crop = torch.split(crop_bbox, [1, 1, 1, 1], dim=1)
        x0 = x0.view(-1)
        y0 = y0.view(-1)
        w_crop = w_crop.view(-1)
        h_crop = h_crop.view(-1)
        crop_center_x = x0 + w_crop/2
        crop_center_y = y0 + h_crop/2
        cx_updated = self.cx + w_crop/2 - crop_center_x
        cy_updated = self.cy + h_crop/2 - crop_center_y
        return BatchCameraManager(
            fx=self.fx, fy=self.fy,
            cx=cx_updated, cy=cy_updated,
            img_h=h_crop, img_w=w_crop,
            in_ndc=False, device=self.device
        )

    def uncrop(self, crop_bbox: torch.Tensor, orig_h: int, orig_w: int):
        """
        The reverse of self.crop()

        Args:
            crop_bbox: (B, 4)
        """
        x0, y0, _, _ = torch.split(crop_bbox, [1, 1, 1, 1], dim=1)
        x0 = x0.view(-1)
        y0 = y0.view(-1)
        w = torch.ones_like(x0) * orig_w
        h = torch.ones_like(x0) * orig_h
        local_to_global = torch.stack([- x0, - y0, w, h], dim=1)
        return self.crop(local_to_global)

    def resize(self, new_h: torch.Tensor, new_w: torch.Tensor):
        scale_x = new_w / self.img_w
        scale_y = new_h / self.img_h
        fx = scale_x * self.fx
        fy = scale_y * self.fy
        cx = scale_x * self.cx
        cy = scale_y * self.cy
        return BatchCameraManager(
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_h=new_h, img_w=new_w,
            in_ndc=False, device=self.device
        )

    def crop_and_resize(self, crop_bbox, output_size):
        raise NotImplementedError(
            "Please call crop() and resize() explicitly")
