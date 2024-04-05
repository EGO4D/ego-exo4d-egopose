import unittest
import numpy as np
import torch

from libzhifan.odlib import box_utils


class BoxUitlsTest(unittest.TestCase):

    def test_xyxy_and_xywh(self):
        box_1d_xyxy = np.float32([2, 3, 5, 7])
        box_1d_xywh_act = box_utils.xyxy_to_xywh(box_1d_xyxy)
        box_1d_xywh = np.float32([2, 3, 3, 4])
        np.testing.assert_equal(box_1d_xywh_act, box_1d_xywh)

        box_2d_xyxy = box_1d_xyxy[None]
        box_2d_xywh_act = box_utils.xyxy_to_xywh(box_2d_xyxy)
        box_2d_xywh = np.float32([[2, 3, 3, 4]])
        np.testing.assert_equal(box_2d_xywh_act, box_2d_xywh)

        box_3d_xyxy = np.float32([
            [
                [2, 3, 5, 7],
                [11, 13, 17, 19]
            ],
            [
                [23, 27, 29, 31],
                [37, 41, 43, 47],
            ]
        ])
        box_3d_xywh_act = box_utils.xyxy_to_xywh(box_3d_xyxy)
        box_3d_xywh = np.float32([
            [
                [2, 3, 3, 4],
                [11, 13, 6, 6]
            ],
            [
                [23, 27, 6, 4],
                [37, 41, 6, 6],
            ]
        ])
        np.testing.assert_equal(box_3d_xywh_act, box_3d_xywh)

        box_1d_xyxy_th = torch.as_tensor(box_1d_xyxy)
        box_1d_xywh_th = torch.FloatTensor([2, 3, 3, 4])
        torch.testing.assert_allclose(box_utils.xyxy_to_xywh(box_1d_xyxy_th), box_1d_xywh_th)

        box_3d_xywh_th = torch.as_tensor(box_3d_xywh)
        box_3d_xyxy_th = torch.as_tensor(box_3d_xyxy)
        torch.testing.assert_allclose(box_utils.xywh_to_xyxy(box_3d_xywh_th), box_3d_xyxy_th)
    
    def test_yxyx(self):
        # TODO?
        pass

    def test_xcycwh(self):
        pass

    def test_overall(self):
        box_2d_xyxy = np.float32([
            [
                [2, 3, 5, 7],
                [11, 13, 17, 19]
            ],
            [
                [23, 27, 29, 31],
                [37, 41, 43, 47],
            ]
        ])
        np.testing.assert_equal(
            box_utils.xcycwh_to_xyxy( 
                box_utils.xywh_to_xcycwh(
                    box_utils.xyxy_to_xywh(box_2d_xyxy)
                )
            ),
            box_2d_xyxy.copy()
        )
        torch.testing.assert_allclose(
            box_utils.xcycwh_to_xyxy( 
                box_utils.xywh_to_xcycwh(
                    box_utils.xyxy_to_xywh(
                        torch.as_tensor(box_2d_xyxy))
                )
            ),
            torch.as_tensor(box_2d_xyxy).clone()
        )
            


if __name__ == '__main__':
    unittest.main()