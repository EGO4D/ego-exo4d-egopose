import math
from typing import Optional

import av
import torch
from torchvision.transforms import Resize


def get_video_meta(path):
    with av.open(path) as cont:
        n_frames = cont.streams[0].frames
        codec = cont.streams[0].codec.name
        tb = cont.streams[0].time_base

        all_pts = []
        for x in cont.demux(video=0):
            if x.pts is None:
                continue
            all_pts.append(x.pts)

        assert len(all_pts) == n_frames
        return {
            "all_pts": sorted(all_pts),
            "codec": codec,
            "tb": tb,
        }


class StridedReader:
    def __init__(self, path, stride, frame_window_size):
        self.path = path
        self.meta = get_video_meta(path)
        self.all_pts = self.meta["all_pts"]
        self.stride = stride
        self.frame_window_size = frame_window_size
        if self.stride == 0:
            self.stride = self.frame_window_size

    def __getitem__(self, _: int) -> torch.Tensor:
        raise AssertionError("Not implemented")

    def __len__(self):
        return int(
            math.ceil((len(self.all_pts) - self.frame_window_size) / self.stride)
        )


class PyAvReader(StridedReader):
    def __init__(
        self,
        path: str,
        resize: Optional[int],
        mean: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        if gpu_idx >= 0:
            raise AssertionError("GPU decoding not support for pyav")

        self.mean = mean
        self.resize = Resize((resize, resize)) if resize is not None else None
        self.path = path
        self.create_underlying_cont(gpu_idx)

    def create_underlying_cont(self, gpu_id):
        self.cont = av.open(self.path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        frame_j_pts = self.all_pts[frame_j]
        # self.cont.streams.video[0].seek(frame_i_pts)
        self.cont.seek(frame_i_pts, stream=self.cont.streams.video[0])
        fs = []
        for f in self.cont.decode(video=0):
            # print(f.pts)
            if f.pts < frame_i_pts:
                continue
            if f.pts >= frame_j_pts:
                break
            fs.append((f.pts, torch.tensor(f.to_ndarray(format="rgb24"))))
        fs = sorted(fs, key=lambda x: x[0])
        ret = torch.stack([x[1] for x in fs])
        if self.resize is not None:
            ret = self.resize(ret)
        if self.mean is not None:
            ret -= self.mean
        return ret
