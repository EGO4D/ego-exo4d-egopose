import numpy as np
import cv2
from moviepy import editor


def frame_subtitle(image,
                   subtitle,
                   subtitle_height=120,
                   subtitle_color=(255, 255, 255),  # white
                   subtitle_thickness=3
                   ) -> np.ndarray:
    sub_canvas = np.zeros([subtitle_height, image.shape[1], 3], dtype=image.dtype)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2
    sub_canvas = cv2.putText(sub_canvas, f"{subtitle}", (10, 45),
                            font, fontScale, subtitle_color, subtitle_thickness, cv2.LINE_AA)
    frame = np.vstack([image, sub_canvas])
    return frame


def video_subtitle(images, subtitles, fps:int,
                   subtitle_height=120,
                   subtitle_color=(255, 255, 255),  # white
                   subtitle_thickness=3):
    """
    Args:
        images: list of images
        subtitles: list of subtitles
    """
    assert len(images) == len(subtitles)

    frames = []
    for i, (image, string) in enumerate(zip(images, subtitles)):
        frame = frame_subtitle(image, string, subtitle_height=subtitle_height,
                               subtitle_color=subtitle_color, subtitle_thickness=subtitle_thickness)
        frames.append(frame)

    clip = editor.ImageSequenceClip(frames, fps=fps)
    return clip
