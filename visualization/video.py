"""Video export utilities for saving rendered frames to MP4/GIF."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pygame
import imageio.v3 as iio


def save_mp4(frames: Sequence, path: str, fps: int = 20) -> None:
    """Save a sequence of pygame surfaces or numpy arrays to an MP4 file."""
    imgs = []
    for frame in frames:
        if hasattr(frame, "get_size"):  # pygame surface
            arr = pygame.surfarray.array3d(frame)
            # pygame returns (W, H, 3); transpose to (H, W, 3)
            imgs.append(arr.transpose(1, 0, 2))
        else:  # assume numpy array
            imgs.append(frame)

    iio.imwrite(path, imgs, fps=fps)
