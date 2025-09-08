"""Video export utilities for saving rendered frames to MP4/GIF."""

from __future__ import annotations

from typing import Sequence


def save_mp4(frames: Sequence, path: str, fps: int = 20) -> None:
    """Save a sequence of pygame surfaces or numpy arrays to an MP4 file.

    Requires imageio[ffmpeg] at runtime; otherwise raises a helpful error.
    """
    try:
        import numpy as np
        import imageio.v3 as iio  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("imageio[ffmpeg] is required to save MP4 videos") from e

    # Convert frames to numpy arrays (H, W, 3)
    imgs = []
    for f in frames:
        # Pygame surface -> string buffer -> numpy
        try:
            import pygame
            arr = pygame.surfarray.array3d(f)
            # pygame returns (W, H, 3); transpose to (H, W, 3)
            imgs.append(arr.transpose(1, 0, 2))
        except Exception:
            # Assume it's already ndarray-like
            imgs.append(f)

    iio.imwrite(path, imgs, fps=fps)
