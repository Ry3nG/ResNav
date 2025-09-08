"""Collision grid utilities: inflate occupancy for robot radius."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _disk_kernel(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    size = 2 * r + 1
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (xx * xx + yy * yy) <= (r * r)
    return mask.astype(bool)


def inflate_grid(grid: np.ndarray, radius_m: float, resolution_m: float) -> np.ndarray:
    """Inflate boolean occupancy by a circular kernel of robot radius.

    Args:
        grid: 2D bool array (True=occupied).
        radius_m: robot radius in meters.
        resolution_m: meters per cell.

    Returns:
        2D bool array of same shape; True where inflated occupancy.
    """
    assert grid.ndim == 2 and grid.dtype == bool
    r_cells = int(np.ceil(max(0.0, float(radius_m)) / float(resolution_m)))
    if r_cells <= 0:
        return grid.copy()
    # Try SciPy first
    try:
        from scipy.ndimage import binary_dilation  # type: ignore

        kernel = _disk_kernel(r_cells)
        return binary_dilation(grid, structure=kernel)
    except Exception:
        # Fallbacks without relying on SciPy availability
        H, W = grid.shape
        out = np.zeros_like(grid)
        # Square kernel dilation via rolling OR
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                # Compute valid slice ranges for shifted copy
                i_src0 = max(0, -di)
                i_src1 = min(H, H - di)
                j_src0 = max(0, -dj)
                j_src1 = min(W, W - dj)
                i_dst0 = max(0, di)
                i_dst1 = i_dst0 + (i_src1 - i_src0)
                j_dst0 = max(0, dj)
                j_dst1 = j_dst0 + (j_src1 - j_src0)
                if i_src0 < i_src1 and j_src0 < j_src1:
                    out[i_dst0:i_dst1, j_dst0:j_dst1] |= grid[i_src0:i_src1, j_src0:j_src1]
        return out
