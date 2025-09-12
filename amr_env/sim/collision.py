"""Collision grid utilities: inflate occupancy for robot radius."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation


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
    kernel = _disk_kernel(r_cells)
    return binary_dilation(grid, structure=kernel)
