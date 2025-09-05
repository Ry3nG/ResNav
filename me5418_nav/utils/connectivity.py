from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


def world_to_cell(x: float, y: float, res: float, H: int, W: int) -> Tuple[int, int]:
    """Convert world meters (x,y) to grid indices (i,j) with clipping.

    i increases downwards, j increases to the right.
    """
    i = int(np.clip(np.floor(y / float(res)), 0, H - 1))
    j = int(np.clip(np.floor(x / float(res)), 0, W - 1))
    return i, j


def cells_connected_free(
    grid: np.ndarray,
    start_ij: Tuple[int, int],
    goal_ij: Tuple[int, int],
    connectivity: int = 8,
) -> bool:
    """Return True if start and goal are connected through free (False) cells.

    grid: bool array with True for occupied, False for free.
    connectivity: 4 or 8.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    H, W = grid.shape
    si, sj = start_ij
    gi, gj = goal_ij
    if si < 0 or sj < 0 or si >= H or sj >= W:
        return False
    if gi < 0 or gj < 0 or gi >= H or gj >= W:
        return False
    if grid[si, sj] or grid[gi, gj]:
        return False

    if connectivity == 4:
        neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif connectivity == 8:
        neigh = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    q: deque[Tuple[int, int]] = deque()
    q.append((si, sj))
    visited = np.zeros_like(grid, dtype=np.uint8)
    visited[si, sj] = 1

    while q:
        i, j = q.popleft()
        if i == gi and j == gj:
            return True
        for di, dj in neigh:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W and not grid[ni, nj] and not visited[ni, nj]:
                visited[ni, nj] = 1
                q.append((ni, nj))
    return False


def reachable_inflated(
    grid_infl: np.ndarray,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    res: float,
) -> bool:
    """Check connectivity on an inflated occupancy grid in world meters.

    Returns True if there exists an 8-connected free path between start and goal.
    """
    H, W = grid_infl.shape
    si, sj = world_to_cell(start_xy[0], start_xy[1], res, H, W)
    gi, gj = world_to_cell(goal_xy[0], goal_xy[1], res, H, W)
    return cells_connected_free(grid_infl, (si, sj), (gi, gj), connectivity=8)


__all__ = [
    "world_to_cell",
    "cells_connected_free",
    "reachable_inflated",
]
