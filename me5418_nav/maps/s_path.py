from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from ..constants import (
    S_PATH_DEFAULT_RES,
    S_PATH_DEFAULT_AMP_M,
    S_PATH_DEFAULT_PERIODS,
    S_PATH_DEFAULT_OBS_FRAC,
    S_PATH_SAMPLE_STEP_M,
    START_GOAL_CLEAR_W_M,
    START_GOAL_CLEAR_H_M,
)


@dataclass
class SMapSpec:
    grid: BinaryOccupancyGrid
    waypoints: np.ndarray  # (M,2)
    start: Tuple[float, float, float]  # x,y,theta
    goal: Tuple[float, float]  # x,y


def make_s_map(
    size: Tuple[int, int] = (150, 120),
    res: float = S_PATH_DEFAULT_RES,
    amp: float = S_PATH_DEFAULT_AMP_M,
    periods: float = S_PATH_DEFAULT_PERIODS,
    obstacle_frac: float = S_PATH_DEFAULT_OBS_FRAC,
    rng: np.random.Generator | None = None,
) -> SMapSpec:
    """
    Create an S-shaped path across the map and place small obstacles along it.

    - size: (H,W) cells, res meters per cell
    - amp: sine amplitude in meters
    - periods: number of sine periods across map width
    - obstacle_frac: fraction of waypoints to block with small rectangles
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = size
    grid = BinaryOccupancyGrid(
        np.zeros((H, W), dtype=bool), cellsize=res, origin=(0, 0)
    )

    # Construct S-path in world coordinates across x in [1.0, xmax-1.0]
    xmin, xmax = 1.0, W * res - 1.0
    x = np.linspace(xmin, xmax, num=int((xmax - xmin) / S_PATH_SAMPLE_STEP_M))
    ymid = (H * res) / 2.0
    L = xmax - xmin
    y = ymid + amp * np.sin(2 * np.pi * periods * (x - xmin) / L)
    waypoints = np.stack([x, y], axis=1)

    # Place obstacles on a subset of waypoints
    n_wp = waypoints.shape[0]
    # choose obstacle indices away from start/goal by a margin
    margin = 30
    candidates = np.arange(margin, n_wp - margin)
    n_obs = int(max(1, obstacle_frac * candidates.size))
    # sample with spacing to avoid clustering
    if n_obs > 0 and candidates.size > 0:
        step = max(1, candidates.size // n_obs)
        sparse_idxs = candidates[::step]
        if sparse_idxs.size > n_obs:
            sparse_idxs = rng.choice(sparse_idxs, size=n_obs, replace=False)
    else:
        sparse_idxs = np.array([], dtype=int)

    for idx in sparse_idxs:
        cx, cy = waypoints[idx]
        # small rectangle around (cx,cy)
        w, h = 0.2, 0.2
        region = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        grid.set(region, True)

    # Define start at first waypoint, facing toward next
    x0, y0 = waypoints[0]
    x1, y1 = waypoints[min(1, n_wp - 1)]
    theta0 = float(np.arctan2(y1 - y0, x1 - x0))
    start = (float(x0), float(y0), theta0)
    goal = (float(waypoints[-1, 0]), float(waypoints[-1, 1]))

    # Ensure start and goal regions are free
    clear_w, clear_h = START_GOAL_CLEAR_W_M, START_GOAL_CLEAR_H_M
    grid.set(
        [x0 - clear_w / 2, y0 - clear_h / 2, x0 + clear_w / 2, y0 + clear_h / 2], False
    )
    grid.set(
        [
            goal[0] - clear_w / 2,
            goal[1] - clear_h / 2,
            goal[0] + clear_w / 2,
            goal[1] + clear_h / 2,
        ],
        False,
    )

    return SMapSpec(grid=grid, waypoints=waypoints, start=start, goal=goal)
