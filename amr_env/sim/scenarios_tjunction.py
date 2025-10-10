"""T-junction warehouse scenario for quick map extensibility experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np


@dataclass
class TJunctionConfig:
    """Configuration for a static T-junction corridor layout."""

    map_width_m: float = 20.0
    map_height_m: float = 20.0
    corridor_w_m: float = 3.0
    wall_th_m: float = 0.3
    resolution_m: float = 0.05
    waypoint_step_m: float = 0.3
    start_x_m: float = 1.0
    goal_x_m: float = 18.0


def create_tjunction(
    cfg: TJunctionConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    tuple[float, float],
    dict[str, float],
]:
    """Generate a deterministic T-junction occupancy grid and reference path."""

    res = float(cfg.resolution_m)
    grid_h = int(round(cfg.map_height_m / res))
    grid_w = int(round(cfg.map_width_m / res))
    grid = np.zeros((grid_h, grid_w), dtype=bool)
    corridor_w = float(cfg.corridor_w_m)
    wall_th = float(cfg.wall_th_m)
    cy = cfg.map_height_m / 2.0

    def fill_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        i0 = int(math.floor(y0 / res))
        i1 = int(math.floor(y1 / res))
        j0 = int(math.floor(x0 / res))
        j1 = int(math.floor(x1 / res))
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        i0 = max(0, min(grid_h - 1, i0))
        i1 = max(0, min(grid_h - 1, i1))
        j0 = max(0, min(grid_w - 1, j0))
        j1 = max(0, min(grid_w - 1, j1))
        grid[i0 : i1 + 1, j0 : j1 + 1] = True

    # Corridor stem horizontal walls
    y_top = cy + corridor_w / 2.0
    y_bot = cy - corridor_w / 2.0
    fill_rect(0.0, y_top, cfg.map_width_m, y_top + wall_th)
    fill_rect(0.0, y_bot - wall_th, cfg.map_width_m, y_bot)

    # Horizontal top bar of the junction
    bar_half = corridor_w / 2.0
    bar_x_lo = cfg.map_width_m / 2.0 - bar_half - wall_th
    bar_x_hi = cfg.map_width_m / 2.0 + bar_half + wall_th
    fill_rect(bar_x_lo, y_top, bar_x_hi, y_top + wall_th)

    start_x = float(cfg.start_x_m)
    goal_x = float(cfg.goal_x_m)
    xs = np.linspace(start_x, goal_x, int((goal_x - start_x) / cfg.waypoint_step_m) + 2)
    waypoints = np.stack([xs, np.full_like(xs, cy)], axis=1)
    start_pose = (start_x, cy, 0.0)
    goal_xy = (goal_x, cy)
    info = {"type": "t_junction", "corridor_width": corridor_w}
    return grid, waypoints, start_pose, goal_xy, info
