"""Scenario generators for training and evaluation.

Phase I: temporary blockage corridor with single-side passage guarantee.
Outputs an occupancy grid, straight global path waypoints, start/goal, and info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class BlockageScenarioConfig:
    """Config for a simple corridor with temporary blockages (pallets)."""

    map_width_m: float = 50.0
    map_height_m: float = 50.0
    corridor_width_min_m: float = 3.0
    corridor_width_max_m: float = 4.0
    wall_thickness_m: float = 0.3
    pallet_width_m: float = 1.1
    pallet_length_m: float = 0.6
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.3
    resolution_m: float = 0.2
    min_passage_m: float = 0.7  # e.g., robot_diameter + 0.2 (0.5 + 0.2)
    min_pallet_x_offset_m: float = 0.6
    num_pallets_min: int = 1
    num_pallets_max: int = 1


def create_blockage_scenario(
    cfg: Optional[BlockageScenarioConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    np.ndarray,  # occupancy grid: True=occupied
    np.ndarray,  # waypoints: shape (N, 2)
    tuple[float, float, float],  # start pose (x, y, theta)
    tuple[float, float],  # goal xy
    dict[str, float],  # info dict with metadata
]:
    """Generate a blockage-only scenario with corridor and pallets.

    Grid convention: True = occupied; indices [row, col] = [y, x].
    """

    c = cfg or BlockageScenarioConfig()
    res = float(c.resolution_m)
    grid_w = int(round(c.map_width_m / res))
    grid_h = int(round(c.map_height_m / res))
    grid = np.zeros((grid_h, grid_w), dtype=bool)
    r = rng or np.random.default_rng()

    corridor_w = float(r.uniform(c.corridor_width_min_m, c.corridor_width_max_m))
    pallet_w_eff = float(c.pallet_width_m)
    cy = c.map_height_m / 2.0
    y_top = cy + corridor_w / 2.0
    y_bot = cy - corridor_w / 2.0

    def fill_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        i0 = int(np.clip(math.floor(y0 / res), 0, grid_h - 1))
        i1 = int(np.clip(math.floor(y1 / res), 0, grid_h - 1))
        j0 = int(np.clip(math.floor(x0 / res), 0, grid_w - 1))
        j1 = int(np.clip(math.floor(x1 / res), 0, grid_w - 1))
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        grid[i0 : i1 + 1, j0 : j1 + 1] = True

    # Draw corridor walls
    fill_rect(0.0, y_top, c.map_width_m, y_top + c.wall_thickness_m)
    fill_rect(0.0, y_bot - c.wall_thickness_m, c.map_width_m, y_bot)

    pallets = []
    pallet_sizes = []

    # Longitudinal bounds with margins
    start_x = float(c.start_x_m)
    goal_x = float(c.map_width_m - c.goal_margin_x_m)
    x_lo = start_x + float(getattr(c, "min_pallet_x_offset_m", 0.6))
    x_hi = goal_x - 0.6

    # Sample number of pallets
    num_pallets = int(r.integers(c.num_pallets_min, c.num_pallets_max + 1))
    for _ in range(num_pallets):
        # Single-side minimum passage T
        T = float(c.min_passage_m)
        eps = 1e-3
        w_min = max(0.2, 0.4 * pallet_w_eff)
        w_max_geom = max(0.0, corridor_w - T - eps)
        w_cap = max(w_min, min(pallet_w_eff, w_max_geom))
        if w_cap <= 0.0:
            w_i = 0.2
        else:
            w_i = float(r.uniform(w_min, w_cap))

        l_min = max(0.3, 0.5 * c.pallet_length_m)
        l_i = float(r.uniform(l_min, c.pallet_length_m))

        # Choose top/bottom bias ensuring >= T free on the other side
        toward_top = bool(r.random() < 0.5)
        if toward_top:
            y_lo = float(y_bot + T + w_i / 2.0)
            y_hi = float(y_top - w_i / 2.0)
        else:
            y_lo = float(y_bot + w_i / 2.0)
            y_hi = float(y_top - T - w_i / 2.0)
        if y_hi < y_lo:
            mid = 0.5 * (y_bot + y_top)
            y_center = float(np.clip(mid, y_bot + w_i / 2.0, y_top - w_i / 2.0))
        else:
            y_center = float(r.uniform(y_lo, y_hi))

        if x_hi > x_lo:
            x_center = float(r.uniform(x_lo, x_hi))
        else:
            x_center = (start_x + goal_x) / 2.0

        fill_rect(
            x_center - l_i / 2.0,
            y_center - w_i / 2.0,
            x_center + l_i / 2.0,
            y_center + w_i / 2.0,
        )
        pallets.append((x_center, y_center))
        pallet_sizes.append((l_i, w_i))

    # Path and endpoints along corridor centerline
    start_y = cy
    goal_y = cy
    n_wp = max(2, int(round((goal_x - start_x) / max(1e-6, c.waypoint_step_m))))
    xs = np.linspace(start_x, goal_x, n_wp)
    ys = np.full_like(xs, start_y)
    waypoints = np.stack([xs, ys], axis=1)

    info = {
        "corridor_width": float(corridor_w),
        "corridor_width_final": float(corridor_w),
        "pallet_width_final": float(pallet_w_eff),
        "num_pallets": int(num_pallets),
        "pallet_centers": [(float(px), float(py)) for (px, py) in pallets],
        "pallet_sizes": [(float(pl), float(pw)) for (pl, pw) in pallet_sizes],
    }
    start_pose = (float(start_x), float(start_y), 0.0)
    goal_xy = (float(goal_x), float(goal_y))
    return grid, waypoints, start_pose, goal_xy, info
