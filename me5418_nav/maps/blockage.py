from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from ..constants import GRID_RESOLUTION_M


@dataclass
class BlockageScenarioConfig:
    """
    Config for the temporary blockage corridor scenario.

    Units in meters unless noted; grid resolution taken from constants by default.
    """

    map_width_m: float = 10.0
    map_height_m: float = 10.0
    corridor_width_m: float = 2.5
    wall_thickness_m: float = 0.3
    pallet_width_m: float = 0.5
    pallet_length_m: float = 0.6
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.3
    resolution_m: float = GRID_RESOLUTION_M


def create_blockage_scenario(
    cfg: BlockageScenarioConfig | None = None,
) -> Tuple[
    BinaryOccupancyGrid,
    np.ndarray,
    Tuple[float, float, float],
    Tuple[float, float],
    Dict[str, float],
]:
    """
    Build a compact corridor map with a temporary blockage (pallet) and a straight path.

    Returns: (sensing_grid, waypoints, start_pose, goal_xy, info)
    - sensing_grid: BinaryOccupancyGrid with raw occupancy (sensing grid)
    - waypoints: (N,2) straight-line path
    - start_pose: (x, y, theta)
    - goal_xy: (x, y)
    - info: scenario metrics (gaps, widths)
    """
    c = cfg or BlockageScenarioConfig()
    res = float(c.resolution_m)
    grid_w = int(round(c.map_width_m / res))
    grid_h = int(round(c.map_height_m / res))
    grid_array = np.zeros((grid_h, grid_w), dtype=bool)

    # Helpers
    def meters_to_grid(x_m: float, y_m: float) -> tuple[int, int]:
        i = int(y_m / res)  # row
        j = int(x_m / res)  # col
        return min(max(i, 0), grid_h - 1), min(max(j, 0), grid_w - 1)

    def fill_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        i0, j0 = meters_to_grid(x0, y0)
        i1, j1 = meters_to_grid(x1, y1)
        i0, i1 = min(i0, i1), max(i0, i1)
        j0, j1 = min(j0, j1), max(j0, j1)
        grid_array[i0 : i1 + 1, j0 : j1 + 1] = True

    # Corridor walls centered vertically
    cy = c.map_height_m / 2.0
    top_wall_y = cy + c.corridor_width_m / 2.0
    bot_wall_y = cy - c.corridor_width_m / 2.0
    fill_rect(0.0, top_wall_y, c.map_width_m, top_wall_y + c.wall_thickness_m)
    fill_rect(0.0, bot_wall_y - c.wall_thickness_m, c.map_width_m, bot_wall_y)

    # Blocking pallet at center (only if it has positive area)
    pallet_x = c.map_width_m / 2.0
    pallet_y = cy
    if c.pallet_width_m > 0.0 and c.pallet_length_m > 0.0:
        fill_rect(
            pallet_x - c.pallet_length_m / 2.0,
            pallet_y - c.pallet_width_m / 2.0,
            pallet_x + c.pallet_length_m / 2.0,
            pallet_y + c.pallet_width_m / 2.0,
        )

    # Build sensing grid
    grid = BinaryOccupancyGrid(grid_array, cellsize=res, origin=(0, 0))

    # Path from left margin to right margin along corridor centerline
    start_x = c.start_x_m
    start_y = cy
    goal_x = c.map_width_m - c.goal_margin_x_m
    goal_y = cy
    n_wp = max(2, int(round((goal_x - start_x) / max(1e-6, c.waypoint_step_m))))
    x_coords = np.linspace(start_x, goal_x, n_wp)
    y_coords = np.full_like(x_coords, start_y)
    waypoints = np.stack([x_coords, y_coords], axis=1)

    # Gaps
    if c.pallet_width_m > 0.0:
        gap_top = top_wall_y - (pallet_y + c.pallet_width_m / 2.0)
        gap_bottom = (pallet_y - c.pallet_width_m / 2.0) - bot_wall_y
    else:
        # No pallet: gaps are simply to each wall from corridor centerline
        gap_top = top_wall_y - pallet_y
        gap_bottom = pallet_y - bot_wall_y

    info = {
        "gap_top": float(gap_top),
        "gap_bottom": float(gap_bottom),
        "corridor_width": float(c.corridor_width_m),
        "pallet_width": float(c.pallet_width_m),
    }

    start_pose = (float(start_x), float(start_y), 0.0)
    goal_xy = (float(goal_x), float(goal_y))
    return grid, waypoints, start_pose, goal_xy, info
