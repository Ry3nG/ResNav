from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from ..constants import GRID_RESOLUTION_M, ROBOT_DIAMETER_M


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
    pallet_width_m: float = 1.1
    pallet_length_m: float = 0.6
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.3
    resolution_m: float = GRID_RESOLUTION_M

    # Enhanced randomization parameters
    num_pallets_range: Tuple[int, int] = (1, 1)  # (min, max) pallet count
    pallet_width_range: Tuple[float, float] = (1.1, 1.1)  # (min, max) width in meters
    pallet_length_range: Tuple[float, float] = (0.6, 0.6)  # (min, max) length in meters
    min_passage_width_m: float = (
        0.6  # minimum passage width (robot diameter + safety margin)
    )
    random_seed: int | None = None  # seed for reproducible scenarios

    # Placement parameters
    pallet_start_offset_m: float = 1.0  # offset from start position
    pallet_end_offset_m: float = 1.0  # offset from goal position




def _generate_random_pallets(
    cfg: BlockageScenarioConfig,
) -> tuple[list[Tuple[float, float, float, float]], int]:
    """
    Generate random pallets with dimensions and positions.

    Args:
        cfg: Configuration with randomization parameters

    Returns:
        Tuple of (pallets_list, actual_seed_used)
        - pallets_list: List of (x_center, y_center, width, length) tuples for each pallet
        - actual_seed_used: The actual seed that was used for generation
    """
    # Local RNG for reproducibility without touching global state
    if cfg.random_seed is not None:
        actual_seed = int(cfg.random_seed)
    else:
        import time
        actual_seed = int((time.time() * 1_000_000) % (2**31))
    rng = np.random.default_rng(actual_seed)

    # Generate number of pallets
    min_pallets, max_pallets = cfg.num_pallets_range
    num_pallets = int(rng.integers(min_pallets, max_pallets + 1))

    # If corridor too narrow for any pallet respecting min passage, return none
    max_allowed_width = cfg.corridor_width_m - cfg.min_passage_width_m
    if max_allowed_width <= 0.0 or num_pallets == 0:
        return [], actual_seed

    # Sample pallet dimensions; clamp widths to allowed max
    pallet_widths = []
    pallet_lengths = []
    for _ in range(num_pallets):
        w_low, w_high = cfg.pallet_width_range
        # Ensure upper bound respects corridor constraint
        w_high_eff = max(w_low, min(w_high, max_allowed_width))
        width = float(rng.uniform(w_low, w_high_eff))
        length = float(rng.uniform(*cfg.pallet_length_range))
        pallet_widths.append(width)
        pallet_lengths.append(length)

    # Generate positions based on distribution strategy
    corridor_center_y = cfg.map_height_m / 2.0
    start_x = cfg.start_x_m + cfg.pallet_start_offset_m
    end_x = cfg.map_width_m - cfg.goal_margin_x_m - cfg.pallet_end_offset_m
    available_length = end_x - start_x

    # Randomized y placement with guaranteed passable side gap

    # Deterministic non-overlapping placement along x with optional mild jitter
    pallets: list[Tuple[float, float, float, float]] = []
    if num_pallets == 1:
        x_pos = (start_x + end_x) / 2.0
        # Sample y with constraint: at least min_passage_width on one side
        w0 = pallet_widths[0]
        base_gap = max(0.0, (cfg.corridor_width_m - w0) / 2.0)
        max_offset = base_gap
        if max_offset <= 0.0:
            y_pos = corridor_center_y
        else:
            min_required_offset = max(0.0, cfg.min_passage_width_m - base_gap)
            lo, hi = float(min_required_offset), float(max_offset)
            mag = float(rng.uniform(lo, hi)) if hi > lo else lo
            y_pos = corridor_center_y + (mag if rng.random() < 0.5 else -mag)
        pallets.append((x_pos, y_pos, w0, pallet_lengths[0]))
        return pallets, actual_seed

    spacing = available_length / (num_pallets + 1)
    # Jitter limited to keep intervals non-overlapping
    length_margin = max(pallet_lengths)
    safe_half = max(0.0, spacing / 2.0 - length_margin / 2.0 - 0.05)

    for i in range(num_pallets):
        base_x = start_x + (i + 1) * spacing
        jitter = float(rng.uniform(-safe_half, safe_half)) if safe_half > 0 else 0.0
        x_pos = max(start_x, min(end_x, base_x + jitter))
        # Sample y with constraint: at least min_passage_width on one side
        wi = pallet_widths[i]
        base_gap = max(0.0, (cfg.corridor_width_m - wi) / 2.0)
        max_offset = base_gap
        if max_offset <= 0.0:
            y_pos = corridor_center_y
        else:
            min_required_offset = max(0.0, cfg.min_passage_width_m - base_gap)
            lo, hi = float(min_required_offset), float(max_offset)
            mag = float(rng.uniform(lo, hi)) if hi > lo else lo
            y_pos = corridor_center_y + (mag if rng.random() < 0.5 else -mag)
        pallets.append((x_pos, y_pos, wi, pallet_lengths[i]))

    return pallets, actual_seed


def _calculate_scenario_difficulty(
    cfg: BlockageScenarioConfig,
    pallets: list[Tuple[float, float, float, float]],
    top_wall_y: float,
    bot_wall_y: float,
) -> Dict[str, float]:
    """Compute simple difficulty metrics.

    For each pallet, compute top and bottom side gaps. The scenario bottleneck
    is approximated as the minimum across pallets of the better side gap.
    """
    corridor_center_y = cfg.map_height_m / 2.0
    corridor_width = cfg.corridor_width_m

    if not pallets:
        gap_top_min = corridor_width / 2.0
        gap_bottom_min = corridor_width / 2.0
        min_passage_width = corridor_width
        total_pallet_width = 0.0
        widest = 0.0
    else:
        widths = [w for (_, _, w, _) in pallets]
        total_pallet_width = float(sum(widths))
        widest = float(max(widths))
        gap_top_min = float("inf")
        gap_bottom_min = float("inf")
        min_passage_width = float("inf")
        for (_, y, w, _) in pallets:
            top_gap = max(0.0, top_wall_y - (y + w / 2.0))
            bot_gap = max(0.0, (y - w / 2.0) - bot_wall_y)
            gap_top_min = min(gap_top_min, top_gap)
            gap_bottom_min = min(gap_bottom_min, bot_gap)
            min_passage_width = min(min_passage_width, max(top_gap, bot_gap))
        if not np.isfinite(min_passage_width):
            min_passage_width = 0.0
        if not np.isfinite(gap_top_min):
            gap_top_min = 0.0
        if not np.isfinite(gap_bottom_min):
            gap_bottom_min = 0.0

    clearance_top = max(0.0, gap_top_min - ROBOT_DIAMETER_M)
    clearance_bottom = max(0.0, gap_bottom_min - ROBOT_DIAMETER_M)
    min_clearance = max(0.0, min_passage_width - ROBOT_DIAMETER_M)
    difficulty_score = (min_passage_width / corridor_width) if corridor_width > 0 else 0.0

    return {
        "gap_top": float(gap_top_min),
        "gap_bottom": float(gap_bottom_min),
        "min_passage_width": float(min_passage_width),
        "total_pallet_width": float(total_pallet_width),
        "widest_pallet": float(widest),
        "clearance_top": float(clearance_top),
        "clearance_bottom": float(clearance_bottom),
        "min_clearance": float(min_clearance),
        "difficulty_score": float(difficulty_score),
        "robot_diameter": float(ROBOT_DIAMETER_M),
    }


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
    Build a compact corridor map with randomized temporary blockages (pallets) and a straight path.

    Returns: (sensing_grid, waypoints, start_pose, goal_xy, info)
    - sensing_grid: BinaryOccupancyGrid with raw occupancy (sensing grid)
    - waypoints: (N,2) straight-line path
    - start_pose: (x, y, theta)
    - goal_xy: (x, y)
    - info: scenario metrics (gaps, widths, pallet info)
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

    # Generate and place multiple random pallets
    pallets, actual_seed = _generate_random_pallets(c)
    pallet_info = []

    for pallet_x, pallet_y, pallet_width, pallet_length in pallets:
        if pallet_width > 0.0 and pallet_length > 0.0:
            fill_rect(
                pallet_x - pallet_length / 2.0,
                pallet_y - pallet_width / 2.0,
                pallet_x + pallet_length / 2.0,
                pallet_y + pallet_width / 2.0,
            )
            pallet_info.append(
                {
                    "x": float(pallet_x),
                    "y": float(pallet_y),
                    "width": float(pallet_width),
                    "length": float(pallet_length),
                }
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

    # Calculate gaps and scenario difficulty
    difficulty_info = _calculate_scenario_difficulty(c, pallets, top_wall_y, bot_wall_y)

    info = {
        "corridor_width": float(c.corridor_width_m),
        "num_pallets": len(pallets),
        "pallets": pallet_info,
        "actual_seed": int(actual_seed),
        **difficulty_info,
    }

    start_pose = (float(start_x), float(start_y), 0.0)
    goal_xy = (float(goal_x), float(goal_y))
    return grid, waypoints, start_pose, goal_xy, info
