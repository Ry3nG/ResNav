"""Scenario generator for the occluded merge & counterflow (OMCF) setup.

This reuses the blockage corridor but carves wall holes near the goal side
so dynamic movers can emerge through the openings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import numpy as np


@dataclass
class OMCFConfig:
    map_width_m: float = 20.0
    map_height_m: float = 20.0
    corridor_width_min_m: float = 3.5
    corridor_width_max_m: float = 4.0
    wall_thickness_m: float = 0.30
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.30
    resolution_m: float = 0.05
    pallet_width_m: float = 1.1
    pallet_length_m: float = 2.0
    num_pallets_min: int = 1
    num_pallets_max: int = 3
    min_passage_m: float = 1.3
    small_length_range_m: Tuple[float, float] = (1.0, 1.2)
    small_width_range_m: Tuple[float, float] = (1.0, 1.2)
    large_length_range_m: Tuple[float, float] = (1.8, 2.2)
    large_width_range_m: Tuple[float, float] = (1.1, 1.3)
    large_fraction: float = 0.4
    holes_enabled: bool = True
    holes_count_pairs: int = 1
    holes_x_lo_m: float = 14.0
    holes_x_hi_m: float = 17.5
    holes_open_len_m: float = 1.6
    holes_min_spacing_m: float = 1.5
    holes_pair_x_candidates: Tuple[float, ...] = ()


def create_omcf_scenario(
    cfg: Optional[OMCFConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[float, float, float],
    Tuple[float, float],
    Dict[str, object],
]:
    """Generate an occluded merge & counterflow corridor."""

    c = cfg or OMCFConfig()
    r = rng or np.random.default_rng()

    res = float(c.resolution_m)
    H = int(round(c.map_height_m / res))
    W = int(round(c.map_width_m / res))
    grid = np.zeros((H, W), dtype=bool)

    def fill_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        i0 = int(np.clip(math.floor(y0 / res), 0, H - 1))
        i1 = int(np.clip(math.floor(y1 / res), 0, H - 1))
        j0 = int(np.clip(math.floor(x0 / res), 0, W - 1))
        j1 = int(np.clip(math.floor(x1 / res), 0, W - 1))
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        grid[i0 : i1 + 1, j0 : j1 + 1] = True

    corridor_w = float(r.uniform(c.corridor_width_min_m, c.corridor_width_max_m))
    cy = c.map_height_m / 2.0
    y_top = cy + corridor_w / 2.0
    y_bot = cy - corridor_w / 2.0

    holes_x: List[float] = []
    if c.holes_enabled and c.holes_count_pairs > 0:
        if c.holes_pair_x_candidates:
            candidates = list(float(x) for x in c.holes_pair_x_candidates)
            if candidates:
                desired = int(c.holes_count_pairs)
                num_pairs = min(desired, len(candidates))
                choices = r.choice(
                    candidates,
                    size=num_pairs,
                    replace=False,
                )
                holes_x.extend(float(x) for x in np.atleast_1d(choices))
                if num_pairs < desired:
                    for _ in range(desired - num_pairs):
                        candidate = float(r.uniform(c.holes_x_lo_m, c.holes_x_hi_m))
                        holes_x.append(candidate)
        else:
            for _ in range(int(c.holes_count_pairs)):
                candidate = None
                for _ in range(16):
                    x_try = float(r.uniform(c.holes_x_lo_m, c.holes_x_hi_m))
                    if all(abs(x_try - existing) >= c.holes_min_spacing_m for existing in holes_x):
                        candidate = x_try
                        break
                if candidate is None:
                    candidate = float(r.uniform(c.holes_x_lo_m, c.holes_x_hi_m))
                holes_x.append(candidate)
    holes_x = sorted(float(x) for x in holes_x)

    def draw_wall_with_gaps(y0: float, y1: float) -> None:
        if not holes_x:
            fill_rect(0.0, y0, c.map_width_m, y1)
            return
        for idx, x_h in enumerate(holes_x):
            half = c.holes_open_len_m / 2.0
            left_hi = max(0.0, x_h - half)
            right_lo = min(c.map_width_m, x_h + half)
            if idx == 0:
                fill_rect(0.0, y0, left_hi, y1)
            else:
                prev = holes_x[idx - 1] + half
                fill_rect(prev, y0, left_hi, y1)
            if idx == len(holes_x) - 1:
                fill_rect(right_lo, y0, c.map_width_m, y1)

    draw_wall_with_gaps(y_top, y_top + c.wall_thickness_m)
    draw_wall_with_gaps(y_bot - c.wall_thickness_m, y_bot)

    start_x = float(c.start_x_m)
    goal_x = float(c.map_width_m - c.goal_margin_x_m)
    x_lo = start_x + 1.0
    x_hi = goal_x - 1.0

    pallets: List[Tuple[float, float, float, float, str]] = []
    small_cnt = 0
    large_cnt = 0
    n_pallets = int(r.integers(c.num_pallets_min, c.num_pallets_max + 1))
    for _ in range(n_pallets):
        is_large = bool(r.random() < max(0.0, min(1.0, c.large_fraction)))
        if is_large:
            length = float(
                r.uniform(
                    min(c.large_length_range_m),
                    max(c.large_length_range_m),
                )
            )
            width = float(
                r.uniform(
                    min(c.large_width_range_m),
                    max(c.large_width_range_m),
                )
            )
            p_type = "large"
        else:
            length = float(
                r.uniform(
                    min(c.small_length_range_m),
                    max(c.small_length_range_m),
                )
            )
            width = float(
                r.uniform(
                    min(c.small_width_range_m),
                    max(c.small_width_range_m),
                )
            )
            p_type = "small"
        # Clamp to feasible corridor bounds
        max_width = max(0.2, corridor_w - c.min_passage_m - 1e-3)
        width = min(width, max_width)

        toward_top = bool(r.random() < 0.5)
        if toward_top:
            y_lo = y_bot + c.min_passage_m + width / 2.0
            y_hi = y_top - width / 2.0
        else:
            y_lo = y_bot + width / 2.0
            y_hi = y_top - c.min_passage_m - width / 2.0
        if y_hi < y_lo:
            y_center = cy
        else:
            y_center = float(r.uniform(y_lo, y_hi))
        if x_hi <= x_lo:
            x_center = (start_x + goal_x) / 2.0
        else:
            x_center = float(r.uniform(x_lo, x_hi))
        fill_rect(
            x_center - length / 2.0,
            y_center - width / 2.0,
            x_center + length / 2.0,
            y_center + width / 2.0,
        )
        pallets.append((x_center, y_center, length, width, p_type))
        if p_type == "large":
            large_cnt += 1
        else:
            small_cnt += 1

    n_wp = max(2, int(round((goal_x - start_x) / max(1e-6, c.waypoint_step_m))))
    xs = np.linspace(start_x, goal_x, n_wp)
    waypoints = np.stack([xs, np.full_like(xs, cy)], axis=1)

    info: Dict[str, object] = {
        "corridor_width": float(corridor_w),
        "holes_x": [float(x) for x in holes_x],
        "y_top": float(y_top),
        "y_bot": float(y_bot),
        "pallets": pallets,
        "pallet_counts": {"small": int(small_cnt), "large": int(large_cnt)},
    }

    start_pose = (float(start_x), float(cy), 0.0)
    goal_xy = (float(goal_x), float(cy))
    return grid, waypoints, start_pose, goal_xy, info
