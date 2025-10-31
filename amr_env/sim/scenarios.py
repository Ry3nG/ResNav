"""Scenario generators for training and evaluation.

Phase I: temporary blockage corridor with single-side passage guarantee.
Outputs an occupancy grid, straight global path waypoints, start/goal, and info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .scenario_generators import BlockageGenerator


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
    r = rng or np.random.default_rng()
    gen = BlockageGenerator(c, r)
    return gen.generate()

