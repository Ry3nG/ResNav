"""Scenario generator for the occluded merge & counterflow (OMCF) setup.

This reuses the blockage corridor but carves wall holes near the goal side
so dynamic movers can emerge through the openings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .scenario_generators import OMCFGenerator


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
    gen = OMCFGenerator(c, r)
    return gen.generate()

