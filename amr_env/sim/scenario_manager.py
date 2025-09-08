"""Curriculum-aware scenario manager.

For Phase I, samples blockage-only scenarios using BlockageScenarioConfig.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .scenarios import BlockageScenarioConfig, create_blockage_scenario


class ScenarioManager:
    """Simple scenario manager for Phase I (blockage-only)."""

    def __init__(self, env_cfg: Dict[str, Any]) -> None:
        """Initialize with Hydra-like env config dictionary.

        Expected keys under env_cfg:
        - map.size_m: [W, H]
        - map.resolution_m
        - corridor_width_m: [min, max]
        - wall_thickness_m, pallet_width_m, pallet_length_m (optional)
        - start_x_m, goal_margin_x_m, waypoint_step_m (optional)
        - min_passage_m, num_pallets_min, num_pallets_max (optional)
        """
        self.env_cfg = env_cfg
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _build_blockage_cfg(self) -> BlockageScenarioConfig:
        msize = self.env_cfg.get("map", {}).get("size_m", [50.0, 50.0])
        resolution = float(self.env_cfg.get("map", {}).get("resolution_m", 0.2))
        cw_min, cw_max = self.env_cfg.get("map", {}).get("corridor_width_m", [3.0, 4.0])
        wall_th = float(self.env_cfg.get("map", {}).get("wall_thickness_m", 0.3))

        pallet_w = float(self.env_cfg.get("map", {}).get("pallet_width_m", 1.1))
        pallet_l = float(self.env_cfg.get("map", {}).get("pallet_length_m", 0.6))
        start_x = float(self.env_cfg.get("map", {}).get("start_x_m", 1.0))
        goal_mx = float(self.env_cfg.get("map", {}).get("goal_margin_x_m", 1.0))
        wp_step = float(self.env_cfg.get("map", {}).get("waypoint_step_m", 0.3))
        min_pass = float(self.env_cfg.get("map", {}).get("min_passage_m", 0.7))
        nmin = int(self.env_cfg.get("map", {}).get("num_pallets_min", 1))
        nmax = int(self.env_cfg.get("map", {}).get("num_pallets_max", 1))

        return BlockageScenarioConfig(
            map_width_m=float(msize[0]),
            map_height_m=float(msize[1]),
            corridor_width_min_m=float(cw_min),
            corridor_width_max_m=float(cw_max),
            wall_thickness_m=wall_th,
            pallet_width_m=pallet_w,
            pallet_length_m=pallet_l,
            start_x_m=start_x,
            goal_margin_x_m=goal_mx,
            waypoint_step_m=wp_step,
            resolution_m=resolution,
            min_passage_m=min_pass,
            num_pallets_min=nmin,
            num_pallets_max=nmax,
        )

    def sample(self) -> Tuple:
        """Sample a scenario according to current phase.

        Currently supports only blockage.
        Returns: (grid, waypoints, start_pose, goal_xy, info)
        """
        cfg = self._build_blockage_cfg()
        return create_blockage_scenario(cfg, self._rng)

