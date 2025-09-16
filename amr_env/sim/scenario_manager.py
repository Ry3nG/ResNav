"""Curriculum-aware scenario manager.

For Phase I, samples blockage-only scenarios using BlockageScenarioConfig.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple
import numpy as np
from collections import deque

from .scenarios import BlockageScenarioConfig, create_blockage_scenario


class ScenarioManager:
    """Simple scenario manager for Phase I (blockage-only)."""

    def __init__(self, env_cfg: Dict[str, Any], robot_radius_m: float | None = None) -> None:
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
        robot_cfg = env_cfg.get("robot", {}) if isinstance(env_cfg, dict) else {}
        if robot_radius_m is not None:
            self._robot_radius_m = float(robot_radius_m)
        else:
            self._robot_radius_m = float(robot_cfg.get("radius_m", 0.25))

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _build_blockage_cfg(self) -> BlockageScenarioConfig:
        msize = self.env_cfg["map"]["size_m"]
        resolution = float(self.env_cfg["map"]["resolution_m"])
        cw_min, cw_max = self.env_cfg["map"]["corridor_width_m"]
        wall_th = float(self.env_cfg["map"]["wall_thickness_m"])

        pallet_w = float(self.env_cfg["map"]["pallet_width_m"])
        pallet_l = float(self.env_cfg["map"]["pallet_length_m"])
        start_x = float(self.env_cfg["map"]["start_x_m"])
        goal_mx = float(self.env_cfg["map"]["goal_margin_x_m"])
        wp_step = float(self.env_cfg["map"]["waypoint_step_m"])
        min_pass = float(self.env_cfg["map"]["min_passage_m"])
        nmin = int(self.env_cfg["map"]["num_pallets_min"])
        nmax = int(self.env_cfg["map"]["num_pallets_max"])

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

        # Try to generate a feasible scenario with retries
        max_retries = 5
        for attempt in range(max_retries):
            grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, self._rng)

            # Check path feasibility
            if self._is_path_feasible(grid, start_pose, goal_xy, cfg.resolution_m):
                return grid, waypoints, start_pose, goal_xy, info

            # If not feasible, try again with different random state
            self._rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # If all retries failed, return the last attempt (fallback)
        return grid, waypoints, start_pose, goal_xy, info

    def _is_path_feasible(self, grid: np.ndarray, start_pose: Tuple[float, float, float],
                         goal_xy: Tuple[float, float], resolution: float) -> bool:
        """Path feasibility check using BFS with robot radius inflation.

        Args:
            grid: Occupancy grid (True=occupied)
            start_pose: (x, y, theta) in meters and radians
            goal_xy: (x, y) in meters
            resolution: Grid resolution in meters per cell

        Returns:
            True if path exists, False otherwise
        """
        from .collision import inflate_grid

        # Create inflated grid for robot radius
        grid_inflated = inflate_grid(grid, self._robot_radius_m, resolution)

        start_x, start_y, _ = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / resolution))
        start_j = int(np.floor(start_x / resolution))
        goal_i = int(np.floor(goal_y / resolution))
        goal_j = int(np.floor(goal_x / resolution))

        H, W = grid_inflated.shape

        # Check bounds
        if (start_i < 0 or start_i >= H or start_j < 0 or start_j >= W or
            goal_i < 0 or goal_i >= H or goal_j < 0 or goal_j >= W):
            return False

        # Check if start/goal are in obstacles
        if grid_inflated[start_i, start_j] or grid_inflated[goal_i, goal_j]:
            return False

        # BFS to find path
        visited = np.zeros_like(grid_inflated, dtype=bool)
        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while queue:
            i, j = queue.popleft()

            if i == goal_i and j == goal_j:
                return True

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (0 <= ni < H and 0 <= nj < W and
                    not visited[ni, nj] and not grid_inflated[ni, nj]):
                    visited[ni, nj] = True
                    queue.append((ni, nj))

        return False
