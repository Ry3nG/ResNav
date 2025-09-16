"""Scenario sampling utilities for ResidualNavEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from amr_env.sim.scenario_manager import ScenarioManager
from amr_env.sim.collision import inflate_grid


@dataclass
class ScenarioSample:
    grid_raw: np.ndarray
    grid_inflated: np.ndarray
    waypoints: np.ndarray
    start_pose: Tuple[float, float, float]
    goal_xy: Tuple[float, float]
    info: Dict[str, Any]
    edt: np.ndarray | None
    edt_ms: float


class ScenarioService:
    """Aggregate helper that wraps ScenarioManager and EDT precomputation."""

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        robot_radius_m: float,
        resolution_m: float,
    ) -> None:
        self._env_cfg = env_cfg
        self._robot_radius = float(robot_radius_m)
        self._resolution = float(resolution_m)
        self._manager = ScenarioManager(env_cfg)

    def set_seed(self, seed: int) -> None:
        self._manager.set_seed(seed)

    def sample(self) -> ScenarioSample:
        """Sample a feasible scenario and build auxiliary products."""
        max_tries = 20
        grid_raw = None
        waypoints = None
        start_pose = None
        goal_xy = None
        info: Dict[str, Any] = {}

        for _ in range(max_tries):
            grid_raw, waypoints, start_pose, goal_xy, info = self._manager.sample()
            grid_inflated = inflate_grid(
                grid_raw, self._robot_radius, self._resolution
            )
            if not self._point_in_grid(grid_inflated, start_pose[:2]):
                break
        else:
            grid_inflated = inflate_grid(
                grid_raw, self._robot_radius, self._resolution
            )

        edt, edt_ms = self._compute_edt(grid_inflated)
        return ScenarioSample(
            grid_raw=grid_raw,
            grid_inflated=grid_inflated,
            waypoints=waypoints,
            start_pose=start_pose,
            goal_xy=goal_xy,
            info=info,
            edt=edt,
            edt_ms=edt_ms,
        )

    def _point_in_grid(self, grid: np.ndarray, xy: Tuple[float, float]) -> bool:
        x, y = xy
        i = int(np.floor(y / self._resolution))
        j = int(np.floor(x / self._resolution))
        H, W = grid.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return bool(grid[i, j])

    def _compute_edt(self, grid_inflated: np.ndarray) -> Tuple[np.ndarray | None, float]:
        try:
            from .edt_utils import compute_edt_meters

            free_mask = (~grid_inflated).astype(np.uint8)
            edt, ms = compute_edt_meters(free_mask, self._resolution)
            return edt, ms
        except Exception:
            return None, 0.0
