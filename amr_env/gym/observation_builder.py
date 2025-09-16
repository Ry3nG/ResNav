"""Observation assembly helpers for ResidualNavEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from amr_env.sim.lidar import GridLidar
from amr_env.planning.path import compute_path_context, PathContext


@dataclass
class ObservationData:
    obs: Dict[str, np.ndarray]
    context: PathContext


class ObservationBuilder:
    """Builds structured observations and caches reusable context."""

    def __init__(
        self,
        lidar: GridLidar,
        preview_offsets: Tuple[float, float, float] = (1.0, 2.0, 3.0),
    ) -> None:
        self._lidar = lidar
        self._preview_offsets = preview_offsets
        self._last_context: PathContext | None = None

    def reset(self) -> None:
        self._last_context = None

    @property
    def last_context(self) -> PathContext | None:
        return self._last_context

    def build(
        self,
        grid: np.ndarray,
        pose: Tuple[float, float, float],
        waypoints: np.ndarray,
        last_u: Tuple[float, float],
        prev_u: Tuple[float, float],
    ) -> ObservationData:
        lidar = self._lidar.sense(grid, pose).astype(np.float32)
        context = compute_path_context(pose, waypoints, self._preview_offsets)
        self._last_context = context

        kin = np.array(
            [last_u[0], last_u[1], prev_u[0], prev_u[1]], dtype=np.float32
        )
        path = np.array(
            [
                context.d_lat,
                context.theta_err,
                *context.previews_robot.flatten().tolist(),
            ],
            dtype=np.float32,
        )
        obs = {"lidar": lidar, "kin": kin, "path": path}
        return ObservationData(obs=obs, context=context)
