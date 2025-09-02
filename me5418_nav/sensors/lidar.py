from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
from ..constants import LIDAR_BEAMS, LIDAR_FOV_DEG, LIDAR_RANGE_M, LIDAR_STEP_M


@dataclass
class Lidar:
    n_beams: int = LIDAR_BEAMS
    fov: float = np.deg2rad(LIDAR_FOV_DEG)
    max_range: float = LIDAR_RANGE_M
    step: float = LIDAR_STEP_M

    def beam_angles(self, heading: float) -> np.ndarray:
        half = self.fov / 2
        rel = (
            np.array([0.0])
            if self.n_beams == 1
            else np.linspace(-half, half, self.n_beams)
        )
        return heading + rel

    def cast(
        self, pose: Tuple[float, float, float], grid: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y, theta = pose
        angles = self.beam_angles(theta)
        ranges = np.full((self.n_beams,), self.max_range, dtype=float)
        endpoints = np.zeros((self.n_beams, 2), dtype=float)
        use_rtb = hasattr(grid, "isoccupied") and callable(getattr(grid, "isoccupied"))
        for i, ang in enumerate(angles):
            r = 0.0
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            while r < self.max_range:
                r += self.step
                px, py = x + r * cos_a, y + r * sin_a
                if use_rtb:
                    if grid.isoccupied((px, py)):
                        break
                else:
                    gy, gx = grid.world_to_grid(px, py)
                    if grid.is_occupied_index(gy, gx):
                        break
            ranges[i] = min(r, self.max_range)
            endpoints[i] = (x + ranges[i] * cos_a, y + ranges[i] * sin_a)
        return ranges, endpoints
