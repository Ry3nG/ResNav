"""LiDAR sensor model over occupancy grids using DDA raycasting.

Design decisions:
- Grid convention: grid[i, j] covers [j*res, (j+1)*res) × [i*res, (i+1)*res), True=occupied.
- Beam angles: inclusive endpoints, beam 0 at theta - FOV/2, beam N-1 at theta + FOV/2.
- Traversal: DDA (Amanatides & Woo) with tie-break stepping both axes when needed.
- Noise: add Gaussian noise after computing true distance, then clip to [0, max_range].
"""

from __future__ import annotations

from math import cos, sin, pi, inf
from typing import Optional, Tuple

import numpy as np


class GridLidar:
    """LiDAR raycaster over a boolean occupancy grid using DDA.

    Args:
        beams: number of beams.
        fov_deg: field of view in degrees (inclusive endpoints).
        max_range_m: maximum sensing range in meters.
        noise_std_m: Gaussian noise std in meters.
        noise_enable: enable/disable noise addition.
        resolution_m: grid resolution (meters per cell).
        rng: optional numpy Generator; if None, created internally.
    """

    def __init__(
        self,
        *,
        beams: int = 24,
        fov_deg: float = 240.0,
        max_range_m: float = 4.0,
        noise_std_m: float = 0.03,
        noise_enable: bool = True,
        resolution_m: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        assert beams >= 1
        self.beams = int(beams)
        self.fov_rad = float(np.deg2rad(fov_deg))
        self.max_range = float(max_range_m)
        self.noise_std = float(noise_std_m)
        self.noise_enable = bool(noise_enable)
        self.res = float(resolution_m)
        self._rng = rng or np.random.default_rng()
        # Precompute beam offsets (inclusive endpoints)
        if self.beams == 1:
            self._angle_offsets = np.array([0.0], dtype=np.float64)
        else:
            self._angle_offsets = np.linspace(
                -0.5 * self.fov_rad, 0.5 * self.fov_rad, self.beams, dtype=np.float64
            )

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sense(self, grid: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
        """Cast beams and return distances in meters.

        grid: 2D bool array, True=occupied.
        pose: (x, y, theta) in meters/radians (world frame).
        """
        assert grid.ndim == 2 and grid.dtype == bool
        H, W = grid.shape
        res = self.res
        x, y, theta = pose[0], pose[1], pose[2]

        # If starting outside the map, return max range for all beams
        if not (0.0 <= x < W * res and 0.0 <= y < H * res):
            d = np.full((self.beams,), self.max_range, dtype=np.float64)
            return self._apply_noise_and_clip(d)

        # Starting cell indices (no clipping needed - position already validated)
        j0 = int(np.floor(x / res))
        i0 = int(np.floor(y / res))

        distances = np.empty((self.beams,), dtype=np.float64)
        # Precompute sin/cos per beam
        angles = theta + self._angle_offsets

        for k in range(self.beams):
            phi = angles[k]
            dirx = cos(phi)
            diry = sin(phi)

            # Handle zero direction components
            step_x = 1 if dirx > 0.0 else -1
            step_y = 1 if diry > 0.0 else -1

            # Initialize DDA parameters
            if dirx == 0.0:
                t_max_x = inf
                t_delta_x = inf
            else:
                next_bx = (j0 + (1 if dirx > 0.0 else 0)) * res
                t_max_x = (next_bx - x) / dirx
                t_delta_x = res / abs(dirx)

            if diry == 0.0:
                t_max_y = inf
                t_delta_y = inf
            else:
                next_by = (i0 + (1 if diry > 0.0 else 0)) * res
                t_max_y = (next_by - y) / diry
                t_delta_y = res / abs(diry)

            i, j = i0, j0
            # Immediate collision if starting in obstacle
            if grid[i, j]:
                distances[k] = 0.0
                continue

            t = 0.0
            eps = 1e-9

            # Traverse until hit, exit, or exceed range
            while t < self.max_range:
                # Choose next boundary to cross
                if abs(t_max_x - t_max_y) <= eps:
                    # Diagonal step
                    t = t_max_x
                    j += step_x
                    i += step_y
                    t_max_x += t_delta_x
                    t_max_y += t_delta_y
                elif t_max_x < t_max_y:
                    t = t_max_x
                    j += step_x
                    t_max_x += t_delta_x
                else:
                    t = t_max_y
                    i += step_y
                    t_max_y += t_delta_y

                # Out of bounds -> return distance to boundary
                if j < 0 or j >= W or i < 0 or i >= H:
                    distances[k] = t
                    break

                # Occupied cell -> hit
                if grid[i, j]:
                    distances[k] = t
                    break
            else:
                # Range exceeded
                distances[k] = self.max_range

        return self._apply_noise_and_clip(distances)

    def _apply_noise_and_clip(self, dists: np.ndarray) -> np.ndarray:
        if self.noise_enable and self.noise_std > 0.0:
            noise = self._rng.normal(loc=0.0, scale=self.noise_std, size=dists.shape)
            dists = dists + noise
        # Clip to [0, max_range]
        np.clip(dists, 0.0, self.max_range, out=dists)
        return dists
