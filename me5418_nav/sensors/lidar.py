from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class Lidar:
    def __init__(self, beams: int, fov_deg: float, max_range_m: float, step_m: float):
        self.beams = int(beams)
        self.fov = float(math.radians(fov_deg))
        self.max_range = float(max_range_m)
        self.step = float(step_m)
        self.rel_angles = np.linspace(-self.fov / 2.0, self.fov / 2.0, self.beams)

    def sense(self, grid: np.ndarray, pose: Tuple[float, float, float], res: float, map_w_m: float, map_h_m: float) -> Tuple[np.ndarray, float]:
        x, y, th = pose
        out = np.empty(self.beams, dtype=float)
        H, W = grid.shape
        for i, a_rel in enumerate(self.rel_angles):
            a = th + float(a_rel)
            ca = math.cos(a)
            sa = math.sin(a)
            d = 0.0
            hit = False
            while d < self.max_range:
                xp = x + ca * d
                yp = y + sa * d
                if xp < 0.0 or yp < 0.0 or xp >= map_w_m or yp >= map_h_m:
                    # Reached map boundary: report distance to boundary instead of max range
                    out[i] = min(d, self.max_range)
                    break
                ii = int(np.clip(yp / res, 0, H - 1))
                jj = int(np.clip(xp / res, 0, W - 1))
                if grid[ii, jj]:
                    hit = True
                    break
                d += self.step
            if hit:
                out[i] = d
            else:
                # If loop ended due to range limit without hits/boundary, set to max range
                if d >= self.max_range and (xp >= 0.0 and yp >= 0.0 and xp < map_w_m and yp < map_h_m):
                    out[i] = self.max_range
                # Else value already written when boundary was reached
        return out, float(np.min(out))
