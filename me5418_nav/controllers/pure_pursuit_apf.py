from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from ..constants import (
    CTRL_LOOKAHEAD_M,
    CTRL_V_NOMINAL_MPS,
    CTRL_K_HEADING,
    CTRL_REPULSE_DIST_M,
    CTRL_REPULSE_GAIN,
    CTRL_ATTRACT_GAIN,
)


@dataclass
class PPAPFConfig:
    lookahead: float = CTRL_LOOKAHEAD_M
    v_nominal: float = CTRL_V_NOMINAL_MPS
    k_heading: float = CTRL_K_HEADING
    repulse_dist: float = CTRL_REPULSE_DIST_M  # meters
    repulse_gain: float = CTRL_REPULSE_GAIN
    attract_gain: float = CTRL_ATTRACT_GAIN


class PurePursuitAPF:
    def __init__(self, cfg: PPAPFConfig = PPAPFConfig()):
        self.cfg = cfg

    def _lookahead_point(self, x: float, y: float, waypoints: np.ndarray) -> np.ndarray:
        # choose the waypoint ahead at a distance ~ lookahead
        d = np.hypot(waypoints[:, 0] - x, waypoints[:, 1] - y)
        idx = int(np.argmin(d))
        # walk forward until distance exceeds lookahead or end
        j = idx
        while (
            j < waypoints.shape[0] - 1
            and np.hypot(waypoints[j, 0] - x, waypoints[j, 1] - y) < self.cfg.lookahead
        ):
            j += 1
        return waypoints[j]

    def action(
        self,
        pose: Tuple[float, float, float],
        waypoints: np.ndarray,
        lidar,
        grid,
        v_limits: Tuple[float, float],
        w_limits: Tuple[float, float],
    ) -> Tuple[float, float]:
        cfg = self.cfg
        x, y, th = pose
        # Attractive direction toward lookahead
        p_look = self._lookahead_point(x, y, waypoints)
        dir_at = p_look - np.array([x, y])
        if np.linalg.norm(dir_at) > 1e-6:
            dir_at = dir_at / np.linalg.norm(dir_at)
        # Repulsive field from LiDAR: sum of vectors away from obstacles within d0
        ranges, _ = lidar.cast(pose, grid)
        angles = lidar.beam_angles(th)
        Frep = np.zeros(2)
        for r, a in zip(ranges, angles):
            if r < cfg.repulse_dist:
                u_obs = np.array([np.cos(a), np.sin(a)])
                mag = (
                    cfg.repulse_gain
                    * (1.0 / r - 1.0 / cfg.repulse_dist)
                    / max(r * r, 1e-6)
                )
                Frep += -mag * u_obs
        # Resultant desired heading
        F = cfg.attract_gain * dir_at + Frep
        if np.linalg.norm(F) < 1e-6:
            theta_des = th
        else:
            theta_des = float(np.arctan2(F[1], F[0]))
        heading_err = float((theta_des - th + np.pi) % (2 * np.pi) - np.pi)

        # Speeds
        v = cfg.v_nominal * max(0.2, min(1.0, np.min(ranges) / cfg.repulse_dist))
        w = cfg.k_heading * heading_err

        # clamp
        v = float(np.clip(v, v_limits[0], v_limits[1]))
        w = float(np.clip(w, w_limits[0], w_limits[1]))
        return v, w


# Note: lidar.cast requires a grid; in this controller we always call action() with grid passed in.
