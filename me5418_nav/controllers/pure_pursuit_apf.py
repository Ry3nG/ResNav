from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from .base import ControlCommand


@dataclass
class PPAPFConfig:
    lookahead: float = 0.75
    v_nominal: float = 0.5
    k_heading: float = 1.5
    repulse_dist: float = 0.5  # meters
    repulse_gain: float = 0.5
    attract_gain: float = 1.2
    tangential_gain: float = 0.5


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
        waypoints: Optional[np.ndarray],
        lidar,
        grids,
        v_limits: Tuple[float, float],
        w_limits: Tuple[float, float],
        goal_xy: Optional[Tuple[float, float]] = None,
        v_curr: float = 0.0,
        w_curr: float = 0.0,
    ) -> ControlCommand:
        """
        Compute control using Pure Pursuit + artificial potential fields.

        Note: Uses sensing grid for LiDAR. Collision handled by env termination.
        """
        cfg = self.cfg
        x, y, th = pose
        # Attractive direction toward lookahead
        wp = waypoints if waypoints is not None else np.array([[x, y]])
        p_look = self._lookahead_point(x, y, wp)
        dir_at = p_look - np.array([x, y])
        if np.linalg.norm(dir_at) > 1e-6:
            dir_at = dir_at / np.linalg.norm(dir_at)
        # Repulsive field from LiDAR: sum of vectors away from obstacles within d0
        ranges, _ = lidar.cast(pose, grids.sensing)
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
        # Simple gap preference: steer tangentially toward the freer side (left/right)
        left_mask = angles > 0.0
        right_mask = angles < 0.0
        # Use minimum distance as conservative side clearance estimate
        left_min = np.min(ranges[left_mask]) if np.any(left_mask) else np.inf
        right_min = np.min(ranges[right_mask]) if np.any(right_mask) else np.inf
        go_left = left_min > right_min
        # Unit vectors tangential to current heading (left/right)
        t_left = np.array([-np.sin(th), np.cos(th)])
        t_right = -t_left
        Ftan = cfg.tangential_gain * (t_left if go_left else t_right)
        # Resultant desired heading
        F = cfg.attract_gain * dir_at + Frep + Ftan
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
        return ControlCommand(v=float(v), w=float(w))


# Note: lidar.cast requires a grid; in this controller we always call action() with grid passed in.
