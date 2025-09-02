from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from .base import ControlCommand


@dataclass
class DWAConfig:
    # Robot kinematic limits
    v_min: float = 0.0
    v_max: float = 1.5
    w_min: float = -2.0
    w_max: float = 2.0
    a_max: float = 1.0  # linear acc limit (m/s^2)
    alpha_max: float = 3.0  # angular acc limit (rad/s^2)

    # Sampling / prediction - aggressive optimization for speed
    num_v_samples: int = 3
    num_w_samples: int = 5
    dt: float = 0.3  # Large timestep for faster prediction
    horizon: float = 0.6  # Short horizon for speed

    # Scoring weights
    weight_heading: float = 1.0
    weight_clearance: float = 1.2
    weight_velocity: float = 0.2
    weight_goal_progress: float = 1.0

    # Safety / geometry
    robot_radius: float = 0.25
    clearance_min: float = 0.3

    # Goal / guidance
    lookahead: float = 1.0
    # Penalty to discourage stalling when safe alternatives exist
    stall_penalty: float = 0.2
    # Lateral gap bias to break deadlocks when front is blocked
    lateral_bias_gain: float = 1.0
    forward_block_thresh: float = 1.2  # meters
    min_drive_when_free: float = 0.3  # m/s


class DynamicWindowApproach:
    def __init__(self, cfg: DWAConfig = DWAConfig()):
        self.cfg = cfg

    def _lookahead_point(
        self, x: float, y: float, waypoints: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if (
            waypoints is None
            or not isinstance(waypoints, np.ndarray)
            or waypoints.shape[0] < 1
        ):
            return None
        d = np.hypot(waypoints[:, 0] - x, waypoints[:, 1] - y)
        idx = int(np.argmin(d))
        j = idx
        while (
            j < waypoints.shape[0] - 1
            and np.hypot(waypoints[j, 0] - x, waypoints[j, 1] - y) < self.cfg.lookahead
        ):
            j += 1
        return waypoints[j]

    def _dynamic_window(
        self, v_curr: float, w_curr: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        cfg = self.cfg
        v_lo = max(cfg.v_min, v_curr - cfg.a_max * cfg.dt)
        v_hi = min(cfg.v_max, v_curr + cfg.a_max * cfg.dt)
        w_lo = max(cfg.w_min, w_curr - cfg.alpha_max * cfg.dt)
        w_hi = min(cfg.w_max, w_curr + cfg.alpha_max * cfg.dt)
        return (v_lo, v_hi), (w_lo, w_hi)

    def _simulate_trajectory(
        self, x: float, y: float, th: float, v: float, w: float
    ) -> np.ndarray:
        cfg = self.cfg
        num_steps = max(1, int(round(cfg.horizon / cfg.dt)))
        traj = np.zeros((num_steps, 3), dtype=float)
        px, py, pth = float(x), float(y), float(th)
        for i in range(num_steps):
            px += v * np.cos(pth) * cfg.dt
            py += v * np.sin(pth) * cfg.dt
            pth += w * cfg.dt
            traj[i] = (px, py, pth)
        return traj

    def _collision_free(self, traj: np.ndarray, grid) -> Tuple[bool, float]:
        """
        Check if a trajectory is collision-free on the provided grid.

        Notes
        -----
        - The grid MUST be a configuration-space (C-space) occupancy grid,
          where obstacles have been inflated by the robot radius and margin.
        - Passing a raw sensing grid here will underestimate collisions.
        """
        # Returns (is_free, min_clearance_est)
        for px, py, _ in traj:
            if grid.isoccupied((float(px), float(py))):
                return False, 0.0
        return True, 1.0  # Simplified clearance estimate

    def _score_trajectory(
        self,
        traj: np.ndarray,
        goal_dir: np.ndarray,
        lidar,
        grid,
        v: float,
    ) -> float:
        cfg = self.cfg
        x_end, y_end, th_end = traj[-1]
        # Heading alignment score (0..1)
        if np.linalg.norm(goal_dir) < 1e-9:
            heading_score = 0.5
        else:
            theta_goal = float(np.arctan2(goal_dir[1], goal_dir[0]))
            dth = (theta_goal - th_end + np.pi) % (2 * np.pi) - np.pi
            heading_score = (1.0 + np.cos(dth)) * 0.5

        # Simplified clearance score - avoid expensive lidar casting
        clearance_score = 1.0  # Assume good clearance if collision-free

        # Velocity score (prefer faster within limits)
        velocity_score = np.clip(
            (v - cfg.v_min) / max(1e-6, (cfg.v_max - cfg.v_min)), 0.0, 1.0
        )

        # Goal progress: projection of end-point displacement on goal_dir
        if np.linalg.norm(goal_dir) < 1e-9:
            progress = 0.0
        else:
            disp = np.array([x_end - traj[0, 0], y_end - traj[0, 1]])
            progress = float(np.dot(disp, goal_dir / (np.linalg.norm(goal_dir) + 1e-9)))
        progress_score = np.clip(0.5 + 0.5 * np.tanh(progress), 0.0, 1.0)

        score = (
            cfg.weight_heading * heading_score
            + cfg.weight_clearance * clearance_score
            + cfg.weight_velocity * velocity_score
            + cfg.weight_goal_progress * progress_score
        )
        # Penalize very low forward speed to avoid stopping in front of obstacles
        if v < 0.1:
            score -= cfg.stall_penalty
        return float(score)

    def action(
        self,
        pose: Tuple[float, float, float],
        waypoints: Optional[np.ndarray],
        lidar,
        grid,
        v_limits: Tuple[float, float],
        w_limits: Tuple[float, float],
        goal_xy: Optional[Tuple[float, float]],
        v_curr: float,
        w_curr: float,
    ) -> ControlCommand:
        """
        Compute DWA control.

        Parameters
        ----------
        pose : (x, y, theta)
        waypoints : optional path for heading/guidance
        lidar : LiDAR sensor (should cast on sensing grid for realism)
        grid : Occupancy grid for collision checks. MUST be the C-space grid.
        v_limits, w_limits : hard command limits
        goal_xy : final goal for guidance if no waypoints
        v_curr, w_curr : current commanded velocities for dynamic window

        Notes
        -----
        - LiDAR casting should use the raw sensing grid to reflect real geometry.
        - Collision checks must use the C-space grid to match environment physics.
        """
        cfg = self.cfg
        x, y, th = pose

        # Goal direction from lookahead waypoint or explicit goal
        p_look = self._lookahead_point(x, y, waypoints)
        if p_look is not None:
            goal_dir = p_look - np.array([x, y])
        elif goal_xy is not None:
            goal_dir = np.array([goal_xy[0] - x, goal_xy[1] - y], dtype=float)
        else:
            goal_dir = np.array([np.cos(th), np.sin(th)], dtype=float)

        # If forward is blocked, bias goal direction laterally toward freer side
        try:
            ranges, _ = lidar.cast((x, y, th), grid)
            angles = lidar.beam_angles(th)
            fwd_mask = np.abs(angles - th) <= np.deg2rad(15.0)
            fwd_min = float(np.min(ranges[fwd_mask])) if np.any(fwd_mask) else np.inf
            left_mask = angles > th
            right_mask = angles < th
            left_min = float(np.min(ranges[left_mask])) if np.any(left_mask) else np.inf
            right_min = (
                float(np.min(ranges[right_mask])) if np.any(right_mask) else np.inf
            )
            if fwd_min < self.cfg.forward_block_thresh:
                steer_left = left_min > right_min
                t_left = np.array([-np.sin(th), np.cos(th)])
                t_right = -t_left
                goal_dir = goal_dir + self.cfg.lateral_bias_gain * (
                    t_left if steer_left else t_right
                )
        except Exception:
            pass

        # Dynamic window bounds intersected with hard limits
        (vd_lo, vd_hi), (wd_lo, wd_hi) = self._dynamic_window(v_curr, w_curr)
        v_lo = max(v_limits[0], vd_lo)
        v_hi = min(v_limits[1], vd_hi)
        w_lo = max(w_limits[0], wd_lo)
        w_hi = min(w_limits[1], wd_hi)

        v_samples = np.linspace(v_lo, v_hi, cfg.num_v_samples)
        w_samples = np.linspace(w_lo, w_hi, cfg.num_w_samples)

        best_score = -np.inf
        best_cmd = (0.0, 0.0)
        start = np.array([x, y, th], dtype=float)

        any_free = False
        best_free_with_drive = (-np.inf, (0.0, 0.0))
        for v in v_samples:
            for w in w_samples:
                traj = self._simulate_trajectory(
                    start[0], start[1], start[2], float(v), float(w)
                )
                free, _ = self._collision_free(traj, grid)
                if not free:
                    continue
                any_free = True
                score = self._score_trajectory(traj, goal_dir, lidar, grid, float(v))
                if score > best_score:
                    best_score = score
                    best_cmd = (float(v), float(w))
                if (
                    v >= self.cfg.min_drive_when_free
                    and score > best_free_with_drive[0]
                ):
                    best_free_with_drive = (score, (float(v), float(w)))

        if not any_free:
            # Rotate toward freer side using lidar
            try:
                ranges, _ = lidar.cast((x, y, th), grid)
                half = len(ranges) // 2
                left_min = float(np.min(ranges[:half])) if half > 0 else 1.0
                right_min = float(np.min(ranges[half:])) if half > 0 else 1.0
                turn_left = left_min > right_min
            except Exception:
                turn_left = True
            w_safe = (w_limits[1] if turn_left else w_limits[0]) * 0.5
            return ControlCommand(v=0.0, w=float(w_safe))

        # Prefer moving solution when available
        if best_free_with_drive[0] > -np.inf:
            v_sel, w_sel = best_free_with_drive[1]
        else:
            v_sel, w_sel = best_cmd
        v_cmd = np.clip(v_sel, v_limits[0], v_limits[1])
        w_cmd = np.clip(w_sel, w_limits[0], w_limits[1])
        return ControlCommand(v=float(v_cmd), w=float(w_cmd))
