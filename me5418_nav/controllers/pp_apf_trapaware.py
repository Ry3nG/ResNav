from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from ..constants import (
    CTRL_LOOKAHEAD_M,
    CTRL_V_NOMINAL_MPS,
    CTRL_K_HEADING,
    CTRL_REPULSE_DIST_M,
    CTRL_REPULSE_GAIN,
    CTRL_ATTRACT_GAIN,
    CTRL_STUCK_WINDOW_S,
    CTRL_MIN_IDX_PROGRESS,
    CTRL_FOLLOW_CLEARANCE_M,
    CTRL_FOLLOW_SPEED_MPS,
    CTRL_TANGENTIAL_GAIN,
    CTRL_NO_POINT_PIVOT_W_RPS,
    CTRL_NO_POINT_PIVOT_V_MPS,
    CTRL_CLEAR_HEADING_GAIN,
    CTRL_TURN_SLOW_SCALE,
    CTRL_CONF_SLOW,
)


@dataclass
class TrapAwareConfig:
    # Pure Pursuit + APF base
    lookahead: float = CTRL_LOOKAHEAD_M
    v_nominal: float = CTRL_V_NOMINAL_MPS
    k_heading: float = CTRL_K_HEADING
    repulse_dist: float = CTRL_REPULSE_DIST_M  # meters
    repulse_gain: float = CTRL_REPULSE_GAIN
    attract_gain: float = CTRL_ATTRACT_GAIN

    # Stuck detection
    stuck_window_s: float = CTRL_STUCK_WINDOW_S
    min_idx_progress: int = CTRL_MIN_IDX_PROGRESS  # min waypoints advanced over window

    # Wall-follow
    follow_clearance: float = CTRL_FOLLOW_CLEARANCE_M
    follow_speed: float = CTRL_FOLLOW_SPEED_MPS
    tangential_gain: float = CTRL_TANGENTIAL_GAIN
    exit_lookahead_points: int = 20  # far waypoint to test line-of-sight


class TrapAwarePPAPF:
    """
    Pure Pursuit + APF with trap-aware behavior:
      - Detects lack of progress along path over a time window
      - Switches to wall-following using LiDAR until line-of-sight to path returns
    """

    def __init__(self, cfg: TrapAwareConfig = TrapAwareConfig(), dt: float = 0.05):
        self.cfg = cfg
        self.dt = dt
        self.mode: str = "track"  # or "wall_follow"
        self._history_idx: list[int] = []
        self._history_len = max(1, int(cfg.stuck_window_s / dt))
        self._follow_side: Optional[str] = None  # 'left' or 'right'
        self.debug: dict = {}
        self._last_theta_des: float | None = None
        self._switch_flag: dict | None = (
            None  # populated for one step when mode switches
        )

    def reset(self):
        self.mode = "track"
        self._history_idx.clear()
        self._follow_side = None
        self._last_theta_des = None
        self._switch_flag = None

    # -------- Helpers --------
    def _nearest_path_index(self, x: float, y: float, waypoints: np.ndarray) -> int:
        d = np.hypot(waypoints[:, 0] - x, waypoints[:, 1] - y)
        return int(np.argmin(d))

    def _lookahead_point(self, x: float, y: float, wp: np.ndarray) -> np.ndarray:
        d = np.hypot(wp[:, 0] - x, wp[:, 1] - y)
        idx = int(np.argmin(d))
        j = idx
        while (
            j < wp.shape[0] - 1
            and np.hypot(wp[j, 0] - x, wp[j, 1] - y) < self.cfg.lookahead
        ):
            j += 1
        return wp[j]

    def _line_of_sight_clear(
        self, grid, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        try:
            idxs = grid.line_w(p1, p2)
            flat = grid.grid.reshape(-1)
            return not bool(flat[idxs].any())
        except Exception:
            # conservative fallback: assume blocked
            return False

    def _stuck_update(self, idx: int) -> bool:
        self._history_idx.append(idx)
        if len(self._history_idx) > self._history_len:
            self._history_idx.pop(0)
        if len(self._history_idx) < self._history_len:
            return False
        progress = self._history_idx[-1] - self._history_idx[0]
        return progress < self.cfg.min_idx_progress

    # -------- Control Laws --------
    def _pp_apf_action(
        self, pose, wp, lidar, grid, v_limits, w_limits
    ) -> Tuple[float, float]:
        cfg = self.cfg
        x, y, th = pose
        p_look = self._lookahead_point(x, y, wp)
        dir_at = p_look - np.array([x, y])
        if np.linalg.norm(dir_at) > 1e-6:
            dir_at = dir_at / np.linalg.norm(dir_at)
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
        F = cfg.attract_gain * dir_at + Frep
        Fnorm = float(np.linalg.norm(F))
        theta_des = th if Fnorm < 1e-6 else float(np.arctan2(F[1], F[0]))
        heading_err = float((theta_des - th + np.pi) % (2 * np.pi) - np.pi)
        min_range = float(np.min(ranges)) if ranges.size else float("inf")
        v = cfg.v_nominal * max(0.2, min(1.0, min_range / max(cfg.repulse_dist, 1e-6)))
        w = cfg.k_heading * heading_err
        v = float(np.clip(v, v_limits[0], v_limits[1]))
        w = float(np.clip(w, w_limits[0], w_limits[1]))
        self.debug = {
            "mode": self.mode,
            "follow_side": self._follow_side,
            "min_range": min_range,
            "heading_err": heading_err,
            "F_norm": Fnorm,
            "F_x": float(F[0]),
            "F_y": float(F[1]),
            "theta_des": float(theta_des),
            "p_look_x": float(p_look[0]),
            "p_look_y": float(p_look[1]),
            "lookahead": cfg.lookahead,
            "v_nominal": cfg.v_nominal,
            "k_heading": cfg.k_heading,
            "repulse_dist": cfg.repulse_dist,
            "repulse_gain": cfg.repulse_gain,
            "attract_gain": cfg.attract_gain,
            "follow_clearance": cfg.follow_clearance,
            "follow_speed": cfg.follow_speed,
            "tangential_gain": cfg.tangential_gain,
            "stuck_window_s": cfg.stuck_window_s,
            "min_idx_progress": cfg.min_idx_progress,
            "exit_lookahead_points": cfg.exit_lookahead_points,
            "v_cmd": v,
            "w_cmd": w,
        }
        if self._switch_flag is not None:
            self.debug.update(self._switch_flag)
            self._switch_flag = None
        self._last_theta_des = float(theta_des)
        return v, w

    def _wall_follow_action(
        self, pose, wp, lidar, grid, v_limits, w_limits
    ) -> Tuple[float, float]:
        cfg = self.cfg
        x, y, th = pose
        ranges, _ = lidar.cast(pose, grid)
        angs_abs = lidar.beam_angles(th)

        # choose side if not set: compare min ranges by relative sign
        if self._follow_side is None:
            angs_rel = self._wrap_pi(angs_abs - th)
            left_min = np.min(ranges[angs_rel > 0]) if np.any(angs_rel > 0) else np.inf
            right_min = np.min(ranges[angs_rel < 0]) if np.any(angs_rel < 0) else np.inf
            self._follow_side = "left" if left_min < right_min else "right"

        wall = self._fit_wall_line(pose, lidar, grid, self._follow_side)
        if wall is None:
            # no points on that side: slow pivot toward chosen side
            w = (
                CTRL_NO_POINT_PIVOT_W_RPS
                if self._follow_side == "left"
                else -CTRL_NO_POINT_PIVOT_W_RPS
            )
            v = CTRL_NO_POINT_PIVOT_V_MPS
            self.debug.update(
                {"mode": self.mode, "reason": "no_wall_points", "v_cmd": v, "w_cmd": w}
            )
            return v, float(np.clip(w, w_limits[0], w_limits[1]))

        C, t, n = wall["C"], wall["t"], wall["n"]

        # orient tangent toward a far waypoint to move forward
        idx = self._nearest_path_index(x, y, wp)
        j = min(idx + cfg.exit_lookahead_points, wp.shape[0] - 1)
        to_far = wp[j] - np.array([x, y])
        if np.linalg.norm(to_far) > 1e-6:
            to_far = to_far / np.linalg.norm(to_far)
            if np.dot(t, to_far) < 0:
                t = -t

        # Signed distance to wall line; positive if outside along +n
        d = float(np.dot(np.array([x, y]) - C, n))
        e = d - cfg.follow_clearance  # >0 too far from wall, <0 too close

        # Heading composition: follow tangent, regulate clearance
        k_clear = CTRL_CLEAR_HEADING_GAIN
        hdg_vec = cfg.tangential_gain * t - k_clear * e * n
        theta_des = (
            th
            if np.linalg.norm(hdg_vec) < 1e-6
            else float(np.arctan2(hdg_vec[1], hdg_vec[0]))
        )
        heading_err = float((theta_des - th + np.pi) % (2 * np.pi) - np.pi)

        # Slow down for sharp turns or poor line fit
        turn_slow = max(0.2, 1.0 - CTRL_TURN_SLOW_SCALE * abs(heading_err) / np.pi)
        conf_slow = CTRL_CONF_SLOW if wall.get("conf", 1.0) < 0.5 else 1.0
        v = min(cfg.follow_speed, v_limits[1]) * turn_slow * conf_slow
        w = float(np.clip(cfg.k_heading * heading_err, w_limits[0], w_limits[1]))

        self.debug.update(
            {
                "mode": self.mode,
                "follow_side": self._follow_side,
                "d_to_wall": d,
                "clear_err": e,
                "heading_err": heading_err,
                "theta_des": float(theta_des),
                "wall_Cx": float(C[0]),
                "wall_Cy": float(C[1]),
                "wall_tx": float(t[0]),
                "wall_ty": float(t[1]),
                "wall_nx": float(n[0]),
                "wall_ny": float(n[1]),
                "wall_conf": float(wall.get("conf", 1.0)),
                "v_cmd": v,
                "w_cmd": w,
            }
        )
        if self._switch_flag is not None:
            self.debug.update(self._switch_flag)
            self._switch_flag = None
        self._last_theta_des = float(theta_des)
        return v, w

    # ---------- New helpers for wall fitting ----------
    def _wrap_pi(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _fit_wall_line(self, pose, lidar, grid, side: str):
        x, y, th = pose
        ranges, _ = lidar.cast(pose, grid)
        angs_abs = lidar.beam_angles(th)
        angs_rel = self._wrap_pi(angs_abs - th)

        lo, hi = np.deg2rad(35), np.deg2rad(145)
        if side == "left":
            mask = (angs_rel > lo) & (angs_rel < hi)
        else:
            mask = (angs_rel < -lo) & (angs_rel > -hi)

        r = ranges[mask]
        a = angs_abs[mask]
        if r.size < 5 or (not np.isfinite(r).any()):
            return None

        px = x + r * np.cos(a)
        py = y + r * np.sin(a)
        P = np.stack([px, py], axis=1)
        C = P.mean(axis=0)
        U, S, Vt = np.linalg.svd(P - C, full_matrices=False)
        t_wall = Vt[0]
        n_wall = Vt[1]
        # Ensure normal points from wall to robot
        if np.dot(np.array([x, y]) - C, n_wall) < 0:
            n_wall = -n_wall
        conf = 0.0 if (S[0] < 1e-3 or (S[1] / (S[0] + 1e-9)) > 0.35) else 1.0
        t_wall = t_wall / (np.linalg.norm(t_wall) + 1e-9)
        n_wall = n_wall / (np.linalg.norm(n_wall) + 1e-9)
        return {"C": C, "t": t_wall, "n": n_wall, "conf": conf}

    # -------- Main policy --------
    def action(
        self, pose, waypoints: np.ndarray, lidar, grid, v_limits, w_limits
    ) -> Tuple[float, float]:
        x, y, th = pose
        idx = self._nearest_path_index(x, y, waypoints)

        # Update stuck detection in tracking mode
        if self.mode == "track":
            stuck = self._stuck_update(idx)
            if stuck:
                prev_theta_des = self._last_theta_des
                self.mode = "wall_follow"
                self._follow_side = None
                self._switch_flag = {
                    "switched_to": "wall_follow",
                    "switch_reason": "stuck",
                    "theta_des_prev": (
                        float(prev_theta_des) if prev_theta_des is not None else None
                    ),
                }

        # Exit wall-follow if line-of-sight to far waypoint is clear
        if self.mode == "wall_follow":
            j = min(idx + self.cfg.exit_lookahead_points, waypoints.shape[0] - 1)
            p_far = (float(waypoints[j, 0]), float(waypoints[j, 1]))
            if self._line_of_sight_clear(grid, (x, y), p_far):
                prev_theta_des = self._last_theta_des
                self.mode = "track"
                self._history_idx.clear()
                self._switch_flag = {
                    "switched_to": "track",
                    "switch_reason": "los_clear",
                    "far_wp_x": p_far[0],
                    "far_wp_y": p_far[1],
                    "theta_des_prev": (
                        float(prev_theta_des) if prev_theta_des is not None else None
                    ),
                }

        if self.mode == "wall_follow":
            return self._wall_follow_action(
                pose, waypoints, lidar, grid, v_limits, w_limits
            )
        else:
            return self._pp_apf_action(pose, waypoints, lidar, grid, v_limits, w_limits)
