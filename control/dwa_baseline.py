"""Dynamic Window Approach baseline (lightweight).

Selects (v, w) from a lattice by forward simulating the unicycle for a short
horizon and scoring goal progress, path alignment, heading deviation, and
obstacle proximity. Designed to be self-contained and consistent with the env.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from amr_env.sim.dynamics import UnicycleModel, UnicycleState
from amr_env.gym.path_utils import compute_path_context


def _min_obstacle_distance(
    x: float, y: float, grid: np.ndarray, res: float, search_radius_m: float = 2.0
) -> float:
    """Approximate min distance from (x,y) to any occupied cell within a window.

    Returns distance in meters; if none found within window, returns search_radius_m.
    """
    H, W = grid.shape
    r_cells = int(np.ceil(search_radius_m / res))
    i0 = int(np.clip(np.floor(y / res), 0, H - 1))
    j0 = int(np.clip(np.floor(x / res), 0, W - 1))
    best = search_radius_m
    for di in range(-r_cells, r_cells + 1):
        ii = i0 + di
        if ii < 0 or ii >= H:
            continue
        for dj in range(-r_cells, r_cells + 1):
            jj = j0 + dj
            if jj < 0 or jj >= W:
                continue
            if not grid[ii, jj]:
                continue
            cx = (jj + 0.5) * res
            cy = (ii + 0.5) * res
            d = float(np.hypot(cx - x, cy - y))
            if d < best:
                best = d
    return best


def dwa_select_action(
    pose: Tuple[float, float, float],
    waypoints: np.ndarray,
    grid_inflated: np.ndarray,
    resolution_m: float,
    v_max: float,
    w_max: float,
    dt: float = 0.1,
    horizon_s: float = 2.0,
    v_samples: int = 15,
    w_samples: int = 15,
    w_progress: float = 1.0,
    w_path: float = 0.8,
    w_heading: float = 0.05,
    w_obst: float = 0.6,
    w_smooth: float = 0.02,
    v_min: float = 0.15,
    d_safe: float = 0.6,
    d_free: float = 1.2,
    d_path_deadzone: float = 0.35,
    w_speed: float = 0.2,
) -> Tuple[float, float]:
    """Select (v, w) by simulating a lattice and scoring.

    Scoring function (to maximize):
      score = + a1 * goal_progress + a2 * path_alignment - a3 * heading_dev - a4 * obstacle_cost
    with simple weights.
    """
    x0, y0, th0 = pose
    horizon_steps = int(max(1, round(horizon_s / dt)))
    vs = np.linspace(max(0.0, min(v_min, v_max)), v_max, v_samples)
    ws = np.linspace(-w_max, w_max, w_samples)

    # Precompute path heading near current pose (segment 0 by projection)
    def _path_heading(pts: np.ndarray, p: np.ndarray) -> float:
        # find nearest segment and use its heading
        best = (1e9, 0.0)
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            d = p1 - p0
            L2 = float(np.dot(d, d))
            if L2 == 0.0:
                continue
            t = float(np.dot(p - p0, d) / L2)
            t = max(0.0, min(1.0, t))
            proj = p0 + t * d
            dist2 = float(np.sum((proj - p) ** 2))
            if dist2 < best[0]:
                best = (dist2, float(np.arctan2(d[1], d[0])))
        return best[1]

    path_th = _path_heading(waypoints, np.array([x0, y0], dtype=float))
    gx, gy = waypoints[-1]

    best_score = -1e18
    best_u = (0.0, 0.0)

    for v in vs:
        for w in ws:
            model = UnicycleModel(v_max=v_max, w_max=w_max)
            model.reset(UnicycleState(x0, y0, th0, 0.0, 0.0))
            collided = False
            min_obs = 1e9
            for _ in range(horizon_steps):
                s = model.step((v, w), dt)
                # collision check against inflated grid
                i = int(np.floor(s.y / resolution_m))
                j = int(np.floor(s.x / resolution_m))
                H, W = grid_inflated.shape
                if i < 0 or i >= H or j < 0 or j >= W or grid_inflated[i, j]:
                    collided = True
                    break
                d_obs = _min_obstacle_distance(s.x, s.y, grid_inflated, resolution_m)
                if d_obs < min_obs:
                    min_obs = d_obs
            if collided:
                continue
            # Goal progress
            dx0 = float(np.hypot(gx - x0, gy - y0))
            dx1 = float(np.hypot(gx - model.get_state().x, gy - model.get_state().y))
            progress = dx0 - dx1
            # Path context at terminal state: lateral + heading error
            s_term = model.get_state()
            ctx = compute_path_context(
                (s_term.x, s_term.y, s_term.theta), waypoints, (1.0, 2.0, 3.0)
            )
            path_pen = -(abs(ctx.d_lat) + 0.5 * abs(ctx.theta_err))
            # Heading deviation wrt path heading (weak term now)
            th = s_term.theta
            heading_dev = float(np.abs(((th - path_th) + np.pi) % (2 * np.pi) - np.pi))
            # Obstacle cost: zero beyond d_free, grows as we enter [d_safe, d_free]
            if min_obs >= d_free:
                obst_cost = 0.0
            else:
                if min_obs <= d_safe:
                    obst_cost = 1.0
                else:
                    obst_cost = float((d_free - min_obs) / max(d_free - d_safe, 1e-6))
            # Smoothness cost: prefer smaller |w|
            smooth_cost = abs(w)
            # Path dead-zone: allow lateral offset and small heading error without penalty
            path_lat_term = max(0.0, abs(ctx.d_lat) - d_path_deadzone)
            path_head_term = max(0.0, abs(ctx.theta_err) - 0.2)
            score = (
                w_progress * progress
                + w_speed * float(v)
                - w_path * (path_lat_term + 0.5 * path_head_term)
                + w_heading * np.cos(heading_dev)
                - w_obst * obst_cost
                - w_smooth * smooth_cost
            )
            if score > best_score:
                best_score = score
                best_u = (float(v), float(w))

    # Fallback if everything collided or score remained extremely low
    if best_score <= -1e17:
        return (0.0, 0.5 * w_max)
    return best_u
