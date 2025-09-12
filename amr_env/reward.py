"""Modular reward computation utilities.

Provides a single source of truth for reward math and a stable
schema to expose breakdowns for logging and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any

import numpy as np

from control.pure_pursuit import compute_u_track
from amr_env.gym.path_utils import compute_path_context


@dataclass
class RewardTerms:
    progress: float
    path: float
    effort: float
    sparse: float


def compute_terms(
    pose: Tuple[float, float, float],
    waypoints: np.ndarray,
    prev_goal_dist: float | None,
    last_u: Tuple[float, float],
    prev_u: Tuple[float, float],
    robot_cfg: Dict,
    reward_cfg: Dict,
    terminated: bool,
    truncated: bool = False,
) -> Tuple[RewardTerms, float]:
    """Compute raw reward terms and updated prev_goal_dist.

    Returns (terms, new_prev_goal_dist).
    """
    x, y, th = pose

    # Sparse (decide sign based on computed goal distance)
    sparse = 0.0

    # Progress
    gx, gy = waypoints[-1]
    goal_dist_t = float(np.hypot(gx - x, gy - y))
    d_prev = goal_dist_t if prev_goal_dist is None else prev_goal_dist
    progress = d_prev - goal_dist_t

    # Path penalty via path context
    # Allow caller to pass a cached context via reward_cfg["_ctx"] to avoid recompute
    ctx: Any = reward_cfg.get("_ctx")
    if ctx is None:
        ctx = compute_path_context((x, y, th), waypoints, (1.0, 2.0, 3.0))
    lat_w = float(reward_cfg["path_penalty"]["lateral_weight"])
    head_w = float(reward_cfg["path_penalty"]["heading_weight"])
    path_pen = -(lat_w * abs(ctx.d_lat) + head_w * abs(ctx.theta_err))

    # Effort on residual (relative to tracker)
    look = float(robot_cfg["controller"]["lookahead_m"])
    v_nom = float(robot_cfg["controller"]["speed_nominal"])
    v_track, w_track = compute_u_track((x, y, th), waypoints, look, v_nom)
    dv = float(last_u[0] - v_track)
    dw = float(last_u[1] - w_track)
    lam = reward_cfg["effort_penalty"]
    effort = -(
        float(lam["lambda_v"]) * abs(dv)
        + float(lam["lambda_w"]) * abs(dw)
        + float(lam["lambda_jerk"])
        * (abs(last_u[0] - prev_u[0]) + abs(last_u[1] - prev_u[1]))
    )

    # Fill sparse after computing goal distance if episode ended
    if terminated or truncated:
        s_cfg = reward_cfg["sparse"]
        goal_bonus = float(s_cfg["goal"])
        coll_pen = float(s_cfg["collision"])
        timeout_pen = float(s_cfg["timeout"])

        if terminated and goal_dist_t < 0.5:
            sparse = goal_bonus  # Goal reached
        elif terminated:
            sparse = coll_pen  # Collision (terminated but not at goal)
        elif truncated:
            sparse = timeout_pen  # Timeout (truncated)
        else:
            sparse = 0.0  # Shouldn't reach here

    terms = RewardTerms(
        progress=float(progress),
        path=float(path_pen),
        effort=float(effort),
        sparse=float(sparse),
    )
    return terms, goal_dist_t


def apply_weights(
    terms: RewardTerms, weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Apply weights to raw terms to produce total and contributions.

    Returns (total, contrib_dict)
    """
    # Build contributions generically from provided weights and available raw terms
    raw = asdict(terms)
    contrib: Dict[str, float] = {}
    for k, w in weights.items():
        try:
            contrib[k] = float(w) * float(raw.get(k, 0.0))
        except Exception:
            contrib[k] = 0.0
    total = float(sum(contrib.values()))
    # Sanitize in case of NaN/inf
    total = float(np.nan_to_num(total))
    contrib = {k: float(np.nan_to_num(v)) for k, v in contrib.items()}
    return total, contrib


def to_breakdown_dict(
    terms: RewardTerms,
    weights: Dict[str, float],
    total: float,
    contrib: Dict[str, float],
) -> Dict[str, object]:
    """Pack a standardized reward_terms dict for logging/visualization."""
    return {
        "version": "1.0",
        "raw": asdict(terms),
        "weights": {k: float(v) for k, v in weights.items()},
        "contrib": contrib,
        "total": float(total),
    }
