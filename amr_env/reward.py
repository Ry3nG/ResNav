"""Modular reward computation utilities.

Provides a single source of truth for reward math and a stable
schema to expose breakdowns for logging and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from amr_env.control.pure_pursuit import compute_u_track
from amr_env.planning.path import compute_path_context


@dataclass
class RewardTerms:
    progress: float
    path: float
    effort: float
    sparse: float


def compute_terms(
    pose: tuple[float, float, float],
    waypoints: np.ndarray,
    prev_goal_dist: float | None,
    last_u: tuple[float, float],
    prev_u: tuple[float, float],
    robot_cfg: dict,
    reward_cfg: dict,
    terminated: bool,
    truncated: bool = False,
) -> tuple[RewardTerms, float]:
    """Compute raw reward terms and updated prev_goal_dist.

    Returns (terms, new_prev_goal_dist).
    """
    x, y, th = pose

    # Sparse (decide sign based on computed goal distance)
    sparse = 0.0

    # Progress: per-step time cost (-1) plus potential shaping with Φ = -kappa * d_goal
    shp = reward_cfg.get("shaping", {})
    gamma = float(shp.get("gamma", 0.99))
    kappa = float(shp.get("kappa", 6.5))
    neutralize_first = bool(shp.get("neutralize_first_step", False))

    gx, gy = waypoints[-1]
    goal_dist_t = float(np.hypot(gx - x, gy - y))  # d_{t+1}
    d_prev = goal_dist_t if prev_goal_dist is None else float(prev_goal_dist)
    progress = -1.0 + kappa * (d_prev - gamma * goal_dist_t)
    if prev_goal_dist is None and neutralize_first:
        progress = -1.0

    # Path term: legacy path deviation penalty or deterministic safety barrier.
    safety = reward_cfg.get("safety", {}) if isinstance(reward_cfg.get("safety"), dict) else None
    safety_source = str(safety.get("source", "")) if safety else ""
    use_map_barrier = safety is not None and safety_source.lower() == "map"

    if use_map_barrier:
        d_safe = float(safety.get("d_safe_m", 0.40))
        lam_bar = float(safety.get("lambda_barrier", 5.0))
        dmin = reward_cfg.get("_dmin")
        dmin = float(dmin) if dmin is not None else float("inf")
        xi_d = max((d_safe - dmin) / max(d_safe, 1e-6), 0.0) ** 2
        path_pen = -lam_bar * xi_d

        if bool(safety.get("ttc_enabled", False)):
            tau_safe = float(safety.get("tau_safe_s", 1.5))
            lam_ttc = float(safety.get("lambda_ttc", 3.0))
            true_ranges = reward_cfg.get("_true_ranges")
            prev_true_ranges = reward_cfg.get("_prev_true_ranges")
            dt = float(reward_cfg.get("_dt", 0.1))
            if true_ranges is not None and prev_true_ranges is not None:
                dr = (true_ranges - prev_true_ranges) / max(dt, 1e-6)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ttc = np.where(dr < 0.0, true_ranges / (-dr + 1e-3), np.inf)
                ttc_min = float(np.min(ttc))
                xi_ttc = max(
                    (1.0 / max(ttc_min, 1e-6) - 1.0 / max(tau_safe, 1e-6)),
                    0.0,
                ) ** 2
                path_pen += -lam_ttc * xi_ttc
    else:
        ctx: Any = reward_cfg.get("_ctx")
        if ctx is None:
            ctx = compute_path_context((x, y, th), waypoints, (1.0, 2.0, 3.0))
        path_cfg = reward_cfg.get("path_penalty", {})
        lat_w = float(path_cfg.get("lateral_weight", 0.0))
        head_w = float(path_cfg.get("heading_weight", 0.0))
        path_pen = -(lat_w * abs(getattr(ctx, "d_lat", 0.0)) + head_w * abs(getattr(ctx, "theta_err", 0.0)))

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
    terms: RewardTerms, weights: dict[str, float]
) -> tuple[float, dict[str, float]]:
    """Apply weights to raw terms to produce total and contributions.

    Returns (total, contrib_dict)
    """
    # Build contributions generically from provided weights and available raw terms
    raw = asdict(terms)
    contrib: dict[str, float] = {}
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
    weights: dict[str, float],
    total: float,
    contrib: dict[str, float],
) -> dict[str, object]:
    """Pack a standardized reward_terms dict for logging/visualization."""
    return {
        "version": "1.0",
        "raw": asdict(terms),
        "weights": {k: float(v) for k, v in weights.items()},
        "contrib": contrib,
        "total": float(total),
    }
