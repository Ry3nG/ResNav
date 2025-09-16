"""Reward bookkeeping for ResidualNavEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from amr_env.reward import (
    RewardTerms,
    apply_weights,
    compute_terms,
    to_breakdown_dict,
)


@dataclass
class RewardResult:
    total: float
    breakdown: Dict[str, Any]
    terms: RewardTerms


class RewardManager:
    """Encapsulates reward configuration and per-episode state."""

    def __init__(
        self,
        robot_cfg: Dict[str, Any],
        reward_cfg: Dict[str, Any],
        dt: float,
    ) -> None:
        self._robot_cfg = robot_cfg
        self._reward_cfg = reward_cfg
        self._dt = float(dt)
        self._prev_goal_dist: float | None = None
        self._prev_true_ranges: np.ndarray | None = None
        self._last_breakdown: Dict[str, Any] = {}

    def reset(self) -> None:
        self._prev_goal_dist = None
        self._prev_true_ranges = None
        self._last_breakdown = {}

    @property
    def last_breakdown(self) -> Dict[str, Any]:
        return self._last_breakdown

    def compute(
        self,
        pose: Tuple[float, float, float],
        waypoints: np.ndarray,
        last_u: Tuple[float, float],
        prev_u: Tuple[float, float],
        terminated: bool,
        truncated: bool,
        path_context: Any | None,
        clearance_m: float | None,
        true_ranges: np.ndarray | None,
    ) -> RewardResult:
        reward_cfg_with_ctx = dict(self._reward_cfg)
        reward_cfg_with_ctx["_ctx"] = path_context
        reward_cfg_with_ctx["_dt"] = self._dt

        safety_cfg = self._reward_cfg.get("safety") or {}
        if not isinstance(safety_cfg, dict):
            safety_cfg = {}
        use_map_barrier = str(safety_cfg.get("source", "")).lower() == "map"

        if use_map_barrier:
            reward_cfg_with_ctx["_dmin"] = float(clearance_m) if clearance_m is not None else None
            ttc_enabled = bool(safety_cfg.get("ttc_enabled", False))
            if ttc_enabled and true_ranges is not None:
                reward_cfg_with_ctx["_true_ranges"] = true_ranges.astype(np.float32)
                reward_cfg_with_ctx["_prev_true_ranges"] = (
                    None
                    if self._prev_true_ranges is None
                    else self._prev_true_ranges.astype(np.float32)
                )
            else:
                reward_cfg_with_ctx["_true_ranges"] = None
                reward_cfg_with_ctx["_prev_true_ranges"] = None
        else:
            reward_cfg_with_ctx["_dmin"] = None
            reward_cfg_with_ctx["_true_ranges"] = None
            reward_cfg_with_ctx["_prev_true_ranges"] = None
            ttc_enabled = False

        terms, new_prev_goal = compute_terms(
            pose,
            waypoints,
            self._prev_goal_dist,
            last_u,
            prev_u,
            self._robot_cfg,
            reward_cfg_with_ctx,
            terminated,
            truncated,
        )
        self._prev_goal_dist = new_prev_goal
        if ttc_enabled:
            self._prev_true_ranges = reward_cfg_with_ctx["_true_ranges"]

        weights = self._reward_cfg["weights"]
        total, contrib = apply_weights(terms, weights)
        breakdown = to_breakdown_dict(terms, weights, total, contrib)
        breakdown.setdefault("metrics", {})
        self._last_breakdown = breakdown
        return RewardResult(total=total, breakdown=breakdown, terms=terms)

    def update_last_breakdown(self, breakdown: Dict[str, Any]) -> None:
        """Allow callers to override the cached breakdown (e.g., to append metrics)."""
        self._last_breakdown = breakdown
