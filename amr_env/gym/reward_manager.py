"""Reward bookkeeping for ResidualNavEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    breakdown: dict[str, Any]
    terms: RewardTerms


class RewardManager:
    """Encapsulates reward configuration and per-episode state."""

    def __init__(
        self,
        robot_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
    ) -> None:
        self._robot_cfg = robot_cfg
        self._reward_cfg = reward_cfg
        self._prev_goal_dist: float | None = None
        self._last_breakdown: dict[str, Any] = {}

    def reset(self) -> None:
        self._prev_goal_dist = None
        self._last_breakdown = {}

    @property
    def last_breakdown(self) -> dict[str, Any]:
        return self._last_breakdown

    def compute(
        self,
        pose: tuple[float, float, float],
        waypoints: np.ndarray,
        last_u: tuple[float, float],
        prev_u: tuple[float, float],
        terminated: bool,
        truncated: bool,
        path_context: Any | None,
        clearance_m: float | None,
    ) -> RewardResult:
        reward_cfg_with_ctx = dict(self._reward_cfg)
        reward_cfg_with_ctx["_ctx"] = path_context

        safety_cfg = self._reward_cfg.get("safety") or {}
        if not isinstance(safety_cfg, dict):
            safety_cfg = {}
        use_map_barrier = str(safety_cfg.get("source", "")).lower() == "map"

        if use_map_barrier:
            reward_cfg_with_ctx["_dmin"] = float(clearance_m) if clearance_m is not None else None
        else:
            reward_cfg_with_ctx["_dmin"] = None

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

        weights = self._reward_cfg["weights"]
        total, contrib = apply_weights(terms, weights)
        breakdown = to_breakdown_dict(terms, weights, total, contrib)
        breakdown.setdefault("metrics", {})
        self._last_breakdown = breakdown
        return RewardResult(total=total, breakdown=breakdown, terms=terms)

    def update_last_breakdown(self, breakdown: dict[str, Any]) -> None:
        """Allow callers to override the cached breakdown (e.g., to append metrics)."""
        self._last_breakdown = breakdown
