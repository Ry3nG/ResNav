from __future__ import annotations

from typing import Dict, Any, Tuple
import math
import numpy as np

from .base import BaseReward


class RewardV0(BaseReward):
    """Legacy reward used in initial experiments.

    r = w_prog*max(0,ds)
        - w_lat*|e_lat| - w_head*|e_head|
        - w_clear*exp(-max(0,min_lidar)/clearance_safe_m)
        - w_dv*|Δv| - w_dw*|Δw|
        + terminals
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        super().__init__(params, **kwargs)
        p = self.params
        # Defaults follow the original environment defaults
        self.w_prog = float(p.get("w_prog", 1.0))
        self.w_lat = float(p.get("w_lat", 0.2))
        self.w_head = float(p.get("w_head", 0.1))
        self.w_clear = float(p.get("w_clear", 0.4))
        self.w_dv = float(p.get("w_dv", 0.05))
        self.w_dw = float(p.get("w_dw", 0.02))
        self.R_goal = float(p.get("R_goal", 50.0))
        self.R_collide = float(p.get("R_collide", 50.0))
        self.R_timeout = float(p.get("R_timeout", 10.0))
        self.clearance_safe_m = float(p.get("clearance_safe_m", 0.5))

    def compute(
        self,
        *,
        ds: float,
        v_cmd: float,
        w_cmd: float,
        v_prev: float,
        w_prev: float,
        ranges: np.ndarray,
        angles: np.ndarray,
        min_lidar: float,
        e_lat: float,
        e_head: float,
        goal: bool,
        collision: bool,
        timeout: bool,
        truncated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        dv = float(v_cmd - v_prev)
        dw = float(w_cmd - w_prev)
        r = 0.0
        r += self.w_prog * max(0.0, ds)
        r -= self.w_lat * abs(e_lat)
        r -= self.w_head * abs(e_head)
        r -= self.w_clear * math.exp(-max(0.0, float(min_lidar)) / max(1e-6, self.clearance_safe_m))
        r -= self.w_dv * abs(dv)
        r -= self.w_dw * abs(dw)
        if collision:
            r -= self.R_collide
        elif timeout or truncated:
            r -= self.R_timeout
        elif goal:
            r += self.R_goal
        dbg = {
            "dv": dv,
            "dw": dw,
        }
        return float(r), dbg

