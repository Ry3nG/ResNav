from __future__ import annotations

from typing import Dict, Any

from .base import BaseReward
from .v0 import RewardV0
from .v1 import RewardV1
from importlib import import_module


def build_reward(
    name: str,
    params: Dict[str, Any],
    *,
    dt: float,
    v_max: float,
    w_max: float,
    lidar_max_range: float,
) -> BaseReward:
    key = (name or "v1").lower()
    # Support dotted-path class names for maximum flexibility
    if "." in name:
        mod_path, cls_name = name.rsplit(".", 1)
        mod = import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(params, dt=dt, v_max=v_max, w_max=w_max, lidar_max_range=lidar_max_range)
    if key in ("v0", "legacy", "baseline"):
        return RewardV0(params, dt=dt, v_max=v_max, w_max=w_max, lidar_max_range=lidar_max_range)
    # Default to v1
    return RewardV1(params, dt=dt, v_max=v_max, w_max=w_max, lidar_max_range=lidar_max_range)


__all__ = ["build_reward", "BaseReward", "RewardV0", "RewardV1"]
