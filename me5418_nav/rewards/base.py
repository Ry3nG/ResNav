from __future__ import annotations

from typing import Dict, Any, Tuple
import abc
import numpy as np


class BaseReward(abc.ABC):
    def __init__(self, params: Dict[str, Any], *, dt: float, v_max: float, w_max: float, lidar_max_range: float) -> None:
        self.params = dict(params)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.lidar_max_range = float(lidar_max_range)

    @abc.abstractmethod
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
        """Return (reward, debug_dict)."""
        raise NotImplementedError

