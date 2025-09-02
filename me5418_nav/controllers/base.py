from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Tuple
import numpy as np


@dataclass
class ControlCommand:
    v: float
    w: float


class Controller(Protocol):
    def action(
        self,
        pose: Tuple[float, float, float],
        waypoints: Optional[np.ndarray],
        lidar,
        grids,  # expects a Grids-like object with .sensing and .cspace
        v_limits: Tuple[float, float],
        w_limits: Tuple[float, float],
        goal_xy: Optional[Tuple[float, float]],
        v_curr: float,
        w_curr: float,
    ) -> ControlCommand:
        ...

