from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from roboticstoolbox.mobile import Unicycle  # type: ignore
from ..constants import (
    ROBOT_V_MIN_MPS,
    ROBOT_V_MAX_MPS,
    ROBOT_W_MIN_RPS,
    ROBOT_W_MAX_RPS,
)


@dataclass
class UnicycleState:
    x: float
    y: float
    theta: float
    v: float
    omega: float


class UnicycleModel:
    """
    Unicycle kinematics model backed by RTB's mobile.Unicycle.

    Provides a stable interface: reset(), step(), as_pose() and exposes
    velocity limits (v_min/v_max/w_min/w_max).
    """

    def __init__(
        self,
        v_limits: tuple[float, float] = (
            ROBOT_V_MIN_MPS,
            ROBOT_V_MAX_MPS,
        ),
        omega_limits: tuple[float, float] = (
            ROBOT_W_MIN_RPS,
            ROBOT_W_MAX_RPS,
        ),
    ) -> None:
        self._veh = None
        self._last_state = UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.v_min, self.v_max = v_limits
        self.w_min, self.w_max = omega_limits
        self._veh = Unicycle()
        self._veh._x = np.array([0.0, 0.0, 0.0])

    def reset(self, state: UnicycleState | None = None) -> UnicycleState:
        s = state or UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_state = s
        self._veh._x = np.array([s.x, s.y, s.theta])
        return self._last_state

    def step(self, action: Tuple[float, float], dt: float) -> UnicycleState:
        v_cmd, w_cmd = float(action[0]), float(action[1])
        # Set the timestep
        self._veh._dt = dt
        # For Unicycle, step expects (v, w) control inputs  
        self._veh.step((v_cmd, w_cmd))
        x = float(self._veh.x[0])
        y = float(self._veh.x[1])
        th = float(self._veh.x[2])
        self._last_state = UnicycleState(x, y, th, v_cmd, w_cmd)
        return self._last_state

    def as_pose(self) -> tuple[float, float, float]:
        s = self._last_state
        return (s.x, s.y, s.theta)
    
    def get_state(self) -> UnicycleState:
        """Get the current robot state."""
        return self._last_state
