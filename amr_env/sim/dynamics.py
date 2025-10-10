"""Unicycle dynamics and integration utilities.

Pure Python, self-contained unicycle kinematics used by the Gym env.
Implements action clipping, Euler integration, and angle normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, pi


@dataclass
class UnicycleState:
    """Robot state for unicycle kinematics.

    - x, y: position (meters)
    - theta: heading (radians, wrapped to [-pi, pi])
    - v: applied linear speed (m/s)
    - omega: applied angular speed (rad/s)
    """

    x: float
    y: float
    theta: float
    v: float
    omega: float


class UnicycleModel:
    """Minimal unicycle model with command limits and Euler integration.

    Interface:
    - reset(state?) -> UnicycleState
    - step((v, w), dt) -> UnicycleState
    - as_pose() -> (x, y, theta)
    - get_state() -> UnicycleState
    """

    def __init__(
        self,
        v_max: float,
        w_max: float,
        v_min: float = 0.0,
        w_min: float | None = None,
    ) -> None:
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.v_min = float(v_min)
        self.w_min = float(-w_max if w_min is None else w_min)
        self._last_state = UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def wrap_to_pi(theta: float) -> float:
        """Normalize angle to [-pi, pi)."""
        wrapped = (theta + pi) % (2.0 * pi) - pi
        if wrapped >= pi:
            wrapped -= 2.0 * pi
        return wrapped

    def clip_action(self, u: tuple[float, float]) -> tuple[float, float]:
        v_cmd, w_cmd = u
        v_applied = min(max(v_cmd, self.v_min), self.v_max)
        w_applied = min(max(w_cmd, self.w_min), self.w_max)
        return v_applied, w_applied

    def reset(self, state: UnicycleState | None = None) -> UnicycleState:
        s = state or UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0)
        # Ensure heading and commanded velocities are consistent with limits
        theta = self.wrap_to_pi(s.theta)
        v_applied, w_applied = self.clip_action((s.v, s.omega))
        self._last_state = UnicycleState(s.x, s.y, theta, v_applied, w_applied)
        return self._last_state

    def step(self, action: tuple[float, float], dt: float) -> UnicycleState:
        """Apply clipped (v, w) for duration dt using Euler integration."""
        v, w = self.clip_action(action)
        x0, y0, th0 = self._last_state.x, self._last_state.y, self._last_state.theta

        x = x0 + v * cos(th0) * dt
        y = y0 + v * sin(th0) * dt
        th = self.wrap_to_pi(th0 + w * dt)

        self._last_state = UnicycleState(x, y, th, v, w)
        return self._last_state

    def as_pose(self) -> tuple[float, float, float]:
        s = self._last_state
        return (s.x, s.y, s.theta)

    def get_state(self) -> UnicycleState:
        return self._last_state
