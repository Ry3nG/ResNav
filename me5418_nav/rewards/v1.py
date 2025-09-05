from __future__ import annotations

from typing import Dict, Any, Tuple
import math
import numpy as np

from .base import BaseReward


class RewardV1(BaseReward):
    """Directional shaping reward that encourages early detours.

    r = w_prog*max(0,ds)
        + r_turn + r_fwd + r_gap + r_brake
        - w_dv*|Δv| - w_dw*|Δw|
        - w_step*dt
        + terminals
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        super().__init__(params, **kwargs)
        p = self.params
        self.w_prog = float(p.get("w_prog", 1.0))
        self.w_turn = float(p.get("w_turn", 0.35))
        self.w_fwd = float(p.get("w_fwd", 0.25))
        self.w_gap = float(p.get("w_gap", 0.10))
        self.w_brake = float(p.get("w_brake", 0.20))
        self.w_step = float(p.get("w_step", 0.25))
        self.w_dv = float(p.get("w_dv", 0.02))
        self.w_dw = float(p.get("w_dw", 0.02))
        self.R_goal = float(p.get("R_goal", 50.0))
        self.R_collide = float(p.get("R_collide", 120.0))
        self.R_timeout = float(p.get("R_timeout", 15.0))
        # Thresholds for braking logic
        self.tau_thr = float(p.get("tau_thr", 0.4))
        self.tau = float(p.get("tau", 0.08))

    @staticmethod
    def _sector_means(s: np.ndarray, w_ang: np.ndarray, angles: np.ndarray, th_f: float, th_lr: float) -> tuple[float, float, float]:
        mask_F = np.abs(angles) <= th_f
        mask_L = (angles > 0.0) & (angles <= th_lr)
        mask_R = (angles < 0.0) & (angles >= -th_lr)

        def wmean(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            ww = w_ang[mask]
            ss = s[mask]
            denom = float(np.sum(ww)) + 1e-6
            return float(np.sum(ww * ss) / denom)

        m_F = wmean(mask_F)
        m_L = wmean(mask_L)
        m_R = wmean(mask_R)
        return m_F, m_L, m_R

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
        # Normalized distances with cap at max range
        s = np.clip(ranges / (self.lidar_max_range + 1e-12), 0.0, 1.0)
        # Front-half weighting
        w_ang = np.maximum(0.0, np.cos(angles))
        th_f = math.radians(35.0)
        th_lr = math.radians(80.0)
        m_F, m_L, m_R = self._sector_means(s, w_ang, angles, th_f, th_lr)
        m_side = max(m_L, m_R)
        denom_lr = (m_L + m_R) + 1e-6
        delta_LR = float((m_R - m_L) / denom_lr)

        v_norm = float(np.clip(v_cmd / self.v_max, -1.0, 1.0))
        w_norm = float(np.clip(w_cmd / self.w_max, -1.0, 1.0))

        # Directional shaping
        r_turn = float(self.w_turn * w_norm * delta_LR)
        denom_fs = (m_F + m_side) + 1e-6
        r_fwd = float(self.w_fwd * (m_F - m_side) / denom_fs)
        r_gap = float(-self.w_gap * abs(m_L - m_R) / (denom_lr))
        sig = 1.0 / (1.0 + math.exp(-((self.tau_thr - m_F) / (self.tau + 1e-6))))
        r_brake = float(-self.w_brake * max(0.0, v_norm) * sig)

        dv = float(v_cmd - v_prev)
        dw = float(w_cmd - w_prev)

        r = 0.0
        # Progress
        r += self.w_prog * max(0.0, ds)
        # Directional components
        r += r_turn + r_fwd + r_gap + r_brake
        # Smoothness
        r -= self.w_dv * abs(dv)
        r -= self.w_dw * abs(dw)
        # Per-step time penalty
        r -= self.w_step * self.dt
        # Terminals
        if collision:
            r -= self.R_collide
        elif timeout or truncated:
            r -= self.R_timeout
        elif goal:
            r += self.R_goal

        dbg = {
            "m_F": float(m_F),
            "m_L": float(m_L),
            "m_R": float(m_R),
            "delta_LR": float(delta_LR),
            "v_norm": float(v_norm),
            "w_norm": float(w_norm),
            "r_turn": float(r_turn),
            "r_fwd": float(r_fwd),
            "r_gap": float(r_gap),
            "r_brake": float(r_brake),
            "dv": float(dv),
            "dw": float(dw),
        }
        return float(r), dbg

