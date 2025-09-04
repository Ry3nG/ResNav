from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import gymnasium as gym

from .unicycle_nav_env import UnicycleNavEnv
from ..maps import create_blockage_scenario, BlockageScenarioConfig


@dataclass
class RewardConfig:
    # Progress reward (increased for stronger incentive)
    alpha_progress: float = 2.5
    # Safety penalties (reduced beta_risk to be less conservative)
    beta_risk: float = 8.0
    beta_lat: float = 0.5
    beta_hdg: float = 0.1
    beta_smooth: float = 0.05
    # Safety margins
    base_margin_m: float = 0.10
    kv_margin_s: float = 0.05
    k_softplus: float = 10.0
    # Terminal rewards (rebalanced)
    goal_bonus: float = 50.0
    collision_penalty: float = 50.0
    timeout_penalty: float = 25.0  # Increased from 10.0 to make timeout costlier
    # Exploration incentives (new)
    velocity_bonus: float = 0.1
    spin_penalty: float = 1.0
    # Reward clipping (fixed)
    clip_abs: float = 60.0  # Increased from 5.0 to not clip terminal rewards
    step_reward_clip: float = 8.0  # Separate clipping for step rewards


class BlockageRLWrapper(gym.Wrapper):
    """
    RL wrapper around UnicycleNavEnv that:
    - Regenerates a blockage scenario on each reset
    - Computes shaped rewards for static blockage navigation
    """

    def __init__(
        self,
        env: UnicycleNavEnv,
        scenario_cfg: Optional[BlockageScenarioConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
        seed_per_episode: bool = True,
    ) -> None:
        super().__init__(env)
        self._scen_cfg = scenario_cfg or BlockageScenarioConfig()
        self._rew_cfg = reward_cfg or RewardConfig()
        self._seed = int(seed) if seed is not None else None
        self._seed_per_episode = seed_per_episode
        self._rng = np.random.default_rng(self._seed)
        self._scen_info: dict | None = None

        # Caches for reward terms
        self._prev_pose: Optional[Tuple[float, float, float]] = None
        self._prev_action = np.zeros(2, dtype=float)
        self._prev_ct_err = 0.0
        self._prev_hdg_err = 0.0

    # --------- Scenario management ----------
    def _regen_scenario(self) -> None:
        cfg = self._scen_cfg
        # Choose a new seed if requested
        if self._seed_per_episode:
            cfg = BlockageScenarioConfig(
                **{**self._scen_cfg.__dict__},  # shallow copy
            )
            cfg.random_seed = int(self._rng.integers(0, 2**31 - 1))

        grid, waypoints, start_pose, goal_xy, scen_info = create_blockage_scenario(cfg)
        # Replace underlying env scenario
        env: UnicycleNavEnv = self.env
        env.grid = grid
        env.path_waypoints = waypoints
        env._start_pose = start_pose
        env.goal_xy = goal_xy
        self._scen_info = dict(scen_info) if scen_info is not None else {}
        # Keep cfg map size consistent (not strictly required but tidy)
        H, W = grid.grid.shape
        env.cfg.map_size = (H, W)
        env.rebuild_cspace()

    # --------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # Reset wrapper RNG as well
            self._rng = np.random.default_rng(seed)
        self._regen_scenario()
        obs, info = self.env.reset(seed=seed, options=options)
        # Initialize caches
        self._prev_pose = self.env.robot.as_pose()
        # Extract path errors directly from observation to avoid recomputation
        n = self.env.cfg.lidar_beams
        path_err_start = n + 2  # [lidar (N), v, omega, path_err(2), ...]
        if obs.shape[0] >= path_err_start + 2:
            self._prev_ct_err = float(obs[path_err_start])
            self._prev_hdg_err = float(obs[path_err_start + 1])
        else:
            self._prev_ct_err, self._prev_hdg_err = 0.0, 0.0
        self._prev_action[:] = 0.0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        pose = self.env.robot.as_pose()
        # Extract current path errors from observation
        n = self.env.cfg.lidar_beams
        path_err_start = n + 2
        if obs.shape[0] >= path_err_start + 2:
            ct_curr = float(obs[path_err_start])
            hdg_curr = float(obs[path_err_start + 1])
        else:
            ct_curr, hdg_curr = 0.0, 0.0

        reward = self._compute_reward(
            self._prev_pose, pose, obs, action, info, ct_curr, hdg_curr
        )
        # Update caches
        self._prev_pose = pose
        self._prev_ct_err, self._prev_hdg_err = ct_curr, hdg_curr
        self._prev_action = np.array(action, dtype=float)
        # Attach scenario metadata for logging at episode end
        try:
            scen = self._scen_info or {}
            info = dict(info)
            # Provide compatibility key for SB3 EvalCallback success-rate
            if "success" in info and "is_success" not in info:
                info["is_success"] = bool(info.get("success", False))
            if "num_pallets" in scen:
                info["scenario_num_pallets"] = int(scen.get("num_pallets", 0))
            if "min_clearance" in scen:
                info["scenario_min_clearance"] = float(scen.get("min_clearance", 0.0))
            if "difficulty_score" in scen:
                info["scenario_difficulty"] = float(scen.get("difficulty_score", 0.0))
            if "actual_seed" in scen:
                info["scenario_seed"] = int(scen.get("actual_seed"))
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

    # --------- Reward terms ----------
    def _softplus(self, x: float) -> float:
        k = float(self._rew_cfg.k_softplus)
        # numerically stable softplus
        return float(np.log1p(np.exp(np.clip(k * x, -50, 50))) / k)

    def _compute_reward(
        self,
        prev_pose: Optional[Tuple[float, float, float]],
        pose: Tuple[float, float, float],
        obs: np.ndarray,
        action: Tuple[float, float],
        info: dict,
        ct_curr: float,
        hdg_curr: float,
    ) -> float:
        rc = self._rew_cfg
        env: UnicycleNavEnv = self.env

        # Terminal bonuses
        if info.get("collision", False):
            return -rc.collision_penalty
        if info.get("success", False):
            return rc.goal_bonus
        if env._step_count >= env.max_steps:
            return -rc.timeout_penalty

        # Extract values
        n = env.cfg.lidar_beams
        r_min = float(np.min(obs[:n])) if n > 0 else env.cfg.lidar_range
        v_curr = float(obs[n]) if obs.shape[0] > n else 0.0
        robot_r = float(env.cfg.robot_radius)
        d_safe_eff = robot_r + rc.base_margin_m + rc.kv_margin_s * max(0.0, v_curr)

        # Progress along path: project displacement onto path tangent
        ds = 0.0
        if prev_pose is not None:
            px, py, _ = prev_pose
            x, y, _ = pose
            # tangent at current pose's nearest segment
            idx = env._nearest_path_index(x, y)
            wp = env.path_waypoints
            j = min(idx + 1, wp.shape[0] - 1)
            ab = wp[j] - wp[idx]
            ab_len = float(np.linalg.norm(ab))
            if ab_len > 1e-9:
                t_hat = ab / ab_len
                disp = np.array([x - px, y - py], dtype=float)
                ds = float(max(0.0, np.dot(disp, t_hat)))

        clearance_factor = min(1.0, r_min / max(1e-6, d_safe_eff))
        # Optional floor to avoid zeroing progress entirely in narrow but valid gaps
        clearance_factor = max(0.2, clearance_factor)
        r_prog = rc.alpha_progress * ds * clearance_factor

        # Safety barrier (smooth)
        xgap = d_safe_eff - r_min
        s = self._softplus(xgap)
        r_risk = -rc.beta_risk * (s * s)

        # Path shaping (potential differences)
        r_path = -rc.beta_lat * (abs(ct_curr) - abs(self._prev_ct_err)) - rc.beta_hdg * (
            abs(hdg_curr) - abs(self._prev_hdg_err)
        )

        # Smoothness
        a_prev = self._prev_action
        a = np.array(action, dtype=float)
        r_smooth = -rc.beta_smooth * float(np.sum((a - a_prev) ** 2))

        # NEW: Velocity bonus (encourage forward motion)
        v_cmd, w_cmd = float(action[0]), float(action[1])
        r_velocity = rc.velocity_bonus * max(0.0, v_cmd)
        
        # NEW: Spin penalty (discourage excessive rotation without progress)
        # Only penalize high angular velocity if linear velocity is low
        if v_cmd < 0.1 and abs(w_cmd) > 0.5:  # Spinning in place
            r_spin = -rc.spin_penalty * abs(w_cmd)
        else:
            r_spin = 0.0

        # Combine step rewards
        r_step = r_prog + r_risk + r_path + r_smooth + r_velocity + r_spin
        
        # Apply step reward clipping (separate from terminal rewards)
        r_step = float(np.clip(r_step, -rc.step_reward_clip, rc.step_reward_clip))
        return r_step
