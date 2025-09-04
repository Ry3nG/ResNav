from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import gymnasium as gym

from .unicycle_nav_env import UnicycleNavEnv
from ..maps import create_blockage_scenario, BlockageScenarioConfig


@dataclass
class RewardConfig:
    """Simplified reward configuration for safer, more stable learning.

    Design goals:
    - 以“目标距离进展”为核心（势函数差分），倒退为负值；
    - 只在进入安全边距内才施加强惩罚（贴障风险平方）；
    - 保留小幅平滑惩罚，去掉逐步时间压力与生硬的前进/旋转奖励；
    - 追加“无进展罚”，防止长期原地小动作。
    """

    # Goal-distance progress: r = alpha_goal * (d_{t-1} - d_t)
    alpha_goal: float = 2.0

    # Safety: only when r_min below effective safety distance
    beta_risk: float = 10.0
    risk_threshold: float = 0.0  # xgap = d_safe_eff - r_min > threshold 才罚

    # Smoothness
    beta_smooth: float = 0.02

    # No-progress penalty
    stuck_steps: int = 8
    stuck_progress_eps: float = 1e-3
    stuck_speed_eps: float = 0.05
    stuck_penalty: float = 0.5

    # Safety margins for effective distance
    base_margin_m: float = 0.12
    kv_margin_s: float = 0.07
    k_softplus: float = 10.0

    # Terminal rewards
    goal_bonus: float = 100.0
    collision_penalty: float = 200.0
    timeout_penalty: float = 80.0

    # Step reward clipping
    clip_abs: float = 200.0
    step_reward_clip: float = 5.0


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
        
        # Learning-time state tracking
        self._prev_goal_distance: Optional[float] = None  # For goal distance progress
        self._no_progress_count = 0                       # For anti-stalling detection
        self._episode_step_count = 0                      # For logging/timeout checks

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
        
        # Reset learning-time trackers
        self._prev_goal_distance = None  # Will be initialized on first step
        self._no_progress_count = 0
        self._episode_step_count = 0
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
        # Timeout handled as truncated at env level; 这里使用计数判断
        if env._step_count >= env.max_steps:
            return -rc.timeout_penalty

        # Extract values
        n = env.cfg.lidar_beams
        r_min = float(np.min(obs[:n])) if n > 0 else env.cfg.lidar_range
        v_curr = float(obs[n]) if obs.shape[0] > n else 0.0
        robot_r = float(env.cfg.robot_radius)
        d_safe_eff = robot_r + rc.base_margin_m + rc.kv_margin_s * max(0.0, v_curr)

        # === PROGRESS: goal-distance potential difference ===
        clearance_factor = float(np.clip(r_min / max(1e-6, d_safe_eff), 0.0, 1.0))
        r_goal = 0.0
        if env.goal_xy is not None:
            gx, gy = env.goal_xy
            x, y, _ = pose
            curr_goal_dist = float(np.hypot(gx - x, gy - y))
            
            if self._prev_goal_distance is not None:
                # Potential difference: moving away is negative
                goal_progress = self._prev_goal_distance - curr_goal_dist
                # 在不安全时弱化该奖励，避免“顶着障碍直冲”
                r_goal = rc.alpha_goal * goal_progress * (0.5 + 0.5 * clearance_factor)
            
            # Update for next step
            self._prev_goal_distance = curr_goal_dist

        # === SAFETY: selective risk penalty ===
        # Only when inside safety margin (xgap > threshold)
        xgap = d_safe_eff - r_min
        if xgap > rc.risk_threshold:
            s = self._softplus(abs(xgap))
            r_risk = -rc.beta_risk * (s * s)
        else:
            r_risk = 0.0

        # === SMOOTHNESS ===
        a_prev = self._prev_action
        a = np.array(action, dtype=float)
        r_smooth = -rc.beta_smooth * float(np.sum((a - a_prev) ** 2))

        # === NO-PROGRESS PENALTY ===
        v_cmd = float(action[0])
        r_stuck = 0.0
        if env.goal_xy is not None and self._prev_goal_distance is not None:
            goal_progress = self._prev_goal_distance - curr_goal_dist
            if abs(goal_progress) < rc.stuck_progress_eps and v_cmd < rc.stuck_speed_eps:
                self._no_progress_count += 1
            else:
                self._no_progress_count = 0
            if self._no_progress_count >= rc.stuck_steps:
                r_stuck = -rc.stuck_penalty
        
        # Update step counter (for completeness / logging)
        self._episode_step_count += 1

        # === COMBINE ===
        r_step = r_goal + r_risk + r_smooth + r_stuck
        
        # Apply step reward clipping (terminal rewards bypass this)
        r_step = float(np.clip(r_step, -rc.step_reward_clip, rc.step_reward_clip))
        return r_step
