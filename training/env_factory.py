from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)
from stable_baselines3.common.monitor import Monitor

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec


def make_env_ctor(
    env_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    seed: int,
) -> Callable[[], Any]:
    def _thunk():
        env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
        env.seed(seed)
        return Monitor(env)

    return _thunk


def make_vec_envs(
    env_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    frame_stack_k: int,
    n_envs: int = 8,
    base_seed: int = 0,
    use_subproc: bool = True,
    normalize_obs: bool = True,
) -> VecEnv:
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    # Build per-rank thunks with unique seeds
    env_fns = [
        make_env_ctor(env_cfg, robot_cfg, reward_cfg, run_cfg, seed=base_seed + i)
        for i in range(n_envs)
    ]
    venv = vec_cls(env_fns)
    # Apply lidar frame stacking
    if frame_stack_k > 1:
        venv = LidarFrameStackVec(venv, k=frame_stack_k)
    if normalize_obs:
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return venv


def make_curriculum_vec_envs(
    env_cfg_list: list[Dict[str, Any]],  # List of env configs (e.g., [basic, medium])
    env_ratios: list[float],             # Ratios for each config (e.g., [0.2, 0.7])
    robot_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    frame_stack_k: int,
    n_envs: int = 8,
    base_seed: int = 0,
    use_subproc: bool = True,
    normalize_obs: bool = True,
) -> VecEnv:
    """Create mixed curriculum environments with specified ratios.

    Example for Stage 2 (70% medium, 20% basic, 10% unused):
        env_cfg_list = [basic_cfg, medium_cfg]
        env_ratios = [0.2, 0.7]  # Will round to allocate 20 total envs

    Args:
        env_cfg_list: List of environment configs
        env_ratios: Ratio for each config (must sum to <= 1.0)
        ... (other args same as make_vec_envs)

    Returns:
        VecEnv with mixed difficulty environments
    """
    assert len(env_cfg_list) == len(env_ratios), "Config and ratio lists must match"
    assert sum(env_ratios) <= 1.0, f"Ratios sum to {sum(env_ratios)}, must be <= 1.0"

    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv

    # Allocate envs based on ratios
    env_fns = []
    env_idx = 0

    for cfg, ratio in zip(env_cfg_list, env_ratios):
        count = int(np.round(n_envs * ratio))
        print(f"  Allocating {count}/{n_envs} envs for difficulty: {cfg.get('name', 'unknown')}")
        for _ in range(count):
            env_fns.append(
                make_env_ctor(cfg, robot_cfg, reward_cfg, run_cfg, seed=base_seed + env_idx)
            )
            env_idx += 1

    # Fill remaining slots with primary (last) config
    remaining = n_envs - env_idx
    if remaining > 0:
        primary_cfg = env_cfg_list[-1]  # Use last config as primary
        print(f"  Filling {remaining} remaining envs with primary: {primary_cfg.get('name', 'unknown')}")
        for _ in range(remaining):
            env_fns.append(
                make_env_ctor(primary_cfg, robot_cfg, reward_cfg, run_cfg, seed=base_seed + env_idx)
            )
            env_idx += 1

    venv = vec_cls(env_fns)

    # Apply lidar frame stacking
    if frame_stack_k > 1:
        venv = LidarFrameStackVec(venv, k=frame_stack_k)

    if normalize_obs:
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    return venv
