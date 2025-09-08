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
from amr_env.gym.wrappers import DictFrameStackVec


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
    n_envs: int = 8,
    base_seed: int = 0,
    use_subproc: bool = True,
    frame_stack_k: int = 4,
    frame_stack_flatten: bool = True,
    normalize_obs: bool = True,
) -> VecEnv:
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    # Build per-rank thunks with unique seeds
    env_fns = [
        make_env_ctor(env_cfg, robot_cfg, reward_cfg, run_cfg, seed=base_seed + i)
        for i in range(n_envs)
    ]
    venv = vec_cls(env_fns)
    # Apply Dict frame stack on lidar only
    if frame_stack_k > 1:
        venv = DictFrameStackVec(
            venv,
            keys=["lidar"],
            k=frame_stack_k,
            flatten=frame_stack_flatten,
            latest_first=True,
        )
    if normalize_obs:
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return venv
