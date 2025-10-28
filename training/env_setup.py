"""Environment creation orchestration for training and evaluation.

Extracted from training/common.py to separate environment setup logic.
Delegates to training.env_factory for actual VecEnv construction.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.vec_env import VecNormalize

from training.env_factory import make_vec_envs


__all__ = ["make_train_and_eval_envs"]


def make_train_and_eval_envs(
    env_cfg: dict[str, Any],
    robot_cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
):
    """Create training and evaluation vectorized environments.

    Args:
        env_cfg: Environment configuration (scenario, lidar, wrappers, etc.)
        robot_cfg: Robot configuration (radius, speed limits, etc.)
        reward_cfg: Reward function configuration
        run_cfg: Run configuration (dt, max_steps, vec_envs, seed)
        algo_cfg: Algorithm configuration (normalize_obs, etc.)

    Returns:
        Tuple of (train_env, eval_env), both VecEnv instances (possibly wrapped in VecNormalize)
    """
    frame_stack = int(env_cfg["wrappers"]["frame_stack"]["k"])
    n_envs = int(run_cfg.get("vec_envs", 1))
    seed = int(run_cfg.get("seed", 0))
    normalize_obs = bool(algo_cfg.get("normalize_obs", True))

    env_run_cfg = {
        "dt": float(run_cfg["dt"]),
        "max_steps": int(run_cfg.get("max_steps", 600)),
    }

    train_env = make_vec_envs(
        env_cfg,
        robot_cfg,
        reward_cfg,
        env_run_cfg,
        frame_stack_k=frame_stack,
        n_envs=n_envs,
        base_seed=seed,
        use_subproc=(n_envs > 1),
        normalize_obs=normalize_obs,
    )
    eval_env = make_vec_envs(
        env_cfg,
        robot_cfg,
        reward_cfg,
        env_run_cfg,
        frame_stack_k=frame_stack,
        n_envs=1,
        base_seed=seed + 1000,
        use_subproc=False,
        normalize_obs=normalize_obs,
    )

    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms
        eval_env.training = False
        eval_env.norm_reward = False
    return train_env, eval_env
