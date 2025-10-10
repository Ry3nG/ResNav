from __future__ import annotations

import os
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from training.callbacks import (
    CheckpointCallbackWithVecnorm,
    RewardTermsLoggingCallback,
    WandbEvalCallback,
)
from training.env_factory import make_vec_envs
from training.feature_extractors import LiDAR1DConvExtractor


def _to_dict(cfg_section: Any) -> dict[str, Any]:
    if isinstance(cfg_section, DictConfig):
        data = OmegaConf.to_container(cfg_section, resolve=True)
    else:
        data = cfg_section
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Expected config section to resolve to a dict")
    return dict(data)


def resolve_cfg(
    cfg: DictConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    env_cfg = _to_dict(cfg.env)
    robot_cfg = _to_dict(cfg.robot)
    reward_cfg = _to_dict(cfg.reward)
    algo_cfg = _to_dict(cfg.algo)
    network_cfg = _to_dict(cfg.network)
    wandb_cfg = _to_dict(cfg.get("wandb", {}))
    run_cfg = _to_dict(cfg.run)
    return env_cfg, robot_cfg, reward_cfg, algo_cfg, network_cfg, wandb_cfg, run_cfg


def maybe_init_wandb(wandb_cfg: dict[str, Any], extra_config: dict[str, Any]):
    mode = str(wandb_cfg.get("mode", "disabled"))
    if mode == "disabled":
        return None

    import wandb

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group"),
        mode=mode,
        sync_tensorboard=True,
        dir=".",
        config=extra_config,
        tags=wandb_cfg.get("tags", []),
    )
    return run


def build_policy_kwargs(network_cfg: dict[str, Any], env_cfg: dict[str, Any], algo_name: str) -> dict[str, Any]:
    actor_cfg = network_cfg.get("actor", {})
    critic_cfg = network_cfg.get("critic", {})
    activation = str(actor_cfg.get("activation", "relu")).lower()
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    net_arch: dict[str, Any]
    if algo_name == "ppo":
        net_arch = {
            "pi": actor_cfg.get("hidden_sizes", []),
            "vf": critic_cfg.get("hidden_sizes", []),
        }
    else:
        net_arch = {
            "pi": actor_cfg.get("hidden_sizes", []),
            "qf": critic_cfg.get("hidden_sizes", []),
        }

    policy_kwargs: dict[str, Any] = {
        "net_arch": net_arch,
        "activation_fn": act_map.get(activation, nn.ReLU),
    }

    fe_cfg = network_cfg.get("feature_extractor", {})
    if isinstance(fe_cfg, dict) and fe_cfg.get("lidar_branch", "mlp") == "cnn1d":
        frame_stack = int(env_cfg["wrappers"]["frame_stack"]["k"])
        beams = int(env_cfg["lidar"]["beams"])
        policy_kwargs["features_extractor_class"] = LiDAR1DConvExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "lidar_k": int(fe_cfg.get("lidar_k", frame_stack)),
            "lidar_beams": int(fe_cfg.get("lidar_beams", beams)),
            "lidar_channels": list(fe_cfg.get("lidar_channels", [16, 32, 16])),
            "kernel_sizes": list(fe_cfg.get("kernel_sizes", [3, 5, 3])),
            "out_dim": int(fe_cfg.get("out_dim", 128)),
            "kin_dim": int(fe_cfg.get("kin_dim", 16)),
            "path_dim": int(fe_cfg.get("path_dim", 16)),
            "temporal_enabled": bool(fe_cfg.get("temporal_enabled", False)),
            "temporal_kernel_size": int(fe_cfg.get("temporal_kernel_size", 3)),
            "temporal_dilation": int(fe_cfg.get("temporal_dilation", 1)),
        }
    return policy_kwargs


def make_train_and_eval_envs(
    env_cfg: dict[str, Any],
    robot_cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
):
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


def configure_logger(model) -> None:
    logger = configure("logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)


def _init_model(algo_name: str, algo_cfg: dict[str, Any], policy_kwargs: dict[str, Any], train_env, seed: int):
    if algo_name == "ppo":
        return PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=float(algo_cfg["lr"]),
            n_steps=int(algo_cfg["n_steps"]),
            batch_size=int(algo_cfg["batch_size"]),
            n_epochs=int(algo_cfg["n_epochs"]),
            gamma=float(algo_cfg["gamma"]),
            gae_lambda=float(algo_cfg["gae_lambda"]),
            clip_range=float(algo_cfg["clip_range"]),
            ent_coef=float(algo_cfg["ent_coef"]),
            vf_coef=float(algo_cfg["vf_coef"]),
            max_grad_norm=float(algo_cfg["max_grad_norm"]),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log="tb_logs",
        )
    return SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=float(algo_cfg["lr"]),
        buffer_size=int(algo_cfg["buffer_size"]),
        batch_size=int(algo_cfg["batch_size"]),
        tau=float(algo_cfg.get("tau", 0.005)),
        gamma=float(algo_cfg["gamma"]),
        train_freq=int(algo_cfg.get("train_freq", 1)),
        gradient_steps=int(algo_cfg.get("gradient_steps", 1)),
        learning_starts=int(algo_cfg.get("learning_starts", 10000)),
        ent_coef=algo_cfg.get("ent_coef", "auto"),
        target_update_interval=int(algo_cfg.get("target_update_interval", 1)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log="tb_logs",
    )


def train_with_algo(cfg: DictConfig, algo_name: str) -> None:
    env_cfg, robot_cfg, reward_cfg, algo_cfg, network_cfg, wandb_cfg, run_cfg = resolve_cfg(cfg)

    with open("resolved.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    seed = int(run_cfg.get("seed", 0))
    total_timesteps = int(run_cfg["total_timesteps"])
    n_envs = int(run_cfg.get("vec_envs", 1))

    wandb_run = maybe_init_wandb(
        wandb_cfg,
        {
            "env": env_cfg,
            "robot": robot_cfg,
            "reward": reward_cfg,
            "algo": algo_cfg,
            "network": network_cfg,
            "seed": seed,
            "n_envs": n_envs,
        },
    )

    train_env, eval_env = make_train_and_eval_envs(
        env_cfg, robot_cfg, reward_cfg, run_cfg, algo_cfg
    )
    policy_kwargs = build_policy_kwargs(network_cfg, env_cfg, algo_name)
    model = _init_model(algo_name, algo_cfg, policy_kwargs, train_env, seed)
    configure_logger(model)

    eval_cb = WandbEvalCallback(
        eval_env,
        best_model_save_path="best",
        log_path="eval",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        wandb_run=wandb_run,
        vecnorm_env=train_env,
    )
    callbacks = [eval_cb, RewardTermsLoggingCallback(wandb_run=wandb_run)]

    ckpt_cfg = algo_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled"):
        callbacks.append(
            CheckpointCallbackWithVecnorm(
                save_freq_steps=int(ckpt_cfg.get("every_steps", 100_000)),
                save_dir="checkpoints",
                vecnorm_env=train_env,
                keep_last_k=int(ckpt_cfg.get("keep_last_k", 3)),
                prefix=str(ckpt_cfg.get("prefix", "ckpt")),
                to_wandb=bool(ckpt_cfg.get("to_wandb", False)),
                wandb_run=wandb_run,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    os.makedirs("final", exist_ok=True)
    model.save("final/final_model")

    if isinstance(train_env, VecNormalize):
        train_env.save("final/vecnorm_final.pkl")
        if os.path.exists("best/best_model.zip") and not os.path.exists("best/vecnorm_best.pkl"):
            os.makedirs("best", exist_ok=True)
            train_env.save("best/vecnorm_best.pkl")

    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact("models", type="model")
        artifact.add_file("final/final_model.zip")
        for path in (
            "best/best_model.zip",
            "final/vecnorm_final.pkl",
            "best/vecnorm_best.pkl",
        ):
            if os.path.exists(path):
                artifact.add_file(path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    train_env.close()
    eval_env.close()
