"""SAC training entrypoint using Hydra config composition.

Mirrors train_ppo.py but instantiates SAC with matching logging, callbacks,
and VecNormalize handling. Saves a resolved config snapshot (resolved.yaml)
in the Hydra run directory alongside artifacts.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import hydra
from stable_baselines3 import SAC
from training.callbacks import (
    WandbEvalCallback,
    CheckpointCallbackWithVecnorm,
    RewardTermsLoggingCallback,
)

from training.env_factory import make_vec_envs
from training.feature_extractors import LiDAR1DConvExtractor


def build_policy_kwargs(
    policy_cfg: Dict[str, Any], env_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    # Map simple config to SB3 net_arch (new format for SB3 v1.8.0+)
    actor_sizes = policy_cfg["actor"]["hidden_sizes"]
    critic_sizes = policy_cfg["critic"]["hidden_sizes"]
    net_arch = dict(pi=actor_sizes, qf=critic_sizes)
    # Activation mapping
    import torch.nn as nn

    act_name = policy_cfg["actor"]["activation"]
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    activation_fn = act_map.get(act_name, nn.ReLU)
    kwargs: Dict[str, Any] = {"net_arch": net_arch, "activation_fn": activation_fn}
    # Optional custom feature extractor for LiDAR
    fe_cfg = policy_cfg.get("feature_extractor", {})
    if isinstance(fe_cfg, dict) and fe_cfg.get("lidar_branch", "mlp") == "cnn1d":
        k = int(fe_cfg.get("lidar_k", env_cfg["wrappers"]["frame_stack"]["k"]))
        beams = int(fe_cfg.get("lidar_beams", env_cfg["lidar"]["beams"]))
        kwargs["features_extractor_class"] = LiDAR1DConvExtractor
        kwargs["features_extractor_kwargs"] = {
            "lidar_k": k,
            "lidar_beams": beams,
            "lidar_channels": list(fe_cfg.get("lidar_channels", [16, 32, 16])),
            "kernel_sizes": list(fe_cfg.get("kernel_sizes", [3, 5, 3])),
            "out_dim": int(fe_cfg.get("out_dim", 128)),
            "kin_dim": int(fe_cfg.get("kin_dim", 16)),
            "path_dim": int(fe_cfg.get("path_dim", 16)),
            # Optional temporal conv along stacked frames (disabled by default)
            "temporal_enabled": bool(fe_cfg.get("temporal_enabled", False)),
            "temporal_kernel_size": int(fe_cfg.get("temporal_kernel_size", 3)),
            "temporal_dilation": int(fe_cfg.get("temporal_dilation", 1)),
        }
    return kwargs


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve composed config to plain dicts where needed
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    robot_cfg = OmegaConf.to_container(cfg.robot, resolve=True)
    reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    network_cfg = OmegaConf.to_container(cfg.network, resolve=True)
    wandb_cfg = OmegaConf.to_container(cfg.get("wandb", {}), resolve=True)
    assert isinstance(env_cfg, dict) and isinstance(robot_cfg, dict)
    assert isinstance(reward_cfg, dict) and isinstance(algo_cfg, dict)
    assert isinstance(network_cfg, dict) and isinstance(wandb_cfg, dict)

    algo_gamma = algo_cfg.get("gamma")
    shaping_gamma = (
        reward_cfg.get("shaping", {}).get("gamma")
        if isinstance(reward_cfg.get("shaping"), dict)
        else None
    )
    try:
        if algo_gamma is not None and shaping_gamma is not None:
            if abs(float(algo_gamma) - float(shaping_gamma)) > 1e-6:
                print(
                    "[warn] algo.gamma ({}) != reward.shaping.gamma ({}); potential shaping drift.".format(
                        algo_gamma, shaping_gamma
                    )
                )
    except Exception:
        pass

    # Run configuration
    run_cfg = {"dt": float(cfg.run["dt"]), "max_steps": 600}

    # Hydra sets CWD to the run directory (runs/YYYYMMDD_HHMMSS)
    # Save a resolved config snapshot for later reproduction
    try:
        with open("resolved.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception:
        pass

    # Determine base seed and envs/timesteps
    seed = int(cfg.run["seed"])
    n_envs = int(cfg.run["vec_envs"])
    total_timesteps = int(cfg.run["total_timesteps"])

    # Inform user of output directory (Hydra CWD)
    try:
        import os

        print(f"Training outputs will be saved to: {os.getcwd()}")
    except Exception:
        pass

    # Initialize Weights & Biases (optional)
    wandb_run = None
    try:
        import wandb  # type: ignore

        mode = str(wandb_cfg["mode"])
        if mode != "disabled":
            wandb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                mode=mode,
                sync_tensorboard=True,
                dir=".",
                config={
                    "env": env_cfg,
                    "robot": robot_cfg,
                    "reward": reward_cfg,
                    "algo": algo_cfg,
                    "network": network_cfg,
                    "seed": seed,
                    "n_envs": n_envs,
                },
                tags=wandb_cfg.get("tags", []),
            )
            try:
                wandb.tensorboard.patch(root_logdir="logs")
            except Exception:
                pass
    except Exception:
        wandb_run = None

    # Train and eval envs
    train_env = make_vec_envs(
        env_cfg,
        robot_cfg,
        reward_cfg,
        run_cfg,
        frame_stack_k=int(env_cfg["wrappers"]["frame_stack"]["k"]),
        n_envs=n_envs,
        base_seed=seed,
        use_subproc=(n_envs > 1),
        normalize_obs=bool(algo_cfg.get("normalize_obs", True)),
    )
    eval_env = make_vec_envs(
        env_cfg,
        robot_cfg,
        reward_cfg,
        run_cfg,
        frame_stack_k=int(env_cfg["wrappers"]["frame_stack"]["k"]),
        n_envs=1,
        base_seed=seed + 1000,
        use_subproc=False,
        normalize_obs=bool(algo_cfg.get("normalize_obs", True)),
    )
    # Share VecNormalize stats between train and eval
    try:
        from stable_baselines3.common.vec_env import VecNormalize as _VN

        if isinstance(train_env, _VN) and isinstance(eval_env, _VN):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.training = False
            eval_env.norm_reward = False
    except Exception:
        pass

    policy_kwargs = build_policy_kwargs(network_cfg, env_cfg)

    model = SAC(
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

    # Ensure SB3 uses TB (W&B will sync TB automatically when enabled above)
    try:
        from stable_baselines3.common.logger import configure

        logger = configure("logs", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)
    except Exception:
        pass

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

    from stable_baselines3.common.callbacks import CallbackList

    rt_cb = RewardTermsLoggingCallback(wandb_run=wandb_run)
    callback: Any = CallbackList([eval_cb, rt_cb])
    # Optional periodic checkpointing
    ckpt_cfg = algo_cfg.get("checkpoint", {})
    try:
        if bool(ckpt_cfg.get("enabled", False)):
            ckpt_cb = CheckpointCallbackWithVecnorm(
                save_freq_steps=int(ckpt_cfg.get("every_steps", 100000)),
                save_dir="checkpoints",
                vecnorm_env=train_env,
                keep_last_k=int(ckpt_cfg.get("keep_last_k", 3)),
                prefix=str(ckpt_cfg.get("prefix", "ckpt")),
                to_wandb=bool(ckpt_cfg.get("to_wandb", False)),
                wandb_run=wandb_run,
            )
            callback = CallbackList([eval_cb, rt_cb, ckpt_cb])
    except Exception:
        pass

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save final model in its own directory (like checkpoints)
    import os

    os.makedirs("final", exist_ok=True)
    model.save("final/final_model")


if __name__ == "__main__":
    main()
