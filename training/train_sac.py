"""SAC training entry point with Hydra configuration."""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from training.callbacks import (
    CheckpointCallbackWithVecnorm,
    RewardTermsLoggingCallback,
    WandbEvalCallback,
)
from training.config_utils import resolve_cfg, maybe_init_wandb
from training.model_builder import build_policy_kwargs, _init_model, configure_logger
from training.env_setup import make_train_and_eval_envs


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train SAC agent with given Hydra configuration."""
    env_cfg, robot_cfg, reward_cfg, algo_cfg, network_cfg, wandb_cfg, run_cfg = (
        resolve_cfg(cfg)
    )

    # Save resolved config for reference
    with open("resolved.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    seed = int(run_cfg.get("seed", 0))
    total_timesteps = int(run_cfg["total_timesteps"])
    n_envs = int(run_cfg.get("vec_envs", 1))

    # Initialize W&B if enabled
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

    # Create environments
    train_env, eval_env = make_train_and_eval_envs(
        env_cfg, robot_cfg, reward_cfg, run_cfg, algo_cfg
    )

    # Build and initialize model
    policy_kwargs = build_policy_kwargs(network_cfg, env_cfg, algo_name="sac")
    model = _init_model("sac", algo_cfg, policy_kwargs, train_env, seed)
    configure_logger(model)

    # Setup callbacks
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

    # Add checkpoint callback if enabled
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

    # Train
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    # Save final model
    os.makedirs("final", exist_ok=True)
    model.save("final/final_model")

    # Save VecNormalize stats
    if isinstance(train_env, VecNormalize):
        train_env.save("final/vecnorm_final.pkl")
        if os.path.exists("best/best_model.zip") and not os.path.exists(
            "best/vecnorm_best.pkl"
        ):
            os.makedirs("best", exist_ok=True)
            train_env.save("best/vecnorm_best.pkl")

    # Upload to W&B if enabled
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

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
