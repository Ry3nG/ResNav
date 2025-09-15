"""PPO training entrypoint using Hydra config composition.

Replaces argparse with Hydra. Saves a resolved config snapshot (resolved.yaml)
in the Hydra run directory alongside artifacts.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import hydra
from stable_baselines3 import PPO
from training.callbacks import (
    WandbEvalCallback,
    CheckpointCallbackWithVecnorm,
    RewardTermsLoggingCallback,
)

from training.env_factory import make_vec_envs
from training.feature_extractors import LiDAR1DConvExtractor


def build_policy_kwargs(policy_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Map simple config to SB3 net_arch (new format for SB3 v1.8.0+)
    actor_sizes = policy_cfg["actor"]["hidden_sizes"]
    critic_sizes = policy_cfg["critic"]["hidden_sizes"]
    net_arch = dict(pi=actor_sizes, vf=critic_sizes)  # Direct dict, not list
    act_name = policy_cfg["actor"]["activation"]
    # Activation mapping
    import torch.nn as nn

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
            # Sync TB event files into W&B
            try:
                # Our SB3 logger writes TensorBoard to "logs" via configure() below
                # so point W&B TB patch to that directory for live streaming.
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
        frame_stack_k=int(
            env_cfg["wrappers"]["frame_stack"]["k"]
        ),
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
        frame_stack_k=int(
            env_cfg["wrappers"]["frame_stack"]["k"]
        ),
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

    model = PPO(
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
    # Optional periodic checkpointing
    ckpt_cfg = algo_cfg.get("checkpoint", {})
    # Always include reward breakdown logging (default ON)
    from stable_baselines3.common.callbacks import CallbackList

    rt_cb = RewardTermsLoggingCallback(wandb_run=wandb_run)
    callback: Any = CallbackList([eval_cb, rt_cb])
    try:
        if bool(ckpt_cfg["enabled"]):
            ckpt_cb = CheckpointCallbackWithVecnorm(
                save_freq_steps=int(ckpt_cfg["every_steps"]),
                save_dir="checkpoints",
                vecnorm_env=train_env,
                keep_last_k=int(ckpt_cfg["keep_last_k"]),
                prefix=str(ckpt_cfg["prefix"]),
                to_wandb=bool(ckpt_cfg["to_wandb"]),
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

    # Save VecNormalize statistics if present
    try:
        from stable_baselines3.common.vec_env import VecNormalize as _VN

        if isinstance(train_env, _VN):
            # New canonical name
            train_env.save("final/vecnorm_final.pkl")
    except Exception:
        pass

    # Best dir backfill: if best_model.zip exists but vecnorm_best.pkl missing, save it now
    try:
        from stable_baselines3.common.vec_env import VecNormalize as _VN

        if os.path.exists("best/best_model.zip") and isinstance(train_env, _VN):
            os.makedirs("best", exist_ok=True)
            if not os.path.exists("best/vecnorm_best.pkl"):
                print("[TRAIN] Best model exists but vecnorm_best.pkl missing, saving backup from final state")
                try:
                    train_env.save("best/vecnorm_best.pkl")
                    print("[TRAIN] Successfully saved backup vecnorm_best.pkl")
                except Exception as e:
                    print(f"[TRAIN] Failed to save backup vecnorm_best.pkl: {e}")
            else:
                print("[TRAIN] Best model and vecnorm_best.pkl both exist, no backup needed")
    except Exception as e:
        print(f"[TRAIN] Error in best dir backfill logic: {e}")

    # Save artifacts to WandB
    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            art = wandb.Artifact("models", type="model")
            art.add_file("final/final_model.zip")
            # Optional best model and vecnorm
            import os

            if os.path.exists("best/best_model.zip"):
                art.add_file("best/best_model.zip")
            if os.path.exists("final/vecnorm_final.pkl"):
                art.add_file("final/vecnorm_final.pkl")
            if os.path.exists("best/vecnorm_best.pkl"):
                art.add_file("best/vecnorm_best.pkl")
            wandb_run.log_artifact(art)
            wandb_run.finish()
        except Exception:
            pass
    try:
        train_env.close()
        eval_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
