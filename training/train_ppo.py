"""PPO training entrypoint with VecEnv, Dict frame stack, and normalization."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from omegaconf import OmegaConf
from stable_baselines3 import PPO
from training.callbacks import WandbEvalCallback, CheckpointCallbackWithVecnorm

from training.env_factory import make_vec_envs


def load_yaml(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def build_policy_kwargs(policy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Map simple config to SB3 net_arch (new format for SB3 v1.8.0+)
    actor_sizes = policy_cfg.get("actor", {}).get("hidden_sizes", [128, 128])
    critic_sizes = policy_cfg.get("critic", {}).get("hidden_sizes", [128, 128])
    net_arch = dict(pi=actor_sizes, vf=critic_sizes)  # Direct dict, not list
    act_name = policy_cfg.get("actor", {}).get("activation", "relu")
    # Activation mapping
    import torch.nn as nn

    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    activation_fn = act_map.get(act_name, nn.ReLU)
    return {"net_arch": net_arch, "activation_fn": activation_fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", default="configs/env/blockage.yaml")
    parser.add_argument("--robot_cfg", default="configs/robot/default.yaml")
    parser.add_argument("--reward_cfg", default="configs/reward/default.yaml")
    parser.add_argument("--algo_cfg", default="configs/algo/ppo.yaml")
    parser.add_argument("--policy_cfg", default="configs/policy/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    try:
        env_cfg = load_yaml(args.env_cfg)
        robot_cfg = load_yaml(args.robot_cfg)
        reward_cfg = load_yaml(args.reward_cfg)
        algo_cfg = load_yaml(args.algo_cfg)
        policy_cfg = load_yaml(args.policy_cfg)
    except FileNotFoundError as e:
        raise SystemExit(f"Config file not found: {e}")
    run_cfg = {"dt": env_cfg.get("run", {}).get("dt", 0.1), "max_steps": 600}

    # Output directory with timestamp to avoid collisions
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = env_cfg.get("run", {}).get("out_dir", "runs")
    out_dir = f"{base_dir}/{timestamp}"

    import os

    os.makedirs(out_dir, exist_ok=True)
    print(f"Training outputs will be saved to: {out_dir}")

    # Initialize Weights & Biases (optional)
    wandb_run = None
    try:
        import wandb  # type: ignore

        wb_cfg = load_yaml("configs/wandb/default.yaml")
        mode = wb_cfg.get("mode", "online")
        if mode != "disabled":
            wandb_run = wandb.init(
                project=wb_cfg.get("project", "amr_residual_nav"),
                entity=wb_cfg.get("entity"),
                mode=mode,
                sync_tensorboard=True,
                dir=out_dir,
                config={
                    "env": env_cfg,
                    "robot": robot_cfg,
                    "reward": reward_cfg,
                    "algo": algo_cfg,
                    "policy": policy_cfg,
                    "seed": args.seed,
                    "n_envs": args.n_envs,
                },
                tags=wb_cfg.get("tags", []),
            )
            # Sync TB event files into W&B
            try:
                wandb.tensorboard.patch(root_logdir=f"{out_dir}/tb_logs")
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
        n_envs=args.n_envs,
        base_seed=args.seed,
        use_subproc=(args.n_envs > 1),
        frame_stack_k=int(
            env_cfg.get("wrappers", {}).get("frame_stack", {}).get("k", 4)
        ),
        frame_stack_flatten=bool(
            env_cfg.get("wrappers", {}).get("frame_stack", {}).get("flatten", True)
        ),
        normalize_obs=bool(algo_cfg.get("normalize_obs", True)),
    )
    eval_env = make_vec_envs(
        env_cfg,
        robot_cfg,
        reward_cfg,
        run_cfg,
        n_envs=1,
        base_seed=args.seed + 1000,
        use_subproc=False,
        frame_stack_k=int(
            env_cfg.get("wrappers", {}).get("frame_stack", {}).get("k", 4)
        ),
        frame_stack_flatten=bool(
            env_cfg.get("wrappers", {}).get("frame_stack", {}).get("flatten", True)
        ),
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

    policy_kwargs = build_policy_kwargs(policy_cfg)

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=float(algo_cfg.get("lr", 3e-4)),
        n_steps=int(algo_cfg.get("n_steps", 2048)),
        batch_size=int(algo_cfg.get("batch_size", 256)),
        n_epochs=int(algo_cfg.get("n_epochs", 10)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(algo_cfg.get("clip_range", 0.2)),
        ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=f"{out_dir}/tb_logs",
    )

    # Ensure SB3 uses TB (W&B will sync TB automatically when enabled above)
    try:
        from stable_baselines3.common.logger import configure

        logger = configure(f"{out_dir}/logs", ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)
    except Exception:
        pass

    eval_cb = WandbEvalCallback(
        eval_env,
        best_model_save_path=f"{out_dir}/best",
        log_path=f"{out_dir}/eval",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        wandb_run=wandb_run,
        vecnorm_env=train_env,
    )
    # Optional periodic checkpointing
    ckpt_cfg = algo_cfg.get("checkpoint", {})
    callback = eval_cb
    try:
        if bool(ckpt_cfg.get("enabled", False)):
            ckpt_cb = CheckpointCallbackWithVecnorm(
                save_freq_steps=int(ckpt_cfg.get("every_steps", 50_000)),
                save_dir=f"{out_dir}/checkpoints",
                vecnorm_env=train_env,
                keep_last_k=int(ckpt_cfg.get("keep_last_k", 3)),
                prefix=str(ckpt_cfg.get("prefix", "ckpt")),
                to_wandb=bool(ckpt_cfg.get("to_wandb", False)),
                wandb_run=wandb_run,
            )
            from stable_baselines3.common.callbacks import CallbackList

            callback = CallbackList([eval_cb, ckpt_cb])
    except Exception:
        pass

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(f"{out_dir}/final_model")
    # Save VecNormalize statistics if present
    try:
        from stable_baselines3.common.vec_env import VecNormalize as _VN

        if isinstance(train_env, _VN):
            train_env.save(f"{out_dir}/vecnorm.pkl")
    except Exception:
        pass

    # Save artifacts to WandB
    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            art = wandb.Artifact("models", type="model")
            art.add_file(f"{out_dir}/final_model.zip")
            # Optional best model and vecnorm
            import os

            if os.path.exists(f"{out_dir}/best/best_model.zip"):
                art.add_file(f"{out_dir}/best/best_model.zip")
            if os.path.exists(f"{out_dir}/vecnorm.pkl"):
                art.add_file(f"{out_dir}/vecnorm.pkl")
            if os.path.exists(f"{out_dir}/best/vecnorm_best.pkl"):
                art.add_file(f"{out_dir}/best/vecnorm_best.pkl")
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
