from __future__ import annotations

import argparse
import os
import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from me5418_nav.envs import UnicycleNavEnv


def make_env(env_cfg: Dict[str, Any], scenario: str, scenario_kwargs: Dict[str, Any]):
    def _fn():
        # Pass full env config dict so nested sections (reward/lidar/preview) are honored
        merged = dict(env_cfg)
        merged.setdefault("scenario", scenario)
        merged.setdefault("scenario_kwargs", scenario_kwargs)
        merged.setdefault("render_mode", None)
        env = UnicycleNavEnv(merged)
        # Pack scenario options into env.reset options via Monitor's info_keywords
        env = Monitor(env)
        return env
    return _fn


def main():
    parser = argparse.ArgumentParser(description="Train PPO on AMR blockage scenario")
    parser.add_argument("--config", type=str, default="configs/ppo_default.yaml", help="Path to YAML config")
    parser.add_argument("--outdir", type=str, default="runs/ppo", help="Output directory for logs and models")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})

    # Paths
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    for d in ("checkpoints", "tb"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)

    # Save resolved config
    with open(os.path.join(outdir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)

    scenario = str(env_cfg.get("scenario", "blockage"))
    scenario_kwargs = dict(env_cfg.get("scenario_kwargs", {}))
    env_maker = make_env(env_cfg, scenario, scenario_kwargs)
    eval_env_maker = make_env(env_cfg, scenario, scenario_kwargs)

    vec_env = DummyVecEnv([env_maker])
    eval_vec_env = DummyVecEnv([eval_env_maker])

    # Device
    device = args.device or str(train_cfg.get("device", "cpu"))

    # Policy kwargs minimal; rely on defaults
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        tensorboard_log=os.path.join(outdir, "tb"),
        verbose=1,
        device=device,
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        seed=seed,
    )

    # Callbacks: evaluation + periodic checkpoints
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=outdir,
        log_path=outdir,
        eval_freq=int(train_cfg.get("eval_freq", 5000)),
        n_eval_episodes=int(train_cfg.get("eval_episodes", 5)),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=int(train_cfg.get("checkpoint_freq", 20000)),
        save_path=os.path.join(outdir, "checkpoints"),
        name_prefix="ppo_checkpoint",
    )

    total_timesteps = int(train_cfg.get("total_timesteps", 100000))
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save final model
    final_path = os.path.join(outdir, "final_model")
    model.save(final_path)

    # Emit summary file with paths
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({
            "best_model": os.path.join(outdir, "best_model.zip"),
            "final_model": final_path + ".zip",
            "tensorboard": os.path.join(outdir, "tb"),
        }, f, indent=2)

    print("[INFO] Training complete.")
    print("[INFO] Best model:", os.path.join(outdir, "best_model.zip"))
    print("[INFO] Final model:", final_path + ".zip")


if __name__ == "__main__":
    main()
