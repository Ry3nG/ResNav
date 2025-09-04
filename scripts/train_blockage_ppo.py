#!/usr/bin/env python3
"""
Train PPO on the static blockage scenario using the BlockageRLWrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import random

import numpy as np
import gymnasium as gym
import torch
import wandb
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecCheckNan,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    ProgressBarCallback,
)

from me5418_nav.envs.unicycle_nav_env import UnicycleNavEnv, EnvConfig
from me5418_nav.envs.rl_wrappers import BlockageRLWrapper, RewardConfig
from me5418_nav.maps import BlockageScenarioConfig
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S


def make_env(seed: int, scen_cfg: BlockageScenarioConfig, rew_cfg: RewardConfig, monitor_file: str | None = None):
    def _thunk():
        # 10x10m map at 0.05m -> 200x200 cells
        cfg = EnvConfig(dt=DT_S, map_size=(200, 200), res=GRID_RESOLUTION_M)
        env = UnicycleNavEnv(cfg=cfg, render_mode=None)
        env = BlockageRLWrapper(env, scenario_cfg=scen_cfg, reward_cfg=rew_cfg, seed=seed)
        # Record episodic stats and key info fields
        env = Monitor(env, filename=monitor_file, info_keywords=(
            "collision", "success", "is_success", "scenario_min_clearance", "scenario_num_pallets", "scenario_difficulty"
        ))
        return env

    return _thunk


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            wandb.log(self.model.logger.name_to_value, step=self.num_timesteps)


class EpisodicStatsCallback(BaseCallback):
    """Aggregates episodic outcomes from VecMonitor infos and logs rates."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = int(window)
        self.hist = []  # list of dicts with keys success/collision/timeout

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos or []:
            # VecMonitor inserts 'episode' at episode end
            if isinstance(info, dict) and ("episode" in info):
                success = bool(info.get("is_success") or info.get("success", False))
                collision = bool(info.get("collision", False))
                timeout = not success and not collision
                self.hist.append({"success": success, "collision": collision, "timeout": timeout})
                if len(self.hist) > self.window:
                    self.hist.pop(0)
                # Log rolling rates
                n = max(1, len(self.hist))
                s = sum(1 for h in self.hist if h["success"])
                c = sum(1 for h in self.hist if h["collision"])
                t = sum(1 for h in self.hist if h["timeout"])
                self.logger.record("train/success_rate", s / n)
                self.logger.record("train/collision_rate", c / n)
                self.logger.record("train/timeout_rate", t / n)
        return True


def main():
    parser = argparse.ArgumentParser(description="Train PPO on blockage maps")
    # Run + reproducibility
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="logs/ppo_blockage")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging directory and W&B")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to PPO .zip to resume training")
    parser.add_argument("--resume-vecnorm", type=str, default=None, help="Path to VecNormalize pickle when resuming")
    # Scenario knobs
    parser.add_argument("--num-pallets", type=str, default="0,5")
    parser.add_argument("--pallet-width", type=str, default="0.5,1.1")
    parser.add_argument("--pallet-length", type=str, default="0.3,0.6")
    # PPO hyperparameters / schedules
    parser.add_argument("--n-steps", type=int, default=2048, help="Total rollout steps per update (across envs)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-range-vf", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--sde-sample-freq", type=int, default=4)
    parser.add_argument("--policy-arch", type=str, default="64,64", help="Comma list of hidden sizes, e.g. 64,64")
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "linear"], help="Learning rate schedule")
    parser.add_argument("--clip-schedule", type=str, default="constant", choices=["constant", "linear"], help="Clip range schedule")
    # Eval/checkpoint options
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Timesteps between evals (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Timesteps between checkpoints (0 to disable)")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Eval windows w/o improvement before stop")
    # Normalization options
    parser.add_argument("--norm-obs", action="store_true")
    parser.add_argument("--norm-reward", action="store_true")
    parser.add_argument("--clip-obs", type=float, default=10.0)
    # W&B options
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="me5418-blockage-ppo")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, help="Set to 'offline' to disable network")

    args = parser.parse_args()

    # Prepare run directory
    run_name = args.run_name or f"seed_{args.seed}"
    outdir = Path(args.logdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    monitor_dir = outdir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    # Seeding for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        torch.cuda.manual_seed_all(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize wandb (optional)
    if not args.no_wandb:
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            sync_tensorboard=True,
            config={
                "timesteps": args.timesteps,
                "seed": args.seed,
                "device": args.device,
                "num_envs": args.num_envs,
                "num_pallets": args.num_pallets,
                "pallet_width": args.pallet_width,
                "pallet_length": args.pallet_length,
            },
        )

    # Scenario config
    num_pallets_range = tuple(map(int, args.num_pallets.split(",")))
    pallet_width_range = tuple(map(float, args.pallet_width.split(",")))
    pallet_length_range = tuple(map(float, args.pallet_length.split(",")))
    scen_cfg = BlockageScenarioConfig(
        num_pallets_range=num_pallets_range,
        pallet_width_range=pallet_width_range,
        pallet_length_range=pallet_length_range,
    )

    rew_cfg = RewardConfig()

    # Vectorized env
    rng = np.random.default_rng(args.seed)
    env_fns = []
    for i in range(args.num_envs):
        seed_i = int(rng.integers(0, 2**31 - 1))
        mon_file = str((monitor_dir / f"train_env_{i}.csv").as_posix())
        env_fns.append(make_env(seed_i, scen_cfg, rew_cfg, monitor_file=mon_file))

    if args.num_envs > 1:
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv(env_fns)
    # Per-env Monitor already provides episode stats; no need for VecMonitor
    if args.norm_obs or args.norm_reward:
        vec = VecNormalize(
            vec, norm_obs=bool(args.norm_obs), norm_reward=bool(args.norm_reward), clip_obs=args.clip_obs
        )
    vec = VecCheckNan(vec, raise_exception=True)
    vec.seed(args.seed)

    # Compute per-env n_steps and enforce divisibility by batch_size
    total_rollout = max(args.n_steps, args.num_envs)
    n_steps_per_env = max(1, total_rollout // max(1, args.num_envs))
    rollout_size = n_steps_per_env * args.num_envs
    batch_size = int(args.batch_size)
    if rollout_size % batch_size != 0:
        # Increase n_steps_per_env minimally to satisfy divisibility
        for k in range(n_steps_per_env, n_steps_per_env + args.num_envs + 1):
            if (k * args.num_envs) % batch_size == 0:
                n_steps_per_env = k
                rollout_size = k * args.num_envs
                break
        # As a fallback, adjust batch_size down to nearest divisor
        if rollout_size % batch_size != 0:
            from math import gcd

            g = gcd(rollout_size, batch_size)
            if g > 0:
                batch_size = g

    # Policy architecture
    try:
        hidden_sizes = tuple(int(x) for x in args.policy_arch.split(",") if x)
    except Exception:
        hidden_sizes = (64, 64)
    policy_kwargs = dict(net_arch=list(hidden_sizes))

    # Build optional schedules
    parser_lr_schedule = getattr(args, "lr_schedule", None)
    parser_clip_schedule = getattr(args, "clip_schedule", None)
    lr = args.learning_rate
    clip_range = args.clip_range
    if parser_lr_schedule == "linear":
        start_lr = float(args.learning_rate)

        def lr_schedule(progress_remaining: float) -> float:
            return start_lr * float(progress_remaining)

        lr = lr_schedule
    if parser_clip_schedule == "linear":
        start_clip = float(args.clip_range)

        def clip_schedule(progress_remaining: float) -> float:
            return start_clip * float(progress_remaining)

        clip_range = clip_schedule

    # Save run config for reproducibility
    try:
        with open(outdir / "config.yaml", "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=True)
    except Exception:
        pass

    # PPO (new or resume)
    if args.resume_from:
        # Optionally load VecNormalize stats
        if args.resume_vecnorm:
            try:
                vec = VecNormalize.load(args.resume_vecnorm, vec)
                vec.training = True
                vec.norm_reward = bool(args.norm_reward)
            except Exception as e:
                print(f"Warning: failed to load VecNormalize stats from {args.resume_vecnorm}: {e}")
        model = PPO.load(args.resume_from, env=vec, device=args.device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec,
            verbose=1,
            seed=args.seed,
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_epochs=args.n_epochs,
            learning_rate=lr,
            clip_range=clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            policy_kwargs=policy_kwargs,
            device=args.device,
            tensorboard_log=str((outdir / "tb").as_posix()),
        )

    # Callbacks: W&B logging, evaluation, checkpointing, progress bar
    callbacks = [EpisodicStatsCallback(window=100)]
    if not args.no_wandb:
        try:
            from wandb.integration.sb3 import WandbCallback as WandbSB3Callback

            callbacks.append(WandbSB3Callback(verbose=0, model_save_path=str(outdir / "wandb_ckpts"), gradient_save_freq=0))
        except Exception:
            callbacks.append(WandbCallback())

    # Helper: find a wrapper instance in a possibly stacked VecEnv
    def _find_wrapper(env, wrapper_type):
        from stable_baselines3.common.vec_env import VecEnvWrapper

        e = env
        while isinstance(e, VecEnvWrapper):
            if isinstance(e, wrapper_type):
                return e
            e = e.venv
        return None

    # Evaluation callback on a deterministic eval env
    if args.eval_freq and args.eval_freq > 0:
        eval_env_fn = make_env(int(rng.integers(0, 2**31 - 1)), scen_cfg, rew_cfg, monitor_file=str((monitor_dir / "eval_env.csv").as_posix()))
        eval_vec = DummyVecEnv([eval_env_fn])
        # If training env uses VecNormalize anywhere in its stack, wrap eval too and sync stats
        train_norm = _find_wrapper(vec, VecNormalize)
        if train_norm is not None:
            eval_vec = VecNormalize(eval_vec, training=False, norm_obs=train_norm.norm_obs, norm_reward=False)
            # copy normalization stats
            eval_norm = _find_wrapper(eval_vec, VecNormalize)
            if eval_norm is not None:
                eval_norm.obs_rms = train_norm.obs_rms
                eval_norm.ret_rms = train_norm.ret_rms
        # Keep wrapper stacks consistent with training env
        eval_vec = VecCheckNan(eval_vec, raise_exception=True)
        stop_cb = None
        if args.early_stop_patience and args.early_stop_patience > 0:
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=args.early_stop_patience,
                min_evals=1,
                verbose=1,
            )
        eval_cb = EvalCallback(
            eval_vec,
            best_model_save_path=str(outdir / "best_model"),
            log_path=str(outdir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            verbose=1,
        )
        callbacks.append(eval_cb)

    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq // max(1, args.num_envs), save_path=str(outdir / "checkpoints"), name_prefix="ppo_blockage")
        callbacks.append(ckpt_cb)

    callbacks.append(ProgressBarCallback())

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        # Save an interrupt checkpoint
        model.save(str(outdir / "ppo_blockage_interrupt"))
        raise
    finally:
        # Ensure envs are closed and W&B is finished on any exit path
        try:
            model.env.close()
        except Exception:
            pass
        try:
            vec.close()
        except Exception:
            pass
        if not args.no_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    # Save final artifacts
    model.save(str(outdir / "ppo_blockage"))
    if isinstance(vec, VecNormalize):
        vec.save(str(outdir / "vecnormalize.pkl"))
    # Note: envs and W&B finalized in finally block above


if __name__ == "__main__":
    main()
