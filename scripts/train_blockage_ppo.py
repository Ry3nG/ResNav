#!/usr/bin/env python3
"""
Train PPO on the static blockage scenario using the BlockageRLWrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from me5418_nav.envs.unicycle_nav_env import UnicycleNavEnv, EnvConfig
from me5418_nav.envs.rl_wrappers import BlockageRLWrapper, RewardConfig
from me5418_nav.maps import BlockageScenarioConfig
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S


def make_env(seed: int, scen_cfg: BlockageScenarioConfig, rew_cfg: RewardConfig):
    def _thunk():
        # 10x10m map at 0.05m -> 200x200 cells
        cfg = EnvConfig(dt=DT_S, map_size=(200, 200), res=GRID_RESOLUTION_M)
        env = UnicycleNavEnv(cfg=cfg, render_mode=None)
        env = BlockageRLWrapper(env, scenario_cfg=scen_cfg, reward_cfg=rew_cfg, seed=seed)
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


def main():
    parser = argparse.ArgumentParser(description="Train PPO on blockage maps")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="logs/ppo_blockage")
    # Scenario knobs
    parser.add_argument("--num-pallets", type=str, default="0,5")
    parser.add_argument("--pallet-width", type=str, default="0.5,1.1")
    parser.add_argument("--pallet-length", type=str, default="0.3,0.6")

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="me5418-blockage-ppo",
        config={
            "timesteps": args.timesteps,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_pallets": args.num_pallets,
            "pallet_width": args.pallet_width,
            "pallet_length": args.pallet_length,
        }
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
    env_fns = [make_env(int(rng.integers(0, 2**31 - 1)), scen_cfg, rew_cfg) for _ in range(args.num_envs)]
    vec = DummyVecEnv(env_fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        verbose=1,
        seed=args.seed,
        n_steps=2048 // max(1, args.num_envs),
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    # Create wandb callback
    wandb_callback = WandbCallback()
    
    model.learn(total_timesteps=args.timesteps, callback=wandb_callback)

    # Save
    outdir = Path(args.logdir) / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save(str(outdir / "ppo_blockage"))
    vec.save(str(outdir / "vecnormalize.pkl"))
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
