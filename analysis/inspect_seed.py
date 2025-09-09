#!/usr/bin/env python3
"""Inspect a trained PPO model on a specific seed without rendering.

Prints per-step reward breakdown and simple stall diagnostics.

Usage:
  python analysis/inspect_seed.py \
    --model runs/20250909_102324/best/best_model.zip \
    --vecnorm runs/20250909_102324/vecnorm.pkl \
    --seed 7355608
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO


def load_yaml(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", default="configs/env/blockage.yaml")
    ap.add_argument("--robot_cfg", default="configs/robot/default.yaml")
    ap.add_argument("--reward_cfg", default="configs/reward/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", default="")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    robot_cfg = load_yaml(args.robot_cfg)
    reward_cfg = load_yaml(args.reward_cfg)
    run_cfg = {"dt": env_cfg.get("run", {}).get("dt", 0.1), "max_steps": args.steps}

    # Factory to mirror training wrappers
    from amr_env.gym.residual_nav_env import ResidualNavEnv

    def make_env():
        e = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
        e.seed(args.seed)
        return e

    venv = DummyVecEnv([make_env])
    # Frame stack wrapper (if enabled)
    try:
        from amr_env.gym.wrappers import DictFrameStackVec

        fs_cfg = env_cfg.get("wrappers", {}).get("frame_stack", {})
        k = int(fs_cfg.get("k", 4))
        flatten = bool(fs_cfg.get("flatten", True))
        if k > 1:
            venv = DictFrameStackVec(venv, keys=["lidar"], k=k, flatten=flatten, latest_first=True)
    except Exception:
        pass

    if args.vecnorm:
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, env=venv, print_system_info=False)
    obs = venv.reset()
    base_env = venv.envs[0]

    print("# Inspecting model on seed:", args.seed)
    print("# reward schema:", getattr(base_env, "_last_reward_terms", {}).get("version", "unknown"))
    print("step, R_total, R_progress, R_path, R_effort, speed, dv_cancel, dw_cancel")

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)

        payload = base_env.get_render_payload()
        terms = payload.get("reward_terms", {})
        contrib = terms.get("contrib", {})
        raw = terms.get("raw", {})
        total = float(terms.get("total", float(rewards[0])))

        # Cancellation diagnostic: compare last_u vs pure pursuit u_track
        from control.pure_pursuit import compute_u_track

        x, y, th = payload["pose"]
        v_track, w_track = compute_u_track((x, y, th), payload["waypoints"], robot_cfg.get("controller", {}).get("lookahead_m", 1.2), robot_cfg.get("controller", {}).get("speed_nominal", 1.0))
        v_cmd, w_cmd = payload["last_u"]
        dv_cancel = float(v_cmd - v_track)
        dw_cancel = float(w_cmd - w_track)

        speed = float(v_cmd)
        print(
            f"{t:04d}, {total:+.3f}, "
            f"{float(contrib.get('progress', 0.0)):+.3f}, "
            f"{float(contrib.get('path', 0.0)):+.3f}, "
            f"{float(contrib.get('effort', 0.0)):+.3f}, "
            f"{speed:.3f}, {dv_cancel:+.3f}, {dw_cancel:+.3f}"
        )

        if dones[0]:
            info = infos[0]
            print("# done. is_success:", bool(info.get("is_success", False)))
            break


if __name__ == "__main__":
    main()

