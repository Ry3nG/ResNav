"""Benchmark runner for Phase I.

Evaluates an agent (ppo, pp, or dwa) over N randomized scenarios and writes
CSV with per-episode metrics plus prints an aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, Tuple

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.path_utils import compute_path_context
from control.pure_pursuit import compute_u_track
from control.dwa_baseline import dwa_select_action
from omegaconf import OmegaConf


def load_yaml(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def make_env(env_cfg: Dict[str, Any], robot_cfg: Dict[str, Any], reward_cfg: Dict[str, Any], run_cfg: Dict[str, Any]) -> ResidualNavEnv:
    e = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    return e


def run_episode_pp(env: ResidualNavEnv, seed: int) -> Tuple[bool, float, int, int]:
    obs, _ = env.reset(seed=seed)
    total_steps = 0
    collisions = 0
    while True:
        action = np.array([0.0, 0.0], dtype=np.float32)  # residual zero
        obs, reward, term, trunc, info = env.step(action)
        total_steps += 1
        if term or trunc:
            success = bool(info.get("is_success", False))
            if term and not success:
                collisions += 1
            return success, float(total_steps * env.dt), collisions, int(trunc)


def run_episode_dwa(env: ResidualNavEnv, seed: int, dwa_cfg: Dict[str, Any]) -> Tuple[bool, float, int, int]:
    obs, _ = env.reset(seed=seed)
    total_steps = 0
    collisions = 0
    while True:
        # Use env internals via payload for map and waypoints (render-safe method)
        payload = env.get_render_payload()
        x, y, th = payload["pose"]
        grid_infl = payload["inflated_grid"]
        waypoints = payload["waypoints"]
        v_max = env.v_max
        w_max = env.w_max
        wcfg = dwa_cfg.get("weights", {})
        lcfg = dwa_cfg.get("lattice", {})
        horizon_s = float(dwa_cfg.get("horizon_s", 2.0))
        u_dwa = dwa_select_action(
            (x, y, th), waypoints, grid_infl, env.resolution_m, v_max, w_max,
            dt=env.dt,
            horizon_s=horizon_s,
            v_samples=int(lcfg.get("v_samples", 15)),
            w_samples=int(lcfg.get("w_samples", 15)),
            w_progress=float(wcfg.get("progress", 1.0)),
            w_path=float(wcfg.get("path", 1.0)),
            w_heading=float(wcfg.get("heading", 0.1)),
            w_obst=float(wcfg.get("obstacle", 0.5)),
            w_smooth=float(wcfg.get("smooth", 0.05)),
        )
        # Convert to residual relative to tracker
        v_track, w_track = compute_u_track((x, y, th), waypoints, env.robot_cfg.get("controller", {}).get("lookahead_m", 1.2), env.robot_cfg.get("controller", {}).get("speed_nominal", 1.0))
        residual = np.array([u_dwa[0] - v_track, u_dwa[1] - w_track], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(residual)
        total_steps += 1
        if term or trunc:
            success = bool(info.get("is_success", False))
            if term and not success:
                collisions += 1
            return success, float(total_steps * env.dt), collisions, int(trunc)


def run_episode_ppo(env_cfg: Dict[str, Any], robot_cfg: Dict[str, Any], reward_cfg: Dict[str, Any], run_cfg: Dict[str, Any], model_path: str, vecnorm_path: str, seed: int) -> Tuple[bool, float, int, int]:
    def _make():
        e = make_env(env_cfg, robot_cfg, reward_cfg, run_cfg)
        e.seed(seed)
        return e

    venv = DummyVecEnv([_make])
    # Apply frame stack like training if configured
    try:
        from amr_env.gym.wrappers import DictFrameStackVec

        fs_cfg = env_cfg.get("wrappers", {}).get("frame_stack", {})
        k = int(fs_cfg.get("k", 4))
        flatten = bool(fs_cfg.get("flatten", True))
        if k > 1:
            venv = DictFrameStackVec(venv, keys=["lidar"], k=k, flatten=flatten, latest_first=True)
    except Exception:
        pass

    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, env=venv, print_system_info=False)
    obs = venv.reset()
    total_steps = 0
    collisions = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        total_steps += 1
        if dones[0]:
            info = infos[0]
            success = bool(info.get("is_success", False))
            # We do not know collision vs timeout from vecenv: infer from success flag and done (no trunc flag). Treat non-success as collision
            collisions += int(not success)
            return success, float(total_steps * run_cfg.get("dt", 0.1)), collisions, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["pp", "dwa", "ppo"], default="pp")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env_cfg", default="configs/env/blockage.yaml")
    parser.add_argument("--robot_cfg", default="configs/robot/default.yaml")
    parser.add_argument("--reward_cfg", default="configs/reward/default.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--vecnorm", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default="runs/benchmark.csv")
    parser.add_argument("--dwa_cfg", default="configs/control/dwa.yaml")
    args = parser.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    robot_cfg = load_yaml(args.robot_cfg)
    reward_cfg = load_yaml(args.reward_cfg)
    run_cfg = {"dt": env_cfg.get("run", {}).get("dt", 0.1), "max_steps": 600}

    rows = []
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 1_000_000))
        if args.agent == "pp":
            env = make_env(env_cfg, robot_cfg, reward_cfg, run_cfg)
            success, time_s, collisions, deadlock = run_episode_pp(env, ep_seed)
        elif args.agent == "dwa":
            env = make_env(env_cfg, robot_cfg, reward_cfg, run_cfg)
            success, time_s, collisions, deadlock = run_episode_dwa(env, ep_seed)
        else:
            assert args.model, "--model required for PPO agent"
            success, time_s, collisions, deadlock = run_episode_ppo(env_cfg, robot_cfg, reward_cfg, run_cfg, args.model, args.vecnorm, ep_seed)
        rows.append({
            "episode": ep,
            "success": int(success),
            "time_s": time_s,
            "collisions": collisions,
            "deadlock": deadlock,
            "seed": ep_seed,
        })

    # Aggregate
    succ = np.mean([r["success"] for r in rows])
    tmean = np.mean([r["time_s"] for r in rows if r["success"] == 1]) if any(r["success"] for r in rows) else float("nan")
    coll = np.mean([r["collisions"] for r in rows])
    dead = np.mean([r["deadlock"] for r in rows])
    print("Summary:")
    print(f"  Success rate: {succ*100:.1f}%")
    if np.isfinite(tmean):
        print(f"  Avg time (success only): {tmean:.2f} s")
    print(f"  Collision rate: {coll*100:.1f}%")
    print(f"  Deadlock rate: {dead*100:.1f}%")

    # Write CSV
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "success", "time_s", "collisions", "deadlock", "seed"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
