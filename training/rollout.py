"""Rollout and video export utilities with optional rendering/recording."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import os
from pathlib import Path

from amr_env.utils import (
    detect_run_root,
    load_config_dict,
    load_resolved_run_config,
)

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec
from visualization.pygame_renderer import Renderer, VizConfig
from visualization.video import save_mp4
from control.pure_pursuit import compute_u_track
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
def resolve_model_and_vecnorm(path: str) -> tuple[str, str | None, str]:
    """Given a directory or zip path, resolve (model_zip, vecnorm_pkl|None, run_dir).

    Directory cases:
      - best/: best_model.zip + vecnorm_best.pkl
      - final/: final_model.zip + vecnorm_final.pkl (fallback vecnorm.pkl)
      - checkpoints/ckpt_step_X/: model.zip + vecnorm.pkl
    Zip file cases are resolved similarly based on filename.
    """
    p = Path(path)
    if p.is_file():
        p = p.parent
    if not p.exists():
        raise SystemExit(f"[ERR] Path does not exist: {path}")
    model_zip: Path | None = None
    vecnorm_pkl: Path | None = None

    if p.is_dir():
        # checkpoint dir
        if (p / "model.zip").exists():
            model_zip = p / "model.zip"
            if (p / "vecnorm.pkl").exists():
                vecnorm_pkl = p / "vecnorm.pkl"
        # best dir
        elif (p / "best_model.zip").exists():
            model_zip = p / "best_model.zip"
            if (p / "vecnorm_best.pkl").exists():
                vecnorm_pkl = p / "vecnorm_best.pkl"
        # final dir
        elif (p / "final_model.zip").exists():
            model_zip = p / "final_model.zip"
            if (p / "vecnorm_final.pkl").exists():
                vecnorm_pkl = p / "vecnorm_final.pkl"
            elif (p / "vecnorm.pkl").exists():
                vecnorm_pkl = p / "vecnorm.pkl"
        else:
            raise SystemExit(
                f"[ERR] No model zip found in directory: {p}\nSuggest one of: best/, final/, checkpoints/ckpt_step_N/"
            )
    else:
        raise SystemExit(f"[ERR] Unsupported path: {path}")

    run_dir = detect_run_root(str(p))
    if model_zip is None:
        raise SystemExit(f"[ERR] Failed to locate model zip under: {p}")

    return (str(model_zip), str(vecnorm_pkl) if vecnorm_pkl else None, run_dir)


def detect_algo_from_run(run_dir: str) -> str:
    """Detect algorithm used for a given run directory.

    Priority:
      1) resolved.yaml (full config) if present
      2) .hydra/overrides.yaml (look for a '- algo=...' line)
      3) Heuristic on algo block keys (buffer_size→sac; n_steps→ppo)
      4) Default 'ppo'
    """
    # 1) Try resolved or hydra config
    cfg = load_resolved_run_config(run_dir)
    try:
        if isinstance(cfg, dict) and "algo" in cfg and isinstance(cfg["algo"], dict):
            algo_block = cfg["algo"]
            # Heuristic: SAC has buffer_size/tau; PPO has n_steps
            if any(k in algo_block for k in ("buffer_size", "tau", "learning_starts")):
                return "sac"
            if any(k in algo_block for k in ("n_steps", "gae_lambda", "clip_range")):
                return "ppo"
    except Exception:
        pass

    # 2) Check overrides
    try:
        from pathlib import Path as _P
        ovr = _P(run_dir) / ".hydra" / "overrides.yaml"
        if ovr.exists():
            text = ovr.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                s = line.strip().lstrip("- ")
                if s.startswith("algo="):
                    return s.split("=", 1)[1].strip()
    except Exception:
        pass

    # 3) Default
    return "ppo"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model directory: best/ | final/ | checkpoints/ckpt_step_N/",
    )
    parser.add_argument("--record", type=str, default="")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_cfg", type=str, default="configs/env/blockage.yaml")
    parser.add_argument("--robot_cfg", type=str, default="configs/robot/default.yaml")
    parser.add_argument("--reward_cfg", type=str, default="configs/reward/default.yaml")
    parser.add_argument("--run_cfg", type=str, default="configs/run/default.yaml")
    parser.add_argument("--vecnorm", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    if args.model:
        # Allow passing a directory containing the model and vecnorm
        model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(args.model)
        # Load run config if available
        cfg = load_resolved_run_config(run_dir)
        env_cfg: Dict[str, Any] = (
            cfg["env"]
            if isinstance(cfg, dict) and "env" in cfg
            else load_config_dict(args.env_cfg)
        )
        robot_cfg: Dict[str, Any] = (
            cfg["robot"]
            if isinstance(cfg, dict) and "robot" in cfg
            else load_config_dict(args.robot_cfg)
        )
        reward_cfg: Dict[str, Any] = (
            cfg["reward"]
            if isinstance(cfg, dict) and "reward" in cfg
            else load_config_dict(args.reward_cfg)
        )
        run_cfg: Dict[str, Any] = (
            cfg["run"]
            if isinstance(cfg, dict) and "run" in cfg
            else load_config_dict(args.run_cfg)
        )
        # Prepare resolved paths for later loading
        args.model = model_zip
        auto_vecnorm = vecnorm_pkl or ""
    else:
        env_cfg = load_config_dict(args.env_cfg)
        robot_cfg = load_config_dict(args.robot_cfg)
        reward_cfg = load_config_dict(args.reward_cfg)
        run_cfg = load_config_dict(args.run_cfg)
        auto_vecnorm = ""

    # Build env factory so we can wrap with VecNormalize when needed
    def make_env():
        e = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
        return e

    venv = DummyVecEnv([make_env])
    # Apply the same frame-stack wrapper as training
    fs_cfg = env_cfg["wrappers"]["frame_stack"]
    k = int(fs_cfg["k"])
    if k > 1:
        venv = LidarFrameStackVec(venv, k=k)
    # Load VecNormalize stats if provided or auto-detected
    vecnorm_path = args.vecnorm or auto_vecnorm
    if vecnorm_path:
        print(f"[INFO] Loading VecNormalize stats: {vecnorm_path}")
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("[WARN] No VecNormalize stats found; using raw observations for playback")
    # Ensure deterministic environment setup when a seed is provided
    obs = venv.reset(seed=int(args.seed))
    base_env = venv.envs[0]
    if args.model:
        # Detect algorithm used for this run
        algo_name = detect_algo_from_run(run_dir)
        ModelClass = SAC if algo_name == "sac" else PPO
        print(f"[INFO] Detected algorithm: {algo_name.upper()} → using {ModelClass.__name__}")
        model = ModelClass.load(args.model, env=venv, print_system_info=False)

    frames = []
    if args.render or args.record:
        map_size = env_cfg["map"]["size_m"]
        viz = env_cfg["viz"]
        vcfg = VizConfig(
            size_px=(800, 800),
            show_inflated=bool(viz["show_inflated"]),
            show_lidar=bool(viz["show_lidar"]),
            show_actions=bool(viz["show_actions"]),
            fps=int(viz["fps"]),
        )
        renderer = Renderer(
            map_size, base_env.resolution_m, viz_cfg=vcfg, display=bool(args.render)
        )

    # Rollout
    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        obs, reward, done, info = venv.step(action)
        # VecEnv API (Gymnasium compatibility in our wrappers): done is array-like
        if np.any(done):
            break
        if args.render or args.record:
            # VecEnv returns a list of infos; unwrap for single-env DummyVecEnv
            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            # Get render payload from environment (includes reward_terms every step)
            payload = base_env.get_render_payload()
            # Prefer reward_terms from payload; fallback to info if present
            rt = {}
            if isinstance(payload, dict):
                rt = payload.get("reward_terms", {}) or {}
            if not rt and isinstance(info_dict, dict):
                rt = info_dict.get("reward_terms", {}) or {}
            # Format HUD lines: total + per-term contributions sorted by |value|
            reward_breakdown = None
            if isinstance(rt, dict) and rt:
                try:
                    lines = {"R_total": float(rt["total"])}
                    contrib = rt.get("contrib", {}) or {}
                    # sort by absolute contribution descending
                    for k, v in sorted(
                        contrib.items(), key=lambda kv: -abs(float(kv[1]))
                    ):
                        lines[f"R_{k}"] = float(v)
                    reward_breakdown = lines
                except Exception:
                    reward_breakdown = None

            # Extract lidar data for rendering
            obs_data = payload.get("obs", {})
            lidar_ranges = obs_data.get("lidar", np.array([]))
            lidar_cfg = payload.get("lidar", {})

            # Build lidar tuple if data available
            lidar_data = None
            if len(lidar_ranges) > 0 and lidar_cfg:
                lidar_data = (
                    lidar_ranges,
                    lidar_cfg["beams"],
                    lidar_cfg["fov_rad"],
                    lidar_cfg["max_range"],
                )

            # Extract action data for rendering
            actions_data = None
            last_u = payload["last_u"]
            prev_u = payload["prev_u"]
            if len(last_u) >= 2 and len(prev_u) >= 2:
                # Compute u_track for reference
                from control.pure_pursuit import compute_u_track

                pose = payload["pose"]
                waypoints = payload.get("waypoints", np.array([]))
                if len(waypoints) > 0:
                    # Use robot config for lookahead and speed parameters
                    controller_cfg = robot_cfg.get("controller", {})
                    lookahead_m = controller_cfg["lookahead_m"]
                    v_nominal = controller_cfg["speed_nominal"]
                    u_track = compute_u_track(pose, waypoints, lookahead_m, v_nominal)
                    last_u_arr = np.array(last_u, dtype=float)
                    u_track_arr = np.array(u_track, dtype=float)
                    du = last_u_arr - u_track_arr
                    actions_data = (
                        tuple(u_track_arr.tolist()),
                        tuple(du.tolist()),
                        tuple(last_u_arr.tolist()),
                    )

            frame = renderer.render_frame(
                raw_grid=payload["raw_grid"],
                inflated_grid=payload.get("inflated_grid"),
                pose=payload["pose"],
                radius_m=payload["radius_m"],
                lidar=lidar_data,
                path=payload.get("waypoints"),
                actions=actions_data,
                hud=reward_breakdown,
            )
            if args.record:
                # Convert pygame Surface to numpy array immediately
                try:
                    import pygame

                    arr = pygame.surfarray.array3d(frame).transpose(1, 0, 2)
                    frames.append(arr)
                except Exception as e:
                    print(f"Failed to convert frame to array: {e}")
            else:
                frames.append(frame)

    if args.record and len(frames) > 0:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps_out = int(env_cfg["viz"]["fps"])
        save_mp4(frames, str(out), fps=fps_out)
        print(f"Saved video with {len(frames)} frames to: {out}")


if __name__ == "__main__":
    main()
