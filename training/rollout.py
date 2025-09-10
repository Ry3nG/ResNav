"""Rollout and video export utilities with optional rendering/recording."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import DictFrameStackVec
from visualization.pygame_renderer import Renderer, VizConfig
from visualization.video import save_mp4
from control.pure_pursuit import compute_u_track
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO


def load_yaml(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def detect_run_dir_from_model(model_path: str) -> str:
    """
    Detect run directory from model path.
    """
    p = Path(model_path).resolve()
    if p.is_dir():
        if p.name in ("best", "final"):
            return str(p.parent)
        if p.parent.name == "checkpoints":
            return str(p.parent.parent)
        return str(p)
    if p.name == "best_model.zip" and p.parent.name == "best":
        return str(p.parent.parent)
    if p.name == "final_model.zip":
        return str(p.parent)
    if p.name == "model.zip" and p.parent.parent.name == "checkpoints":
        return str(p.parent.parent.parent)
    return str(p.parent)


def load_run_config(run_dir: str) -> Dict[str, Any] | None:
    rd = Path(run_dir)
    resolved = rd / "resolved.yaml"
    hydra_cfg = rd / ".hydra" / "config.yaml"
    try:
        from typing import Any as _Any
        from omegaconf import OmegaConf as _OC

        if resolved.exists():
            cfg = _OC.to_container(_OC.load(str(resolved)), resolve=True)
            return cfg if isinstance(cfg, dict) else None
        if hydra_cfg.exists():
            cfg = _OC.to_container(_OC.load(str(hydra_cfg)), resolve=True)
            return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None
    return None


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
        run_dir = detect_run_dir_from_model(str(p))
        return (str(model_zip), str(vecnorm_pkl) if vecnorm_pkl else None, run_dir)
    raise SystemExit(f"[ERR] Unsupported path: {path}")


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
    parser.add_argument("--agent", choices=["ppo", "pp", "dwa"], default="ppo")

    args = parser.parse_args()

    if args.agent == "ppo" and args.model:
        # Allow passing a directory containing the model and vecnorm
        model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(args.model)
        # Load run config if available
        cfg = load_run_config(run_dir)
        env_cfg: Dict[str, Any] = (
            cfg["env"]
            if isinstance(cfg, dict) and "env" in cfg
            else load_yaml(args.env_cfg)
        )
        robot_cfg: Dict[str, Any] = (
            cfg["robot"]
            if isinstance(cfg, dict) and "robot" in cfg
            else load_yaml(args.robot_cfg)
        )
        reward_cfg: Dict[str, Any] = (
            cfg["reward"]
            if isinstance(cfg, dict) and "reward" in cfg
            else load_yaml(args.reward_cfg)
        )
        run_cfg: Dict[str, Any] = (
            cfg["run"]
            if isinstance(cfg, dict) and "run" in cfg
            else load_yaml(args.run_cfg)
        )
        # Prepare resolved paths for later loading
        args.model = model_zip
        auto_vecnorm = vecnorm_pkl or ""
    else:
        env_cfg = load_yaml(args.env_cfg)
        robot_cfg = load_yaml(args.robot_cfg)
        reward_cfg = load_yaml(args.reward_cfg)
        run_cfg = load_yaml(args.run_cfg)
        auto_vecnorm = ""

    # Build env factory so we can wrap with VecNormalize when needed
    def make_env():
        e = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
        return e

    base_env = None
    obs = None
    if args.agent == "ppo" and (args.model or args.vecnorm):
        venv = DummyVecEnv([make_env])
        # Apply the same frame-stack wrapper as training
        fs_cfg = env_cfg.get("wrappers", {}).get("frame_stack", {})
        k = int(fs_cfg.get("k", 4))
        flatten = bool(fs_cfg.get("flatten", True))
        if k > 1:
            venv = DictFrameStackVec(
                venv, keys=["lidar"], k=k, flatten=flatten, latest_first=True
            )
        # Load VecNormalize stats if provided or auto-detected
        vecnorm_path = args.vecnorm or auto_vecnorm
        if vecnorm_path:
            print(f"[INFO] Loading VecNormalize stats: {vecnorm_path}")
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = False
            venv.norm_reward = False
        else:
            print(
                "[WARN] No VecNormalize stats found; using raw observations for playback"
            )
        obs = venv.reset()
        base_env = venv.envs[0]
        if args.model:
            model = PPO.load(args.model, env=venv, print_system_info=False)
    else:
        base_env = make_env()
        obs, _ = base_env.reset(seed=args.seed)

    frames = []
    if args.render or args.record:
        map_size = env_cfg["map"]["size_m"]
        viz = env_cfg.get("viz", {})
        vcfg = VizConfig(
            size_px=(800, 800),
            show_inflated=bool(viz.get("show_inflated", True)),
            show_lidar=bool(viz.get("show_lidar", True)),
            show_actions=bool(viz.get("show_actions", True)),
            fps=int(viz.get("fps", 20)),
        )
        renderer = Renderer(
            map_size, base_env.resolution_m, viz_cfg=vcfg, display=bool(args.render)
        )

    # Rollout
    for t in range(args.steps):
        if args.agent == "ppo" and (args.model or args.vecnorm):
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, done, info = venv.step(action)
            # VecEnv API (Gymnasium compatibility in our wrappers): done is array-like
            if np.any(done):
                break
        else:
            u_track = compute_u_track(base_env.robot_state, base_env.path_preview)
            action = np.array(u_track, dtype=np.float32)
            obs, reward, terminated, truncated, info = base_env.step(action)
            if terminated or truncated:
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
                    lines = {"R_total": float(rt.get("total", 0.0))}
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
                    lidar_cfg.get("beams", 24),
                    lidar_cfg.get("fov_rad", np.radians(240)),
                    lidar_cfg.get("max_range", 4.0)
                )
            
            # Extract action data for rendering
            actions_data = None
            last_u = payload.get("last_u", np.array([0.0, 0.0]))
            prev_u = payload.get("prev_u", np.array([0.0, 0.0]))
            if len(last_u) >= 2 and len(prev_u) >= 2:
                # Compute u_track for reference
                from control.pure_pursuit import compute_u_track
                pose = payload["pose"]
                waypoints = payload.get("waypoints", np.array([]))
                if len(waypoints) > 0:
                    # Use robot config for lookahead and speed parameters
                    controller_cfg = robot_cfg.get("controller", {})
                    lookahead_m = controller_cfg.get("lookahead_m", 1.0)
                    v_nominal = controller_cfg.get("speed_nominal", 0.5)
                    u_track = compute_u_track(pose, waypoints, lookahead_m, v_nominal)
                    du = last_u - np.array(u_track)
                    actions_data = (tuple(u_track), tuple(du), tuple(last_u))
            
            frame = renderer.render_frame(
                raw_grid=payload["raw_grid"],
                inflated_grid=payload.get("inflated_grid"),
                pose=payload["pose"], 
                radius_m=payload["radius_m"],
                lidar=lidar_data,
                path=payload.get("waypoints"),
                actions=actions_data,
                hud=reward_breakdown
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
        fps_out = int(env_cfg.get("viz", {}).get("fps", 20))
        save_mp4(frames, str(out), fps=fps_out)
        print(f"Saved video with {len(frames)} frames to: {out}")


if __name__ == "__main__":
    main()
