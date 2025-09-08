"""Rollout and video export utilities with optional rendering/recording."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", default="configs/env/blockage.yaml")
    parser.add_argument("--robot_cfg", default="configs/robot/default.yaml")
    parser.add_argument("--reward_cfg", default="configs/reward/default.yaml")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--vecnorm", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--agent", choices=["ppo", "pp", "dwa"], default="ppo")
    parser.add_argument("--dwa_cfg", default="configs/control/dwa.yaml")
    args = parser.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    robot_cfg = load_yaml(args.robot_cfg)
    reward_cfg = load_yaml(args.reward_cfg)
    run_cfg = {"dt": args.dt, "max_steps": args.steps}

    # Build env factory so we can wrap with VecNormalize when needed
    def make_env():
        e = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
        e.seed(args.seed)
        return e

    model = None
    venv = None
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
        # Load VecNormalize stats if provided
        if args.vecnorm:
            venv = VecNormalize.load(args.vecnorm, venv)
            venv.training = False
            venv.norm_reward = False
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
        if args.agent == "ppo" and model is not None:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = venv.step(action)
            reward = float(rewards[0])
            term = bool(dones[0])
            trunc = False
            info = infos[0]
        else:
            # Baselines compute residual directly
            if args.agent == "pp":
                residual = np.array([0.0, 0.0], dtype=np.float32)
            elif args.agent == "dwa":
                from control.dwa_baseline import dwa_select_action

                payload = base_env.get_render_payload()
                x, y, th = payload["pose"]
                waypoints = payload["waypoints"]
                grid_infl = payload["inflated_grid"]
                v_track, w_track = compute_u_track(
                    (x, y, th),
                    waypoints,
                    robot_cfg.get("controller", {}).get("lookahead_m", 1.2),
                    robot_cfg.get("controller", {}).get("speed_nominal", 1.0),
                )
                # Load DWA config
                dwa_cfg = load_yaml(args.dwa_cfg)
                wcfg = dwa_cfg.get("weights", {})
                lcfg = dwa_cfg.get("lattice", {})
                horizon_s = float(dwa_cfg.get("horizon_s", 2.0))
                v_dwa, w_dwa = dwa_select_action(
                    (x, y, th),
                    waypoints,
                    grid_infl,
                    base_env.resolution_m,
                    base_env.v_max,
                    base_env.w_max,
                    dt=base_env.dt,
                    horizon_s=horizon_s,
                    v_samples=int(lcfg.get("v_samples", 15)),
                    w_samples=int(lcfg.get("w_samples", 15)),
                    w_progress=float(wcfg.get("progress", 1.0)),
                    w_path=float(wcfg.get("path", 1.0)),
                    w_heading=float(wcfg.get("heading", 0.1)),
                    w_obst=float(wcfg.get("obstacle", 0.5)),
                    w_smooth=float(wcfg.get("smooth", 0.05)),
                )
                residual = np.array([v_dwa - v_track, w_dwa - w_track], dtype=np.float32)
            else:
                residual = np.array([0.0, 0.0], dtype=np.float32)
            obs, reward, term, trunc, info = base_env.step(residual)
        # Build a unified single-env obs view for rendering
        if args.agent == "ppo" and model is not None:
            obs_view = {
                k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in obs.items()
            }
        else:
            obs_view = obs
        if args.render or args.record:
            payload = base_env.get_render_payload()
            x, y, th = payload["pose"]
            # Safe get controller params
            ctrl = robot_cfg.get("controller", {})
            lookahead_m = float(ctrl.get("lookahead_m", 1.2))
            speed_nominal = float(ctrl.get("speed_nominal", 1.0))
            # Recompute u_track for visualization (pure function)
            v_track, w_track = compute_u_track(
                payload["pose"], payload["waypoints"], lookahead_m, speed_nominal
            )
            u_final = payload["last_u"]
            du = (u_final[0] - v_track, u_final[1] - w_track)
            # First preview point in robot frame -> world
            path_vec = payload["obs"]["path"]
            xr, yr = float(path_vec[2]), float(path_vec[3])
            look_x = x + np.cos(th) * xr - np.sin(th) * yr
            look_y = y + np.sin(th) * xr + np.cos(th) * yr
            # Render
            frame = renderer.render_frame(
                raw_grid=payload["raw_grid"],
                inflated_grid=payload["inflated_grid"],
                pose=(x, y, th),
                radius_m=payload["radius_m"],
                lidar=(
                    payload["obs"]["lidar"],
                    payload["lidar"]["beams"],
                    payload["lidar"]["fov_rad"],
                    payload["lidar"]["max_range"],
                ),
                path=payload["waypoints"],
                proj=None,
                lookahead=(look_x, look_y),
                actions=((v_track, w_track), du, u_final),
                hud={"reward": reward},
            )
            if args.record:
                # Convert to array immediately to avoid accumulating Surfaces
                try:
                    import pygame

                    arr = pygame.surfarray.array3d(frame).transpose(1, 0, 2)
                except Exception:
                    arr = None
                if arr is not None:
                    frames.append(arr)
        # Handle window events to avoid freezing
        if args.render:
            try:
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        term = True
                        trunc = True
                        break
            except Exception:
                pass
        if term or trunc:
            break

    if args.record and len(frames) > 0:
        fps_out = int(env_cfg.get("viz", {}).get("fps", 20))
        save_mp4(frames, args.record, fps=fps_out)

    # Cleanup
    try:
        if args.render or args.record:
            renderer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
