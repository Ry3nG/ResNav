"""Rollout and video export utilities with optional rendering/recording."""

from __future__ import annotations

import argparse
import json
from typing import Any
import warnings

import numpy as np
import os
from pathlib import Path
import math
import pygame

# Suppress pygame pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)

from amr_env.utils import (
    detect_run_root,
    load_config_dict,
    load_resolved_run_config,
)

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec
from amr_env.viz.pygame_renderer import Renderer, VizConfig
from amr_env.viz.video import save_mp4
from amr_env.control.pure_pursuit import compute_u_track
from amr_env.control.apf import compute_apf_command
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC


def _ensure_parent(path: str | Path) -> Path:
    out_path = Path(path)
    parent = out_path.parent if out_path.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _save_trajectory_csv(path: str, data: np.ndarray) -> None:
    out = _ensure_parent(path)
    header = "t_sec,x_m,y_m,theta_rad,v_cmd_mps,w_cmd_rps,goal_dist_m"
    np.savetxt(out, data, delimiter=",", header=header, comments="")


def _save_metrics_json(path: str, metrics: dict) -> None:
    out = _ensure_parent(path)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def _accumulate_heat_grid(
    grid_shape: tuple[int, int],
    samples: list[tuple[float, float]] | np.ndarray,
    radius_m: float,
    resolution_m: float,
) -> np.ndarray:
    heat = np.zeros(grid_shape, dtype=float)
    if radius_m <= 0.0 or resolution_m <= 0.0:
        return heat
    if samples is None or len(samples) == 0:
        return heat
    rad_cells = max(1, int(math.ceil(radius_m / resolution_m)))
    H, W = grid_shape
    for sample in samples:
        if sample is None or len(sample) < 2:
            continue
        x, y = float(sample[0]), float(sample[1])
        jc = int(x / resolution_m)
        ic = int(y / resolution_m)
        if ic < 0 or ic >= H or jc < 0 or jc >= W:
            continue
        i0 = max(0, ic - rad_cells)
        i1 = min(H - 1, ic + rad_cells)
        j0 = max(0, jc - rad_cells)
        j1 = min(W - 1, jc + rad_cells)
        if i0 > i1 or j0 > j1:
            continue
        yy, xx = np.ogrid[i0 : i1 + 1, j0 : j1 + 1]
        mask = (xx - jc) ** 2 + (yy - ic) ** 2 <= rad_cells**2
        heat_region = heat[i0 : i1 + 1, j0 : j1 + 1]
        heat_region[mask] += 1.0
    return heat


def _compute_rollout_metrics(
    traj: np.ndarray, dt: float, success: bool
) -> dict[str, float | int | bool | None]:
    metrics: dict[str, float | int | bool | None] = {}
    if traj.size == 0:
        return metrics

    times = traj[:, 0]
    xs = traj[:, 1]
    ys = traj[:, 2]
    v_cmd = traj[:, 4]
    w_cmd = traj[:, 5]
    goal_dist = traj[:, 6]

    total_time = float(times[-1])
    if len(traj) > 1:
        xy = np.column_stack((xs, ys))
        segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        path_length = float(np.sum(segment_lengths))
    else:
        path_length = 0.0
    avg_speed = path_length / total_time if total_time > 1e-6 else 0.0
    lin_smooth = (
        float(np.mean(np.square(np.diff(v_cmd)))) if len(v_cmd) > 1 else 0.0
    )
    ang_smooth = (
        float(np.mean(np.square(np.diff(w_cmd)))) if len(w_cmd) > 1 else 0.0
    )

    metrics.update(
        {
            "samples": int(len(traj)),
            "time_elapsed_s": total_time,
            "path_length_m": path_length,
            "avg_speed_mps": avg_speed,
            "success": bool(success),
            "time_to_goal_s": total_time if success else None,
            "final_goal_dist_m": float(goal_dist[-1]),
            "min_goal_dist_m": float(np.min(goal_dist)),
            "linear_smoothness": lin_smooth,
            "angular_smoothness": ang_smooth,
            "smoothness_score": lin_smooth + ang_smooth,
        }
    )
    return metrics


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

    Since we only use SAC now, always returns 'sac'.
    Kept for backward compatibility with existing run directories.
    """
    return "sac"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model directory: best/ | final/ | checkpoints/ckpt_step_N/",
    )
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_cfg", type=str, default="configs/env/blockage.yaml")
    parser.add_argument("--robot_cfg", type=str, default="configs/robot/default.yaml")
    parser.add_argument("--reward_cfg", type=str, default="configs/reward/default.yaml")
    parser.add_argument("--run_cfg", type=str, default="")
    parser.add_argument("--vecnorm", type=str, default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--traj_out",
        type=str,
        default="",
        help="Optional CSV path to dump per-step pose/control data.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="",
        help="Optional JSON path to save rollout summary metrics.",
    )
    parser.add_argument(
        "--snapshot_out",
        type=str,
        default="",
        help="Optional path to save the final rendered frame (PNG).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="residual",
        choices=["residual", "apf", "pure_pursuit"],
        help="Controller to use for rollout. 'residual' uses the RL policy (default).",
    )

    args = parser.parse_args()
    log_rollout = bool(args.traj_out or args.metrics_out)

    controller_mode = str(args.controller).lower()

    if args.model:
        # Allow passing a directory containing the model and vecnorm
        model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(args.model)
        # Load run config if available
        cfg = load_resolved_run_config(run_dir)

        # Use command-line config if explicitly provided, else fallback to run config
        env_cfg: dict[str, Any] = (
            load_config_dict(args.env_cfg)
            if args.env_cfg
            != "configs/env/blockage.yaml"  # Non-default means user wants override
            else (
                cfg["env"]
                if isinstance(cfg, dict) and "env" in cfg
                else load_config_dict(args.env_cfg)
            )
        )
        robot_cfg: dict[str, Any] = (
            load_config_dict(args.robot_cfg)
            if args.robot_cfg != "configs/robot/default.yaml"
            else (
                cfg["robot"]
                if isinstance(cfg, dict) and "robot" in cfg
                else load_config_dict(args.robot_cfg)
            )
        )
        reward_cfg: dict[str, Any] = (
            load_config_dict(args.reward_cfg)
            if args.reward_cfg != "configs/reward/default.yaml"
            else (
                cfg["reward"]
                if isinstance(cfg, dict) and "reward" in cfg
                else load_config_dict(args.reward_cfg)
            )
        )
        run_cfg: dict[str, Any]
        if isinstance(cfg, dict) and "run" in cfg:
            run_cfg = cfg["run"]
        elif args.run_cfg:
            run_cfg = load_config_dict(args.run_cfg)
        else:
            try:
                base_cfg = load_config_dict("configs/config.yaml")
                run_cfg = base_cfg.get("run", {"dt": 0.1, "max_steps": 600})
            except Exception:
                run_cfg = {"dt": 0.1, "max_steps": 600}
        # Prepare resolved paths for later loading
        args.model = model_zip
        auto_vecnorm = vecnorm_pkl or ""
    else:
        env_cfg: dict[str, Any] = load_config_dict(args.env_cfg)
        robot_cfg: dict[str, Any] = load_config_dict(args.robot_cfg)
        reward_cfg: dict[str, Any] = load_config_dict(args.reward_cfg)
        if args.run_cfg:
            run_cfg = load_config_dict(args.run_cfg)
        else:
            try:
                base_cfg = load_config_dict("configs/config.yaml")
                run_cfg = base_cfg.get("run", {"dt": 0.1, "max_steps": 600})
            except Exception:
                run_cfg = {"dt": 0.1, "max_steps": 600}
        auto_vecnorm = ""

    dt = float(run_cfg.get("dt", 0.1))

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
    base_seed = int(args.seed)
    if hasattr(venv, "seed"):
        venv.seed(base_seed)
    obs = venv.reset()

    # Unwrap potential VecNormalize to reach underlying Dummy/Subproc env
    base_env_ref = venv
    while hasattr(base_env_ref, "venv"):
        base_env_ref = base_env_ref.venv
    base_env = base_env_ref.envs[0]
    action_low = np.array(base_env.action_space.low, dtype=np.float32)
    action_high = np.array(base_env.action_space.high, dtype=np.float32)
    resolution_m = float(getattr(base_env, "resolution_m", 0.05))
    scenario = getattr(base_env, "_scenario", None)
    goal_xy_global = (
        tuple(map(float, scenario.goal_xy)) if scenario is not None else (0.0, 0.0)
    )
    trajectory_samples: list[tuple[float, float, float, float, float, float, float]] = []
    mover_samples: list[tuple[float, float]] = []
    robot_trail_pts: list[tuple[float, float]] = []
    mover_trail_map: dict[int, dict[str, Any]] = {}
    sim_time = 0.0
    map_grid = None
    scenario_info_payload: dict[str, Any] = {}
    initial_payload = base_env.get_render_payload()
    if log_rollout:
        payload0 = initial_payload
        if isinstance(payload0, dict):
            pose0 = payload0.get("pose", (0.0, 0.0, 0.0))
            last_u0 = payload0.get("last_u", (0.0, 0.0))
            goal_dist0 = float(
                np.hypot(goal_xy_global[0] - pose0[0], goal_xy_global[1] - pose0[1])
            )
            trajectory_samples.append(
                (
                    sim_time,
                    float(pose0[0]),
                    float(pose0[1]),
                    float(pose0[2]),
                    float(last_u0[0]),
                    float(last_u0[1]),
                    goal_dist0,
                )
            )
            sim_time += dt
            raw_grid0 = payload0.get("raw_grid")
            if raw_grid0 is not None:
                map_grid = raw_grid0
            elif payload0.get("inflated_grid") is not None:
                map_grid = payload0.get("inflated_grid")
            scenario_info_payload = payload0.get("scenario_info", {}) or {}
    initial_pose = None
    if isinstance(initial_payload, dict):
        initial_pose = initial_payload.get("pose")
    if initial_pose:
        robot_trail_pts.append((float(initial_pose[0]), float(initial_pose[1])))

    robot_radius = float(robot_cfg.get("radius_m", 0.45))
    mover_radius = float(
        env_cfg.get("dynamic_movers", {}).get("radius_m", robot_radius)
    )
    controller_cfg = robot_cfg.get("controller", {})
    lookahead_m = float(controller_cfg.get("lookahead_m", 1.0))
    v_nominal = float(controller_cfg.get("speed_nominal", 0.0))

    apf_params = {
        "v_max": float(robot_cfg.get("v_max", 1.5)),
        "v_min": float(robot_cfg.get("v_min", -0.3)),
        "w_max": float(robot_cfg.get("w_max", 2.0)),
        "k_att": 2.0,  # Tuned: goal attraction
        "k_rep": 1.0,  # Tuned: obstacle repulsion
        "d_influence": 1.5,  # Tuned: obstacle influence radius
        "k_v": 1.0,  # Tuned: velocity gain
        "k_omega": 4.0,  # Tuned: angular velocity gain
        "robot_radius": float(robot_cfg.get("radius_m", 0.45)),
    }
    lidar_angles = None
    if controller_mode == "apf":
        lidar_angles = np.linspace(
            -base_env.lidar.fov_rad / 2.0,
            base_env.lidar.fov_rad / 2.0,
            base_env.lidar.beams,
        )

    model = None
    if controller_mode == "residual" and args.model:
        print(f"[INFO] Loading SAC model from: {args.model}")
        model = SAC.load(args.model, env=venv, print_system_info=False)
    elif controller_mode == "residual":
        print(
            "[INFO] No model provided: using zero-residual (pure pursuit) for playback."
        )
    elif controller_mode == "pure_pursuit":
        print("[INFO] Using pure pursuit controller without residual.")
    elif controller_mode == "apf":
        print("[INFO] Using Artificial Potential Field controller.")

    frames = []
    record_arg = args.record
    if isinstance(record_arg, str) and record_arg.strip().lower() in {"", "none"}:
        args.record = ""

    renderer = None
    need_renderer = bool(args.render or args.record or args.snapshot_out)
    if need_renderer:
        map_size_render = env_cfg["map"]["size_m"]
        viz = env_cfg["viz"]
        vcfg = VizConfig(
            size_px=(800, 800),
            show_inflated=bool(viz["show_inflated"]),
            show_lidar=bool(viz["show_lidar"]),
            show_actions=bool(viz["show_actions"]),
            fps=int(viz["fps"]),
        )
        renderer = Renderer(
            map_size_render, base_env.resolution_m, viz_cfg=vcfg, display=bool(args.render)
        )
    snapshot_surface = None

    final_info: dict[str, Any] = {}
    episode_done = False

    # Rollout
    for t in range(args.steps):
        action = None
        controller_payload = None
        if controller_mode == "apf":
            controller_payload = base_env.get_render_payload()
            if not isinstance(controller_payload, dict):
                raise RuntimeError("Failed to obtain environment payload for controller.")
            pose = controller_payload["pose"]
            scenario = getattr(base_env, "_scenario", None)
            if scenario is None:
                raise RuntimeError("Scenario not initialized in environment.")
            goal_xy = tuple(map(float, scenario.goal_xy))
            obs_dict = controller_payload.get("obs", {})
            lidar_scan = np.asarray(obs_dict.get("lidar", np.array([])))
            if lidar_scan.size == 0:
                raise RuntimeError("LiDAR data unavailable for controller.")
            v_cmd, w_cmd = compute_apf_command(
                pose=pose,
                goal=goal_xy,
                lidar_scan=lidar_scan,
                lidar_angles=lidar_angles,
                **apf_params,
            )

            waypoints = controller_payload.get("waypoints")
            if waypoints is None or len(waypoints) == 0:
                v_track = 0.0
                w_track = 0.0
            else:
                v_track, w_track = compute_u_track(
                    pose,
                    np.asarray(waypoints),
                    lookahead_m,
                    v_nominal,
                )
            dv = float(np.clip(v_cmd - v_track, action_low[0], action_high[0]))
            dw = float(np.clip(w_cmd - w_track, action_low[1], action_high[1]))
            action = np.array([[dv, dw]], dtype=np.float32)
        elif model is not None:
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        if action is None:
            action = np.zeros((venv.num_envs, 2), dtype=np.float32)

        obs, reward, done, info = venv.step(action)
        payload = base_env.get_render_payload()
        info_dict = info[0] if isinstance(info, (list, tuple)) else info

        if log_rollout and isinstance(payload, dict):
            pose = payload.get("pose", (0.0, 0.0, 0.0))
            last_u = payload.get("last_u", (0.0, 0.0))
            goal_dist = float(
                np.hypot(goal_xy_global[0] - pose[0], goal_xy_global[1] - pose[1])
            )
            trajectory_samples.append(
                (
                    sim_time,
                    float(pose[0]),
                    float(pose[1]),
                    float(pose[2]),
                    float(last_u[0]),
                    float(last_u[1]),
                    goal_dist,
                )
            )
            sim_time += dt
            grid_candidate = payload.get("raw_grid")
            if grid_candidate is None:
                grid_candidate = payload.get("inflated_grid")
            if grid_candidate is not None:
                map_grid = grid_candidate
            movers_list = payload.get("movers", []) or []
            for mover in movers_list:
                x = getattr(mover, "x", None)
                y = getattr(mover, "y", None)
                if x is None or y is None:
                    continue
                mover_samples.append((float(x), float(y)))
            scenario_info_payload = payload.get("scenario_info", {}) or scenario_info_payload
        movers_list = payload.get("movers", []) or []
        for mover in movers_list:
            mid = id(mover)
            trail_entry = mover_trail_map.setdefault(
                mid,
                {"points": [], "type": getattr(mover, "mover_type", "lateral")},
            )
            trail_entry["points"].append((float(mover.x), float(mover.y)))
        if isinstance(payload, dict):
            pose_tail = payload.get("pose")
            if pose_tail:
                robot_trail_pts.append((float(pose_tail[0]), float(pose_tail[1])))

        # VecEnv API (Gymnasium compatibility in our wrappers): done is array-like
        if np.any(done):
            episode_done = True
            final_info = info_dict if isinstance(info_dict, dict) else {}
            break
        if need_renderer:
            # VecEnv returns a list of infos; unwrap for single-env DummyVecEnv
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
                pose = payload["pose"]
                waypoints = payload.get("waypoints", np.array([]))
                if len(waypoints) > 0:
                    controller_cfg_local = robot_cfg.get("controller", {})
                    lookahead_cfg = controller_cfg_local.get("lookahead_m", lookahead_m)
                    v_nom_cfg = controller_cfg_local.get("speed_nominal", v_nominal)
                    u_track = compute_u_track(
                        pose, waypoints, float(lookahead_cfg), float(v_nom_cfg)
                    )
                    last_u_arr = np.array(last_u, dtype=float)
                    u_track_arr = np.array(u_track, dtype=float)
                    du = last_u_arr - u_track_arr
                    actions_data = (
                        tuple(u_track_arr.tolist()),
                        tuple(du.tolist()),
                        tuple(last_u_arr.tolist()),
                    )

            mover_trails_payload = [
                (entry["points"], entry["type"]) for entry in mover_trail_map.values()
            ]
            mover_trails_payload = [
                (entry["points"], entry["type"]) for entry in mover_trail_map.values()
            ]
            frame = renderer.render_frame(
                raw_grid=payload["raw_grid"],
                inflated_grid=payload.get("inflated_grid"),
                pose=payload["pose"],
                radius_m=payload["radius_m"],
                lidar=lidar_data,
                path=payload.get("waypoints"),
                actions=actions_data,
                hud=reward_breakdown,
                movers=payload.get("movers", []),
                trajectory=robot_trail_pts,
                mover_trails=mover_trails_payload,
            )
            snapshot_surface = frame.copy()
            if args.record:
                # Convert pygame Surface to numpy array immediately
                try:
                    arr = pygame.surfarray.array3d(frame).transpose(1, 0, 2)
                    frames.append(arr)
                except Exception as e:
                    print(f"Failed to convert frame to array: {e}")

    if args.record and len(frames) > 0:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        fps_out = int(env_cfg["viz"]["fps"])
        save_mp4(frames, str(out), fps=fps_out)
        print(f"Saved video with {len(frames)} frames to: {out}")

    if args.snapshot_out and snapshot_surface is not None:
        out_path = _ensure_parent(args.snapshot_out)
        pygame.image.save(snapshot_surface, str(out_path))
        print(f"[INFO] Snapshot saved to {out_path}")

    if log_rollout and trajectory_samples:
        traj_arr = np.array(trajectory_samples, dtype=float)
        success_flag = bool(final_info.get("is_success", False))
        metrics = _compute_rollout_metrics(traj_arr, dt, success_flag)
        if args.traj_out:
            _save_trajectory_csv(args.traj_out, traj_arr)
            print(f"[INFO] Trajectory samples saved to {args.traj_out}")
        if args.metrics_out:
            _save_metrics_json(args.metrics_out, metrics)
            print(f"[INFO] Metrics saved to {args.metrics_out}")
        summary = (
            f"success={success_flag} | time={metrics.get('time_elapsed_s', 0.0):.2f}s | "
            f"path={metrics.get('path_length_m', 0.0):.2f}m | smoothness={metrics.get('smoothness_score', 0.0):.4f}"
        )
        print(f"[INFO] Rollout summary: {summary}")


if __name__ == "__main__":
    main()
