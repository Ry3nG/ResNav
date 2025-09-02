from __future__ import annotations

import argparse
import numpy as np

from me5418_nav.envs import UnicycleNavEnv, EnvConfig
from me5418_nav.maps.s_path import make_s_map
from me5418_nav.controllers.pp_apf_trapaware import TrapAwarePPAPF, TrapAwareConfig
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S


def run_baseline(
    episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    debug_log: str | None = None,
):
    spec = make_s_map(size=(150, 120), res=GRID_RESOLUTION_M, amp=2.0, periods=1.5)
    cfg = EnvConfig(
        dt=DT_S,
        map_size=(spec.grid.grid.shape[0], spec.grid.grid.shape[1]),
        res=spec.grid._cellsize,
    )
    env = UnicycleNavEnv(
        cfg=cfg,
        render_mode="human" if render else None,
        grid=spec.grid,
        path_waypoints=spec.waypoints,
        start_pose=spec.start,
        goal_xy=spec.goal,
    )
    # Override env internal horizon with CLI-provided max_steps
    env.max_steps = max_steps
    ctrl = TrapAwarePPAPF(TrapAwareConfig(), dt=cfg.dt)

    # optional debug csv
    writer = None
    f_csv = None
    if debug_log is not None:
        import csv, os

        os.makedirs(os.path.dirname(debug_log) or ".", exist_ok=True)
        f_csv = open(debug_log, "w", newline="")
        writer = csv.writer(f_csv)
        header = [
            "episode",
            "step",
            "mode",
            "follow_side",
            "switch_to",
            "switch_reason",
            "x",
            "y",
            "theta",
            "v_cmd",
            "w_cmd",
            "min_range",
            "heading_err",
            "theta_des",
            "theta_prev",
            "theta_jump",
            "wp_idx",
            "p_look_x",
            "p_look_y",
            "F_x",
            "F_y",
            "far_wp_x",
            "far_wp_y",
            "los_clear",
            "wall_Cx",
            "wall_Cy",
            "wall_tx",
            "wall_ty",
            "wall_nx",
            "wall_ny",
            "wall_conf",
            "d_to_wall",
            "clear_err",
            "terminated",
            "truncated",
            "success",
            "collision",
        ]
        writer.writerow(header)

    n_success = 0
    n_collision = 0
    n_truncated = 0
    for ep in range(episodes):
        obs, info = env.reset()
        total_steps = 0
        prev_theta_des = None
        while True:
            pose = env.robot.as_pose()
            v_limits = (env.robot.v_min, env.robot.v_max)
            w_limits = (env.robot.w_min, env.robot.w_max)
            action = ctrl.action(
                pose, env.path_waypoints, env.lidar, env.grid, v_limits, w_limits
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            if render:
                # pass controller debug info for overlay
                try:
                    idx = env._nearest_path_index(pose[0], pose[1])
                except Exception:
                    idx = None
                status = dict(ctrl.debug)
                status.update(
                    {
                        "step": total_steps,
                        "wp_idx": idx,
                    }
                )
                env._debug_status = status
                env.render()
            # debug logging
            if writer is not None:
                import math

                d = dict(ctrl.debug)
                mode = d.get("mode")
                theta_des = d.get("theta_des")
                switched_to = d.get("switched_to")
                switch_reason = d.get("switch_reason")
                theta_jump = None
                if (
                    prev_theta_des is not None
                    and theta_des is not None
                    and switched_to is not None
                ):
                    a = float(theta_des)
                    b = float(prev_theta_des)
                    theta_jump = ((a - b + math.pi) % (2 * math.pi)) - math.pi
                min_range = (
                    float(np.min(obs[: env.cfg.lidar_beams]))
                    if env.cfg.lidar_beams > 0
                    else float("nan")
                )
                row = [
                    ep,
                    total_steps,
                    mode,
                    d.get("follow_side"),
                    switched_to,
                    switch_reason,
                    pose[0],
                    pose[1],
                    pose[2],
                    action[0],
                    action[1],
                    min_range,
                    d.get("heading_err"),
                    theta_des,
                    d.get("theta_des_prev"),
                    theta_jump,
                    idx,
                    d.get("p_look_x"),
                    d.get("p_look_y"),
                    d.get("F_x"),
                    d.get("F_y"),
                    d.get("far_wp_x"),
                    d.get("far_wp_y"),
                    1 if switch_reason == "los_clear" else 0,
                    d.get("wall_Cx"),
                    d.get("wall_Cy"),
                    d.get("wall_tx"),
                    d.get("wall_ty"),
                    d.get("wall_nx"),
                    d.get("wall_ny"),
                    d.get("wall_conf"),
                    d.get("d_to_wall"),
                    d.get("clear_err"),
                    int(terminated),
                    int(truncated),
                    int(info.get("success", False)),
                    int(info.get("collision", False)),
                ]
                writer.writerow(row)
                prev_theta_des = theta_des
            if terminated or truncated or total_steps >= max_steps:
                if info.get("success", False):
                    n_success += 1
                elif info.get("collision", False):
                    n_collision += 1
                else:
                    n_truncated += 1
                break
    if f_csv is not None:
        f_csv.close()
    print(f"Episodes: {episodes}")
    print(f"Success: {n_success}")
    print(f"Collision: {n_collision}")
    print(f"Timeout: {n_truncated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--debug-log",
        type=str,
        default=None,
        help="Path to CSV for detailed per-step controller debug",
    )
    args = parser.parse_args()
    run_baseline(
        episodes=args.episodes,
        max_steps=args.steps,
        render=args.render,
        debug_log=args.debug_log,
    )
