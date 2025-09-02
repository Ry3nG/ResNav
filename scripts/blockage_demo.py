#!/usr/bin/env python3
"""
Simple temporary blockage demo with realistic dimensions and proper zoom.
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from me5418_nav.envs import UnicycleNavEnv, EnvConfig
from me5418_nav.controllers.pure_pursuit_apf import PurePursuitAPF, PPAPFConfig
from me5418_nav.controllers.dwa import DynamicWindowApproach, DWAConfig
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from me5418_nav.maps import (
    create_blockage_scenario as build_blockage_map,
    BlockageScenarioConfig,
)


def print_blockage_summary(
    grid: BinaryOccupancyGrid, waypoints: np.ndarray, info: dict
):
    H, W = grid.grid.shape
    res = float(getattr(grid, "_cellsize", GRID_RESOLUTION_M))
    map_w_m = W * res
    map_h_m = H * res
    print("Compact blockage scenario created:")
    print(f"  Map: {map_w_m:.1f}x{map_h_m:.1f}m ({W}x{H} cells)")
    print(f"  Corridor: {info.get('corridor_width', 0.0):.1f}m wide")
    print(f"  Pallet: {info.get('pallet_width', 0.0):.1f}m wide")
    print(f"  Gap above pallet: {info.get('gap_top', 0.0):.1f}m")
    print(f"  Gap below pallet: {info.get('gap_bottom', 0.0):.1f}m")
    print(f"  Path: {len(waypoints)} waypoints")
    occ = int(np.sum(grid.grid.astype(bool)))
    total = int(grid.grid.size)
    print(f"  Grid occupancy: {occ}/{total} ({occ/total*100:.1f}%)")


def run_demo(
    episodes: int = 1,
    max_steps: int = 1000,
    render: bool = True,
    controller: str = "dwa",
):
    """Run the blockage demo"""
    print("Creating compact blockage scenario...")
    # Use map generator for separation of concerns
    grid, waypoints, start_pose, goal_pos, scenario_info = build_blockage_map(
        BlockageScenarioConfig()
    )
    print_blockage_summary(grid, waypoints, scenario_info)

    # Create environment with smaller map
    cfg = EnvConfig(
        dt=DT_S,
        map_size=grid.grid.shape,
        res=grid._cellsize,
    )

    env = UnicycleNavEnv(
        cfg=cfg,
        render_mode="human" if render else None,
        grid=grid,
        path_waypoints=waypoints,
        start_pose=start_pose,
        goal_xy=goal_pos,
    )
    env.max_steps = max_steps

    # Print robot size and clearance using env configuration
    robot_diameter = 2.0 * float(env.cfg.robot_radius)
    gap_top = scenario_info["gap_top"]
    gap_bottom = scenario_info["gap_bottom"]
    clearance = max(gap_top, gap_bottom) - robot_diameter
    print(f"  Robot: {robot_diameter:.1f}m diameter")
    print(f"  Clearance (best gap): {clearance:.1f}m")

    # Create controller
    if controller == "ppapf":
        ctrl_name = "Pure Pursuit + APF"
        ctrl = PurePursuitAPF(PPAPFConfig())
    else:
        ctrl_name = "Dynamic Window Approach"
        ctrl = DynamicWindowApproach(DWAConfig())

    print(f"\nRunning {episodes} episode(s)...")

    results = []

    for ep in range(episodes):
        obs, info = env.reset()
        total_steps = 0

        print(f"\nEpisode {ep + 1}/{episodes}")

        while True:
            pose = env.robot.as_pose()
            v_limits = (env.robot.v_min, env.robot.v_max)
            w_limits = (env.robot.w_min, env.robot.w_max)

            rs = env.robot.get_state()
            v_curr, w_curr = float(rs.v), float(rs.omega)
            # Controllers expect different grid formats
            if controller == "ppapf":
                grid_arg = env.grids  # PPAPF needs full grids object
            else:
                grid_arg = env.grids.sensing  # DWA uses sensing grid for LiDAR casting
                
            cmd = ctrl.action(
                pose,
                env.path_waypoints,
                env.lidar,
                grid_arg,
                v_limits,
                w_limits,
                env.goal_xy,
                v_curr,
                w_curr,
            )
            obs, reward, terminated, truncated, info = env.step((cmd.v, cmd.w))
            total_steps += 1

            if render:
                status = {
                    "step": total_steps,
                    "scenario": "APF Navigation (3.0m corridor, 0.8m pallet)",
                    "controller": ctrl_name,
                }
                env._debug_status = status
                env.render()

            # Progress logging
            if total_steps % 25 == 0:
                min_range = (
                    np.min(obs[: env.cfg.lidar_beams]) if env.cfg.lidar_beams > 0 else 0
                )
                print(
                    f"  Step {total_steps}: {ctrl_name}, Pos=({pose[0]:.1f}, {pose[1]:.1f}), MinRange={min_range:.2f}m"
                )

            # Check termination
            if terminated or truncated or total_steps >= max_steps:
                success = info.get("success", False)
                collision = info.get("collision", False)

                if success:
                    print(f"  ✅ SUCCESS in {total_steps} steps!")
                    results.append("success")
                elif collision:
                    print(f"  💥 COLLISION at step {total_steps}")
                    results.append("collision")
                else:
                    print(f"  ⏱️  TIMEOUT after {total_steps} steps")
                    results.append("timeout")
                break

    # Results summary
    success_count = results.count("success")
    collision_count = results.count("collision")
    timeout_count = results.count("timeout")

    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(
        f"Collision: {collision_count}/{episodes} ({collision_count/episodes*100:.1f}%)"
    )
    print(f"Timeout: {timeout_count}/{episodes} ({timeout_count/episodes*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compact blockage demo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--controller", choices=["ppapf", "dwa"], default="dwa")

    args = parser.parse_args()

    print("Blockage Navigation Demo")
    print("=" * 28)
    print("Setup:")
    print("- Compact 10x10m map for better zoom")
    # Keep banner consistent with scenario parameters defined above
    print("- Corridor width: 2.5m")
    print("- Pallet width: 1.0m (short along corridor)")
    print(f"- Controller: {args.controller.upper()}")
    print("- Displays collision in renderer when geometric overlap occurs")
    print()

    results = run_demo(
        episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
        controller=args.controller,
    )


if __name__ == "__main__":
    main()
