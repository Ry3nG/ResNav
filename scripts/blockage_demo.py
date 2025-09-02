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
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid


def create_blockage_scenario():
    """Create a smaller, more focused temporary blockage scenario"""
    # Smaller, focused map: 10m x 10m
    map_width_m = 10.0
    map_height_m = 10.0
    resolution = GRID_RESOLUTION_M  # 0.1m

    grid_width = int(map_width_m / resolution)  # 200 cells
    grid_height = int(map_height_m / resolution)  # 100 cells

    # Create empty grid
    grid_array = np.zeros((grid_height, grid_width), dtype=bool)

    # Corridor setup - generous width for demonstration
    corridor_y_center = map_height_m / 2  # 5.0m
    corridor_width = 2.5
    wall_thickness = 0.3

    # Convert meters to grid indices
    def meters_to_grid(x_m, y_m):
        i = int(y_m / resolution)  # row (y)
        j = int(x_m / resolution)  # col (x)
        return min(max(i, 0), grid_height - 1), min(max(j, 0), grid_width - 1)

    def fill_rectangle(x_min, y_min, x_max, y_max):
        """Fill rectangle in grid with obstacles"""
        i_min, j_min = meters_to_grid(x_min, y_min)
        i_max, j_max = meters_to_grid(x_max, y_max)
        i_min, i_max = min(i_min, i_max), max(i_min, i_max)
        j_min, j_max = min(j_min, j_max), max(j_min, j_max)
        grid_array[i_min : i_max + 1, j_min : j_max + 1] = True

    # Top corridor wall
    top_wall_y = corridor_y_center + corridor_width / 2
    fill_rectangle(0, top_wall_y, map_width_m, top_wall_y + wall_thickness)

    # Bottom corridor wall
    bottom_wall_y = corridor_y_center - corridor_width / 2
    fill_rectangle(0, bottom_wall_y - wall_thickness, map_width_m, bottom_wall_y)

    # Blocking pallet - small obstacle to create clear gap
    pallet_x = map_width_m / 2  # 10.0m (center)
    pallet_width = 1  # Small: 0.8m cross-corridor width
    pallet_length = 0.6  # Short along corridor
    pallet_y = corridor_y_center  # Offset to create 1.7m gap on bottom

    fill_rectangle(
        pallet_x - pallet_length / 2,
        pallet_y - pallet_width / 2,
        pallet_x + pallet_length / 2,
        pallet_y + pallet_width / 2,
    )

    # Create BinaryOccupancyGrid
    grid = BinaryOccupancyGrid(grid_array, cellsize=resolution, origin=(0, 0))

    # Create straight path through corridor
    start_x, start_y = 1.0, corridor_y_center
    goal_x, goal_y = map_width_m - 1.0, corridor_y_center

    # Generate waypoints
    num_waypoints = int((goal_x - start_x) / 0.3)
    x_coords = np.linspace(start_x, goal_x, num_waypoints)
    y_coords = np.full_like(x_coords, start_y)
    waypoints = np.stack([x_coords, y_coords], axis=1)

    start_pose = (start_x, start_y, 0.0)  # facing right
    goal_pos = (goal_x, goal_y)

    # Calculate actual gaps
    gap_top = top_wall_y - (pallet_y + pallet_width / 2)  # Gap above pallet
    gap_bottom = (pallet_y - pallet_width / 2) - bottom_wall_y  # Gap below pallet
    print(f"Compact blockage scenario created:")
    print(f"  Map: {map_width_m}x{map_height_m}m ({grid_width}x{grid_height} cells)")
    print(f"  Corridor: {corridor_width:.1f}m wide")
    print(
        f"  Pallet: {pallet_width:.1f}m wide, offset +{pallet_y - corridor_y_center:.1f}m"
    )
    print(f"  Gap above pallet: {gap_top:.1f}m")
    print(f"  Gap below pallet: {gap_bottom:.1f}m")
    print(f"  Path: {len(waypoints)} waypoints")
    print(
        f"  Grid occupancy: {np.sum(grid_array)}/{grid_array.size} ({np.sum(grid_array)/grid_array.size*100:.1f}%)"
    )

    scenario_info = {
        "gap_top": gap_top,
        "gap_bottom": gap_bottom,
        "corridor_width": corridor_width,
        "pallet_width": pallet_width,
    }

    return grid, waypoints, start_pose, goal_pos, scenario_info


def run_demo(episodes: int = 1, max_steps: int = 1000, render: bool = True):
    """Run the blockage demo"""
    print("Creating compact blockage scenario...")
    grid, waypoints, start_pose, goal_pos, scenario_info = create_blockage_scenario()

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

    # Create controller - regular APF (no wall following)
    ctrl = PurePursuitAPF(PPAPFConfig())

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

            action = ctrl.action(
                pose, env.path_waypoints, env.lidar, env.grid, v_limits, w_limits
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            if render:
                # Simple APF controller doesn't have complex debug info
                status = {
                    "step": total_steps,
                    "scenario": "APF Navigation (3.0m corridor, 0.8m pallet)",
                    "controller": "Pure Pursuit + APF (no wall following)",
                }
                env._debug_status = status
                env.render()

            # Progress logging
            if total_steps % 25 == 0:
                min_range = (
                    np.min(obs[: env.cfg.lidar_beams]) if env.cfg.lidar_beams > 0 else 0
                )
                print(
                    f"  Step {total_steps}: APF Controller, "
                    f"Pos=({pose[0]:.1f}, {pose[1]:.1f}), MinRange={min_range:.2f}m"
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

    args = parser.parse_args()

    print("APF Blockage Navigation Demo")
    print("=" * 28)
    print("Setup:")
    print("- Compact 10x10m map for better zoom")
    # Keep banner consistent with scenario parameters defined above
    print("- Corridor width: 2.5m")
    print("- Pallet width: 1.0m (short along corridor)")
    print("- Controller: Pure Pursuit + APF (no wall following)")
    print("- Displays collision in renderer when geometric overlap occurs")
    print()

    results = run_demo(
        episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
