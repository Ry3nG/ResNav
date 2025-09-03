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
    print("Blockage scenario created:")
    print(f"  Map: {map_w_m:.1f}x{map_h_m:.1f}m ({W}x{H} cells)")
    print(f"  Corridor: {info.get('corridor_width', 0.0):.1f}m wide")

    num_pallets = info.get("num_pallets", 0)
    print(f"  Pallets: {num_pallets} total")

    if num_pallets > 0:
        pallets = info.get("pallets", [])
        for i, pallet in enumerate(pallets):
            print(
                f"    Pallet {i+1}: {pallet['width']:.1f}m x {pallet['length']:.1f}m at ({pallet['x']:.1f}, {pallet['y']:.1f})"
            )

        print(f"  Total pallet width: {info.get('total_pallet_width', 0.0):.1f}m")
        print(f"  Min passage width: {info.get('min_passage_width', 0.0):.1f}m")
        print(f"  Gap above: {info.get('gap_top', 0.0):.1f}m")
        print(f"  Gap below: {info.get('gap_bottom', 0.0):.1f}m")
        print(f"  Min clearance: {info.get('min_clearance', 0.0):.1f}m")
        print(f"  Difficulty score: {info.get('difficulty_score', 0.0):.2f}")
    else:
        print(f"  No pallets - full corridor available")

    print(f"  Robot diameter: {info.get('robot_diameter', 0.0):.1f}m")
    print(f"  Path: {len(waypoints)} waypoints")
    occ = int(np.sum(grid.grid.astype(bool)))
    total = int(grid.grid.size)
    print(f"  Grid occupancy: {occ}/{total} ({occ/total*100:.1f}%)")
    print(f"  Map seed: {info.get('actual_seed', 'unknown')}")


def run_demo(
    episodes: int = 1,
    max_steps: int = 1000,
    render: bool = True,
    controller: str = "dwa",
    scenario_config: BlockageScenarioConfig | None = None,
):
    """Run the blockage demo"""
    if scenario_config is None:
        scenario_config = BlockageScenarioConfig()
        print("Creating blockage scenario...")

    # Use map generator for separation of concerns
    grid, waypoints, start_pose, goal_pos, scenario_info = build_blockage_map(
        scenario_config
    )

    # Print the actual seed that was used for map generation
    actual_seed = scenario_info.get("actual_seed", "unknown")
    if scenario_config.random_seed is not None:
        print(f"Using seed: {scenario_config.random_seed}")
    else:
        print(f"Using random seed: {actual_seed}")

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

        # Build a concise scenario label for renderer/debug
        scenario_label = (
            f"Corridor {scenario_info.get('corridor_width', 0):.1f}m, "
            f"{scenario_info.get('num_pallets', 0)} pallet(s), "
            f"min pass {scenario_info.get('min_passage_width', 0):.1f}m"
        )

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
                env._debug_status = {
                    "step": total_steps,
                    "scenario": scenario_label,
                    "controller": ctrl_name,
                }
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
    parser = argparse.ArgumentParser(description="Blockage demo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--controller", choices=["ppapf", "dwa"], default="dwa")

    # Scenario options
    parser.add_argument(
        "--num-pallets",
        type=str,
        default="0,5",
        help="Min,max number of pallets (e.g., '1,3')",
    )
    parser.add_argument(
        "--pallet-width",
        type=str,
        default="0.5,1.1",
        help="Min,max pallet width in meters (e.g., '0.9,1.2')",
    )
    parser.add_argument(
        "--pallet-length",
        type=str,
        default="0.3,0.6",
        help="Min,max pallet length in meters (e.g., '0.4,0.8')",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible scenarios"
    )

    args = parser.parse_args()

    # Parse ranges
    num_pallets_range = tuple(map(int, args.num_pallets.split(",")))
    pallet_width_range = tuple(map(float, args.pallet_width.split(",")))
    pallet_length_range = tuple(map(float, args.pallet_length.split(",")))

    # Create enhanced scenario config
    scenario_config = BlockageScenarioConfig(
        num_pallets_range=num_pallets_range,
        pallet_width_range=pallet_width_range,
        pallet_length_range=pallet_length_range,
        random_seed=args.seed,
    )

    print("Blockage Demo")
    print("=" * 13)
    print("Setup:")
    print(f"- Map: 10x10m with 2.5m corridor")
    print(f"- Pallets: {num_pallets_range[0]}-{num_pallets_range[1]} count")
    print(
        f"- Pallet size: {pallet_width_range[0]:.1f}-{pallet_width_range[1]:.1f}m x {pallet_length_range[0]:.1f}-{pallet_length_range[1]:.1f}m"
    )
    print(f"- Seed: {args.seed if args.seed else 'random'}")
    print(f"- Controller: {args.controller.upper()}")
    # Note: Actual seed will be printed after map generation

    results = run_demo(
        episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
        controller=args.controller,
        scenario_config=scenario_config,
    )


if __name__ == "__main__":
    main()
