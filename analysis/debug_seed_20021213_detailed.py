#!/usr/bin/env python3
"""Detailed debugging of seed 20021213 behavior.

This script investigates why seed 20021213 now appears feasible
when it was previously infeasible.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid


def debug_seed_20021213_detailed():
    """Detailed debugging of seed 20021213."""

    print("Detailed debugging of seed 20021213")
    print("=" * 50)

    # Configuration
    cfg = BlockageScenarioConfig(
        map_width_m=20.0,
        map_height_m=20.0,
        corridor_width_min_m=3.0,
        corridor_width_max_m=5.0,
        wall_thickness_m=0.3,
        pallet_width_m=1.2,
        pallet_length_m=1.0,
        start_x_m=1.0,
        goal_margin_x_m=1.0,
        waypoint_step_m=0.3,
        resolution_m=0.2,
        min_passage_m=0.7,
        min_pallet_x_offset_m=1.0,
        num_pallets_min=3,
        num_pallets_max=5,
    )

    # Generate scenario with seed 20021213
    rng = np.random.default_rng(20021213)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    print(f"Scenario details:")
    print(f"  Start pose: {start_pose}")
    print(f"  Goal: {goal_xy}")
    print(f"  Corridor width: {info['corridor_width']:.2f}m")
    print(f"  Number of pallets: {info['num_pallets']}")
    print(f"  Pallet centers: {info['pallet_centers']}")
    print(f"  Pallet sizes: {info['pallet_sizes']}")

    # Check path feasibility with different robot sizes
    print(f"\nPath feasibility check:")

    for robot_radius in [0.25, 0.3, 0.35, 0.4]:
        grid_inflated = inflate_grid(grid, robot_radius, cfg.resolution_m)
        is_feasible = check_path_feasibility(grid_inflated, start_pose, goal_xy, cfg.resolution_m)
        print(f"  Robot radius {robot_radius:.2f}m: {'✅ Feasible' if is_feasible else '❌ Infeasible'}")

    # Analyze corridor and pallet placement
    print(f"\nCorridor analysis:")
    corridor_center_y = cfg.map_height_m / 2.0
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    print(f"  Corridor center Y: {corridor_center_y:.2f}m")
    print(f"  Corridor top: {y_top:.2f}m")
    print(f"  Corridor bottom: {y_bot:.2f}m")
    print(f"  Corridor width: {corridor_width:.2f}m")

    print(f"\nPallet analysis:")
    for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
        print(f"  Pallet {i+1}: center=({px:.2f}, {py:.2f}), size=({pl:.2f}, {pw:.2f})")

        pallet_top = py + pw/2
        pallet_bot = py - pw/2

        passage_above = y_top - pallet_top
        passage_below = pallet_bot - y_bot

        print(f"    Bounds: top={pallet_top:.2f}m, bottom={pallet_bot:.2f}m")
        print(f"    Passage above: {passage_above:.2f}m")
        print(f"    Passage below: {passage_below:.2f}m")
        print(f"    Min required: {cfg.min_passage_m:.2f}m")

        robot_diameter = 0.5
        can_pass_above = passage_above >= robot_diameter
        can_pass_below = passage_below >= robot_diameter

        print(f"    Robot can pass above: {can_pass_above}")
        print(f"    Robot can pass below: {can_pass_below}")

    # Check if this is the same scenario as before
    print(f"\nComparison with previous analysis:")
    print(f"  This scenario has corridor width {info['corridor_width']:.2f}m")
    print(f"  Previous analysis showed 3.18m corridor width")
    print(f"  Pallet count: {info['num_pallets']} (previous: 3)")

    # Check if the scenario is actually different
    if abs(info['corridor_width'] - 3.18) < 0.01 and info['num_pallets'] == 3:
        print(f"  ⚠️  This appears to be the same scenario as before!")
        print(f"  The path feasibility check might be different now.")
    else:
        print(f"  ✅ This is a different scenario than before.")
        print(f"  The scenario generation might have changed.")


def check_path_feasibility(grid_inflated: np.ndarray, start_pose: tuple, goal_xy: tuple, resolution: float) -> bool:
    """Check path feasibility using BFS."""
    from collections import deque

    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_xy

    # Convert to grid coordinates
    start_i = int(np.floor(start_y / resolution))
    start_j = int(np.floor(start_x / resolution))
    goal_i = int(np.floor(goal_y / resolution))
    goal_j = int(np.floor(goal_x / resolution))

    H, W = grid_inflated.shape

    # Check bounds
    if (start_i < 0 or start_i >= H or start_j < 0 or start_j >= W or
        goal_i < 0 or goal_i >= H or goal_j < 0 or goal_j >= W):
        return False

    # Check if start/goal are in obstacles
    if grid_inflated[start_i, start_j] or grid_inflated[goal_i, goal_j]:
        return False

    # BFS to find path
    visited = np.zeros_like(grid_inflated, dtype=bool)
    queue = deque([(start_i, start_j)])
    visited[start_i, start_j] = True

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        i, j = queue.popleft()

        if i == goal_i and j == goal_j:
            return True

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (0 <= ni < H and 0 <= nj < W and
                not visited[ni, nj] and not grid_inflated[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))

    return False


if __name__ == "__main__":
    debug_seed_20021213_detailed()
