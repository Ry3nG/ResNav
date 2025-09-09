#!/usr/bin/env python3
"""Compare different path checking implementations."""

import numpy as np
import sys
import os
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid
from amr_env.sim.scenario_manager import ScenarioManager


def path_check_with_inflation(grid, start_pose, goal_xy, resolution, robot_radius=0.25):
    """Path check with robot radius inflation."""
    # Create inflated grid
    grid_inflated = inflate_grid(grid, robot_radius, resolution)

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


def path_check_without_inflation(grid, start_pose, goal_xy, resolution):
    """Path check without robot radius inflation (original scenario manager)."""
    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_xy

    # Convert to grid coordinates
    start_i = int(np.floor(start_y / resolution))
    start_j = int(np.floor(start_x / resolution))
    goal_i = int(np.floor(goal_y / resolution))
    goal_j = int(np.floor(goal_x / resolution))

    H, W = grid.shape

    # Check bounds
    if (start_i < 0 or start_i >= H or start_j < 0 or start_j >= W or
        goal_i < 0 or goal_i >= H or goal_j < 0 or goal_j >= W):
        return False

    # Check if start/goal are in obstacles
    if grid[start_i, start_j] or grid[goal_i, goal_j]:
        return False

    # BFS to find path
    visited = np.zeros_like(grid, dtype=bool)
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
                not visited[ni, nj] and not grid[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))

    return False


def main():
    """Compare different path checking methods."""

    print("Comparing path checking methods")
    print("=" * 50)

    # Generate scenario with seed 20021213
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

    rng = np.random.default_rng(20021213)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    print(f"Scenario: {info['corridor_width']:.2f}m corridor, {info['num_pallets']} pallets")

    # Test different path checking methods
    print(f"\nPath feasibility results:")

    # 1. Without inflation (scenario manager method)
    feasible_no_inflation = path_check_without_inflation(grid, start_pose, goal_xy, 0.2)
    print(f"  Without robot inflation: {'✅ Feasible' if feasible_no_inflation else '❌ Infeasible'}")

    # 2. With inflation (realistic method)
    for robot_radius in [0.25, 0.3, 0.35, 0.4]:
        feasible_with_inflation = path_check_with_inflation(grid, start_pose, goal_xy, 0.2, robot_radius)
        print(f"  With robot radius {robot_radius:.2f}m: {'✅ Feasible' if feasible_with_inflation else '❌ Infeasible'}")

    # 3. Scenario manager method
    env_cfg = {'map': {'size_m': [20.0, 20.0], 'resolution_m': 0.2, 'corridor_width_m': [3.0, 5.0], 'wall_thickness_m': 0.3, 'pallet_width_m': 1.2, 'pallet_length_m': 1.0, 'start_x_m': 1.0, 'goal_margin_x_m': 1.0, 'waypoint_step_m': 0.3, 'min_passage_m': 0.7, 'min_pallet_x_offset_m': 1.0, 'num_pallets_min': 3, 'num_pallets_max': 5}}
    sm = ScenarioManager(env_cfg)
    sm.set_seed(20021213)
    sm_grid, sm_waypoints, sm_start_pose, sm_goal_xy, sm_info = sm.sample()
    feasible_sm = sm._is_path_feasible(sm_grid, sm_start_pose, sm_goal_xy, 0.2)
    print(f"  Scenario manager method: {'✅ Feasible' if feasible_sm else '❌ Infeasible'}")

    print(f"\nAnalysis:")
    print(f"  The scenario manager uses raw grid without robot inflation")
    print(f"  This means it only checks if a point-sized robot can pass")
    print(f"  But real robots have radius 0.25m, which requires inflation")
    print(f"  This explains why the scenario manager says 'feasible' but")
    print(f"  the realistic check with inflation says 'infeasible'")


if __name__ == "__main__":
    main()
