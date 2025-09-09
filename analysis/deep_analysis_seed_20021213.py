#!/usr/bin/env python3
"""Deep analysis of why seed 20021213 map is not feasible.

Even though individual pallets have sufficient passage, the combination
of multiple pallets might create a bottleneck that makes the path infeasible.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Set
import sys
import os
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid


def deep_analyze_seed_20021213():
    """Deep analysis of the specific seed 20021213 map."""

    print("Deep analysis of seed 20021213 map")
    print("=" * 50)

    seed = 20021213
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

    rng = np.random.default_rng(seed)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    # Create inflated grid
    robot_radius = 0.25
    safety_margin = 0.0
    grid_inflated = inflate_grid(grid, robot_radius + safety_margin, cfg.resolution_m)

    print(f"Map info:")
    print(f"  Dimensions: {grid.shape}")
    print(f"  Corridor width: {info['corridor_width']:.2f}m")
    print(f"  Start: {start_pose}")
    print(f"  Goal: {goal_xy}")
    print(f"  Pallets: {len(info['pallet_centers'])}")

    # Analyze corridor structure
    corridor_center_y = cfg.map_height_m / 2.0
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    print(f"\nCorridor structure:")
    print(f"  Center Y: {corridor_center_y:.2f}m")
    print(f"  Top wall: {y_top:.2f}m")
    print(f"  Bottom wall: {y_bot:.2f}m")
    print(f"  Width: {corridor_width:.2f}m")

    # Analyze each pallet's impact
    print(f"\nPallet analysis:")
    for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
        print(f"\nPallet {i+1}:")
        print(f"  Position: ({px:.2f}, {py:.2f})")
        print(f"  Size: {pl:.2f} x {pw:.2f}")

        pallet_top = py + pw/2
        pallet_bot = py - pw/2

        passage_above = y_top - pallet_top
        passage_below = pallet_bot - y_bot

        print(f"  Bounds: top={pallet_top:.2f}m, bottom={pallet_bot:.2f}m")
        print(f"  Passage above: {passage_above:.2f}m")
        print(f"  Passage below: {passage_below:.2f}m")
        print(f"  Min required: {cfg.min_passage_m:.2f}m")

        # Check if robot can pass
        robot_diameter = 0.5
        can_pass_above = passage_above >= robot_diameter
        can_pass_below = passage_below >= robot_diameter

        print(f"  Robot can pass above: {can_pass_above}")
        print(f"  Robot can pass below: {can_pass_below}")

    # Check path feasibility with different robot sizes
    print(f"\nPath feasibility analysis:")

    for robot_radius in [0.25, 0.3, 0.35, 0.4]:
        grid_test = inflate_grid(grid, robot_radius, cfg.resolution_m)
        feasible = check_path_feasibility_detailed(grid_test, start_pose, goal_xy, cfg.resolution_m)
        print(f"  Robot radius {robot_radius:.2f}m: {'✅ FEASIBLE' if feasible else '❌ NOT FEASIBLE'}")

    # Analyze the bottleneck
    print(f"\nBottleneck analysis:")
    analyze_bottlenecks(grid_inflated, start_pose, goal_xy, cfg.resolution_m, info)

    # Create detailed visualization
    create_detailed_visualization(grid, grid_inflated, start_pose, goal_xy, waypoints, info, cfg, seed)

    return grid, grid_inflated, start_pose, goal_xy, waypoints, info


def check_path_feasibility_detailed(grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                                   goal_xy: Tuple[float, float], resolution_m: float) -> bool:
    """Detailed pathfinding with path reconstruction."""

    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_xy

    # Convert to grid coordinates
    start_i = int(np.floor(start_y / resolution_m))
    start_j = int(np.floor(start_x / resolution_m))
    goal_i = int(np.floor(goal_y / resolution_m))
    goal_j = int(np.floor(goal_x / resolution_m))

    H, W = grid_inflated.shape

    # Check bounds
    if (start_i < 0 or start_i >= H or start_j < 0 or start_j >= W or
        goal_i < 0 or goal_i >= H or goal_j < 0 or goal_j >= W):
        return False

    # Check if start/goal are in obstacles
    if grid_inflated[start_i, start_j] or grid_inflated[goal_i, goal_j]:
        return False

    # BFS with path reconstruction
    visited = np.zeros_like(grid_inflated, dtype=bool)
    parent = np.full((H, W, 2), -1, dtype=int)
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
                parent[ni, nj] = [i, j]
                queue.append((ni, nj))

    return False


def analyze_bottlenecks(grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                       goal_xy: Tuple[float, float], resolution_m: float, info: dict):
    """Analyze where the path gets blocked."""

    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_xy

    # Convert to grid coordinates
    start_i = int(np.floor(start_y / resolution_m))
    start_j = int(np.floor(start_x / resolution_m))
    goal_i = int(np.floor(goal_y / resolution_m))
    goal_j = int(np.floor(goal_x / resolution_m))

    H, W = grid_inflated.shape

    # Find all reachable cells from start
    visited = np.zeros_like(grid_inflated, dtype=bool)
    queue = deque([(start_i, start_j)])
    visited[start_i, start_j] = True

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        i, j = queue.popleft()

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (0 <= ni < H and 0 <= nj < W and
                not visited[ni, nj] and not grid_inflated[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))

    # Check if goal is reachable
    goal_reachable = visited[goal_i, goal_j]
    print(f"  Goal reachable from start: {goal_reachable}")

    if not goal_reachable:
        # Find the closest reachable point to goal
        min_dist = float('inf')
        closest_i, closest_j = -1, -1

        for i in range(H):
            for j in range(W):
                if visited[i, j]:
                    dist = np.sqrt((i - goal_i)**2 + (j - goal_j)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_i, closest_j = i, j

        if closest_i != -1:
            closest_x = closest_j * resolution_m
            closest_y = closest_i * resolution_m
            print(f"  Closest reachable point to goal: ({closest_x:.2f}, {closest_y:.2f})")
            print(f"  Distance to goal: {min_dist * resolution_m:.2f}m")

    # Analyze corridor width at different X positions
    print(f"\nCorridor width analysis:")
    corridor_center_y = 10.0  # Fixed for this map
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    for x in np.linspace(1.0, 19.0, 19):
        j = int(np.floor(x / resolution_m))
        if 0 <= j < W:
            # Find top and bottom free cells at this X
            top_free = -1
            bottom_free = -1

            for i in range(H):
                if not grid_inflated[i, j]:
                    if top_free == -1:
                        top_free = i
                    bottom_free = i

            if top_free != -1 and bottom_free != -1:
                width = (bottom_free - top_free + 1) * resolution_m
                print(f"  X={x:.1f}m: width={width:.2f}m (top={top_free}, bottom={bottom_free})")
            else:
                print(f"  X={x:.1f}m: BLOCKED")


def create_detailed_visualization(grid: np.ndarray, grid_inflated: np.ndarray,
                                 start_pose: Tuple[float, float, float], goal_xy: Tuple[float, float],
                                 waypoints: np.ndarray, info: dict, cfg: BlockageScenarioConfig, seed: int):
    """Create detailed visualization of the map."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Raw grid
    ax1.imshow(grid, cmap='gray_r', origin='lower')
    ax1.set_title('Raw Grid')

    # Inflated grid
    ax2.imshow(grid_inflated, cmap='gray_r', origin='lower')
    ax2.set_title('Inflated Grid (Robot + Safety)')

    # Corridor width analysis
    corridor_center_y = 10.0
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    # Plot corridor bounds
    for ax in [ax1, ax2]:
        ax.axhline(y=corridor_center_y, color='blue', linestyle='--', alpha=0.5, label='Corridor center')
        ax.axhline(y=y_top, color='red', linestyle='-', alpha=0.7, label='Corridor top')
        ax.axhline(y=y_bot, color='red', linestyle='-', alpha=0.7, label='Corridor bottom')

    # Plot start, goal, waypoints, and pallets
    def world_to_grid(x, y):
        i = int(np.floor(y / cfg.resolution_m))
        j = int(np.floor(x / cfg.resolution_m))
        return i, j

    start_x, start_y, start_theta = start_pose
    goal_x, goal_y = goal_xy

    start_i, start_j = world_to_grid(start_x, start_y)
    goal_i, goal_j = world_to_grid(goal_x, goal_y)

    for ax in [ax1, ax2]:
        ax.plot(start_j, start_i, 'go', markersize=10, label='Start')
        ax.plot(goal_j, goal_i, 'ro', markersize=10, label='Goal')

        # Plot waypoints
        wp_ijs = [world_to_grid(wp[0], wp[1]) for wp in waypoints]
        wp_is, wp_js = zip(*wp_ijs)
        ax.plot(wp_js, wp_is, 'b-', linewidth=2, alpha=0.7, label='Path')

        # Plot pallets
        for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
            pi, pj = world_to_grid(px, py)
            rect_w = int(pl / cfg.resolution_m)
            rect_h = int(pw / cfg.resolution_m)
            rect = plt.Rectangle((pj - rect_w//2, pi - rect_h//2), rect_w, rect_h,
                               facecolor='red', alpha=0.7, edgecolor='darkred')
            ax.add_patch(rect)
            ax.text(pj, pi, f'P{i+1}', ha='center', va='center', color='white', fontweight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)

    # Corridor width profile
    x_positions = np.linspace(1.0, 19.0, 100)
    widths = []

    for x in x_positions:
        j = int(np.floor(x / cfg.resolution_m))
        if 0 <= j < grid_inflated.shape[1]:
            # Find top and bottom free cells
            top_free = -1
            bottom_free = -1

            for i in range(grid_inflated.shape[0]):
                if not grid_inflated[i, j]:
                    if top_free == -1:
                        top_free = i
                    bottom_free = i

            if top_free != -1 and bottom_free != -1:
                width = (bottom_free - top_free + 1) * cfg.resolution_m
                widths.append(width)
            else:
                widths.append(0)
        else:
            widths.append(0)

    ax3.plot(x_positions, widths, 'b-', linewidth=2)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Robot diameter')
    ax3.axhline(y=0.7, color='orange', linestyle='--', label='Min passage')
    ax3.set_xlabel('X position (m)')
    ax3.set_ylabel('Corridor width (m)')
    ax3.set_title('Corridor Width Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Robot path attempt
    ax4.imshow(grid_inflated, cmap='gray_r', origin='lower')
    ax4.plot(start_j, start_i, 'go', markersize=10, label='Start')
    ax4.plot(goal_j, goal_i, 'ro', markersize=10, label='Goal')

    # Try to find a path manually
    try_path_visualization(ax4, grid_inflated, start_pose, goal_xy, cfg.resolution_m)

    ax4.set_title('Path Attempt Visualization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/home/gong-zerui/code/ME5418-Project/seed_{seed}_deep_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nDetailed visualization saved as: seed_{seed}_deep_analysis.png")


def try_path_visualization(ax, grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                          goal_xy: Tuple[float, float], resolution_m: float):
    """Try to visualize a path attempt."""

    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_xy

    # Convert to grid coordinates
    start_i = int(np.floor(start_y / resolution_m))
    start_j = int(np.floor(start_x / resolution_m))
    goal_i = int(np.floor(goal_y / resolution_m))
    goal_j = int(np.floor(goal_x / resolution_m))

    H, W = grid_inflated.shape

    # Simple straight-line path attempt
    path_i = np.linspace(start_i, goal_i, 50).astype(int)
    path_j = np.linspace(start_j, goal_j, 50).astype(int)

    # Check which points are valid
    valid_points = []
    for i, j in zip(path_i, path_j):
        if 0 <= i < H and 0 <= j < W and not grid_inflated[i, j]:
            valid_points.append((i, j))
        else:
            break

    if valid_points:
        valid_i, valid_j = zip(*valid_points)
        ax.plot(valid_j, valid_i, 'g-', linewidth=3, alpha=0.8, label='Attempted path')
        ax.plot(valid_j[-1], valid_i[-1], 'yo', markersize=8, label='Path end')
    else:
        ax.plot(start_j, start_i, 'yo', markersize=8, label='No valid path')


if __name__ == "__main__":
    grid, grid_inflated, start_pose, goal_xy, waypoints, info = deep_analyze_seed_20021213()

    print(f"\n{'='*60}")
    print(f"CONCLUSION: Seed 20021213 creates a map where the robot cannot")
    print(f"navigate from start to goal due to corridor width constraints.")
    print(f"The issue is likely a combination of narrow corridor width")
    print(f"and pallet placement that creates bottlenecks.")
    print(f"{'='*60}")
