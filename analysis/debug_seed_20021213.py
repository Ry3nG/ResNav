#!/usr/bin/env python3
"""Debug script to analyze seed 20021213 map feasibility.

This script generates the map for seed 20021213 and checks if there's a feasible path
from start to goal, analyzing the pallet placement and corridor configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid
from amr_env.sim.dynamics import UnicycleModel, UnicycleState


def analyze_map_feasibility(seed: int = 20021213):
    """Analyze if the map generated with given seed has a feasible path."""

    print(f"Analyzing map feasibility for seed {seed}")
    print("=" * 50)

    # Create scenario with same config as training
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

    # Generate map with specific seed
    rng = np.random.default_rng(seed)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    print(f"Map dimensions: {grid.shape}")
    print(f"Corridor width: {info['corridor_width']:.2f}m")
    print(f"Number of pallets: {info['num_pallets']}")
    print(f"Start pose: {start_pose}")
    print(f"Goal: {goal_xy}")
    print(f"Pallet centers: {info['pallet_centers']}")
    print(f"Pallet sizes: {info['pallet_sizes']}")

    # Create inflated grid (robot radius + safety margin)
    robot_radius = 0.25  # 0.5m diameter / 2
    safety_margin = 0.0
    grid_inflated = inflate_grid(grid, robot_radius + safety_margin, cfg.resolution_m)

    # Check if start and goal are in free space
    start_x, start_y, start_theta = start_pose
    goal_x, goal_y = goal_xy

    def point_in_inflated(x: float, y: float) -> bool:
        i = int(np.floor(y / cfg.resolution_m))
        j = int(np.floor(x / cfg.resolution_m))
        H, W = grid_inflated.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return bool(grid_inflated[i, j])

    start_collision = point_in_inflated(start_x, start_y)
    goal_collision = point_in_inflated(goal_x, goal_y)

    print(f"\nCollision checks:")
    print(f"Start collision: {start_collision}")
    print(f"Goal collision: {goal_collision}")

    # Analyze corridor and pallet placement
    corridor_center_y = cfg.map_height_m / 2.0
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    print(f"\nCorridor analysis:")
    print(f"Corridor center Y: {corridor_center_y:.2f}m")
    print(f"Corridor top: {y_top:.2f}m")
    print(f"Corridor bottom: {y_bot:.2f}m")
    print(f"Corridor width: {corridor_width:.2f}m")

    # Check pallet placement and passage widths
    print(f"\nPallet analysis:")
    for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
        print(f"Pallet {i+1}: center=({px:.2f}, {py:.2f}), size=({pl:.2f}, {pw:.2f})")

        # Calculate passage widths
        pallet_top = py + pw/2
        pallet_bot = py - pw/2

        passage_top = y_top - pallet_top
        passage_bot = pallet_bot - y_bot

        print(f"  Passage above: {passage_top:.2f}m")
        print(f"  Passage below: {passage_bot:.2f}m")
        print(f"  Min required: {cfg.min_passage_m:.2f}m")

        if passage_top < cfg.min_passage_m and passage_bot < cfg.min_passage_m:
            print(f"  ⚠️  WARNING: Both passages too narrow!")

    # Simple pathfinding check using A* on grid
    feasible = check_path_feasibility(grid_inflated, start_pose, goal_xy, cfg.resolution_m)
    print(f"\nPath feasibility: {'✅ FEASIBLE' if feasible else '❌ NOT FEASIBLE'}")

    # Visualize the map
    visualize_map(grid, grid_inflated, start_pose, goal_xy, waypoints, info, cfg)

    return feasible, grid, grid_inflated, start_pose, goal_xy, waypoints, info


def check_path_feasibility(grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                          goal_xy: Tuple[float, float], resolution_m: float) -> bool:
    """Simple A* pathfinding to check if path exists."""
    from collections import deque

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


def visualize_map(grid: np.ndarray, grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                 goal_xy: Tuple[float, float], waypoints: np.ndarray, info: dict, cfg: BlockageScenarioConfig):
    """Visualize the generated map."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Raw grid
    ax1.imshow(grid, cmap='gray_r', origin='lower')
    ax1.set_title('Raw Grid (True=occupied)')
    ax1.set_xlabel('X (grid cells)')
    ax1.set_ylabel('Y (grid cells)')

    # Inflated grid
    ax2.imshow(grid_inflated, cmap='gray_r', origin='lower')
    ax2.set_title('Inflated Grid (robot + safety)')
    ax2.set_xlabel('X (grid cells)')
    ax2.set_ylabel('Y (grid cells)')

    # Convert world coordinates to grid coordinates
    def world_to_grid(x, y):
        i = int(np.floor(y / cfg.resolution_m))
        j = int(np.floor(x / cfg.resolution_m))
        return i, j

    # Plot start and goal
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
        for (px, py), (pl, pw) in zip(info['pallet_centers'], info['pallet_sizes']):
            pi, pj = world_to_grid(px, py)
            # Draw rectangle
            rect_w = int(pl / cfg.resolution_m)
            rect_h = int(pw / cfg.resolution_m)
            rect = plt.Rectangle((pj - rect_w//2, pi - rect_h//2), rect_w, rect_h,
                               facecolor='red', alpha=0.7, edgecolor='darkred')
            ax.add_patch(rect)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/home/gong-zerui/code/ME5418-Project/seed_{seed}_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nVisualization saved as: seed_{seed}_analysis.png")


if __name__ == "__main__":
    seed = 20021213
    feasible, grid, grid_inflated, start_pose, goal_xy, waypoints, info = analyze_map_feasibility(seed)

    print(f"\n{'='*50}")
    print(f"FINAL RESULT: Map with seed {seed} is {'FEASIBLE' if feasible else 'NOT FEASIBLE'}")
    print(f"{'='*50}")
