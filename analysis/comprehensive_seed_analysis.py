#!/usr/bin/env python3
"""Comprehensive analysis of seed feasibility across a large range.

This script tests seeds from 0 to 10000 to quantify the proportion of
infeasible maps and identify patterns in problematic seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os
from collections import defaultdict
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid


def test_seed_feasibility(seed: int, cfg: BlockageScenarioConfig) -> Dict:
    """Test if a single seed produces a feasible map."""

    rng = np.random.default_rng(seed)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    # Create inflated grid
    robot_radius = 0.25
    safety_margin = 0.0
    grid_inflated = inflate_grid(grid, robot_radius + safety_margin, cfg.resolution_m)

    # Check path feasibility
    feasible = check_path_feasibility(grid_inflated, start_pose, goal_xy, cfg.resolution_m)

    # Analyze corridor and pallet characteristics
    corridor_width = info['corridor_width']
    num_pallets = info['num_pallets']

    # Calculate minimum passage width
    corridor_center_y = cfg.map_height_m / 2.0
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    min_passage = float('inf')
    for (px, py), (pl, pw) in zip(info['pallet_centers'], info['pallet_sizes']):
        pallet_top = py + pw/2
        pallet_bot = py - pw/2

        passage_above = y_top - pallet_top
        passage_below = pallet_bot - y_bot

        min_passage = min(min_passage, passage_above, passage_below)

    return {
        'seed': seed,
        'feasible': feasible,
        'corridor_width': corridor_width,
        'num_pallets': num_pallets,
        'min_passage': min_passage,
        'pallet_centers': info['pallet_centers'],
        'pallet_sizes': info['pallet_sizes']
    }


def check_path_feasibility(grid_inflated: np.ndarray, start_pose: Tuple[float, float, float],
                          goal_xy: Tuple[float, float], resolution_m: float) -> bool:
    """Check if path exists using A* algorithm."""

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


def analyze_infeasible_patterns(results: List[Dict]) -> Dict:
    """Analyze patterns in infeasible maps."""

    infeasible = [r for r in results if not r['feasible']]
    feasible = [r for r in results if r['feasible']]

    print(f"Total seeds tested: {len(results)}")
    print(f"Feasible: {len(feasible)} ({len(feasible)/len(results)*100:.1f}%)")
    print(f"Infeasible: {len(infeasible)} ({len(infeasible)/len(results)*100:.1f}%)")

    if not infeasible:
        return {}

    # Analyze corridor width distribution
    feasible_cw = [r['corridor_width'] for r in feasible]
    infeasible_cw = [r['corridor_width'] for r in infeasible]

    print(f"\nCorridor width analysis:")
    print(f"  Feasible maps - mean: {np.mean(feasible_cw):.3f}, std: {np.std(feasible_cw):.3f}")
    print(f"  Infeasible maps - mean: {np.mean(infeasible_cw):.3f}, std: {np.std(infeasible_cw):.3f}")

    # Analyze minimum passage width
    feasible_mp = [r['min_passage'] for r in feasible]
    infeasible_mp = [r['min_passage'] for r in infeasible]

    print(f"\nMinimum passage width analysis:")
    print(f"  Feasible maps - mean: {np.mean(feasible_mp):.3f}, std: {np.std(feasible_mp):.3f}")
    print(f"  Infeasible maps - mean: {np.mean(infeasible_mp):.3f}, std: {np.std(infeasible_mp):.3f}")

    # Analyze pallet count
    feasible_pc = [r['num_pallets'] for r in feasible]
    infeasible_pc = [r['num_pallets'] for r in infeasible]

    print(f"\nPallet count analysis:")
    print(f"  Feasible maps - mean: {np.mean(feasible_pc):.1f}, std: {np.std(feasible_pc):.1f}")
    print(f"  Infeasible maps - mean: {np.mean(infeasible_pc):.1f}, std: {np.std(infeasible_pc):.1f}")

    # Find problematic seed ranges
    infeasible_seeds = [r['seed'] for r in infeasible]
    print(f"\nInfeasible seed ranges:")
    ranges = []
    start = infeasible_seeds[0]
    end = infeasible_seeds[0]

    for i in range(1, len(infeasible_seeds)):
        if infeasible_seeds[i] == end + 1:
            end = infeasible_seeds[i]
        else:
            ranges.append((start, end))
            start = infeasible_seeds[i]
            end = infeasible_seeds[i]
    ranges.append((start, end))

    for start, end in ranges:
        if start == end:
            print(f"  Seed {start}")
        else:
            print(f"  Seeds {start}-{end} ({end-start+1} seeds)")

    return {
        'infeasible_count': len(infeasible),
        'feasible_count': len(feasible),
        'infeasible_rate': len(infeasible) / len(results),
        'corridor_width_stats': {
            'feasible': {'mean': np.mean(feasible_cw), 'std': np.std(feasible_cw)},
            'infeasible': {'mean': np.mean(infeasible_cw), 'std': np.std(infeasible_cw)}
        },
        'min_passage_stats': {
            'feasible': {'mean': np.mean(feasible_mp), 'std': np.std(feasible_mp)},
            'infeasible': {'mean': np.mean(infeasible_mp), 'std': np.std(infeasible_mp)}
        },
        'pallet_count_stats': {
            'feasible': {'mean': np.mean(feasible_pc), 'std': np.std(feasible_pc)},
            'infeasible': {'mean': np.mean(infeasible_pc), 'std': np.std(infeasible_pc)}
        },
        'infeasible_seeds': infeasible_seeds,
        'infeasible_ranges': ranges
    }


def create_visualization(results: List[Dict], analysis: Dict):
    """Create comprehensive visualizations of the analysis."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Feasibility rate over seed range
    seeds = [r['seed'] for r in results]
    feasible = [1 if r['feasible'] else 0 for r in results]

    ax1.plot(seeds, feasible, 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Feasible (1) / Infeasible (0)')
    ax1.set_title('Feasibility by Seed')
    ax1.grid(True, alpha=0.3)

    # 2. Corridor width distribution
    feasible_cw = [r['corridor_width'] for r in results if r['feasible']]
    infeasible_cw = [r['corridor_width'] for r in results if not r['feasible']]

    ax2.hist(feasible_cw, bins=30, alpha=0.7, label='Feasible', color='green')
    ax2.hist(infeasible_cw, bins=30, alpha=0.7, label='Infeasible', color='red')
    ax2.set_xlabel('Corridor Width (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Corridor Width Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Minimum passage width distribution
    feasible_mp = [r['min_passage'] for r in results if r['feasible']]
    infeasible_mp = [r['min_passage'] for r in results if not r['feasible']]

    ax3.hist(feasible_mp, bins=30, alpha=0.7, label='Feasible', color='green')
    ax3.hist(infeasible_mp, bins=30, alpha=0.7, label='Infeasible', color='red')
    ax3.axvline(x=0.5, color='black', linestyle='--', label='Robot diameter')
    ax3.set_xlabel('Minimum Passage Width (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Minimum Passage Width Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Pallet count distribution
    feasible_pc = [r['num_pallets'] for r in results if r['feasible']]
    infeasible_pc = [r['num_pallets'] for r in results if not r['feasible']]

    ax4.hist(feasible_pc, bins=range(1, 7), alpha=0.7, label='Feasible', color='green')
    ax4.hist(infeasible_pc, bins=range(1, 7), alpha=0.7, label='Infeasible', color='red')
    ax4.set_xlabel('Number of Pallets')
    ax4.set_ylabel('Count')
    ax4.set_title('Pallet Count Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/gong-zerui/code/ME5418-Project/comprehensive_seed_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_results_to_csv(results: List[Dict], filename: str):
    """Save results to CSV file for further analysis."""

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'feasible', 'corridor_width', 'num_pallets', 'min_passage'])
        writer.writeheader()

        for result in results:
            writer.writerow({
                'seed': result['seed'],
                'feasible': result['feasible'],
                'corridor_width': result['corridor_width'],
                'num_pallets': result['num_pallets'],
                'min_passage': result['min_passage']
            })


def main():
    """Main analysis function."""

    print("Comprehensive Seed Feasibility Analysis")
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

    # Test seeds from 0 to 10000
    print("Testing seeds from 0 to 10000...")
    results = []

    for seed in range(10001):
        if seed % 1000 == 0:
            print(f"  Progress: {seed}/10000")

        result = test_seed_feasibility(seed, cfg)
        results.append(result)

    print("Analysis complete!")

    # Analyze patterns
    analysis = analyze_infeasible_patterns(results)

    # Create visualizations
    create_visualization(results, analysis)

    # Save results
    save_results_to_csv(results, '/home/gong-zerui/code/ME5418-Project/seed_feasibility_results.csv')

    print(f"\nResults saved to: seed_feasibility_results.csv")
    print(f"Visualization saved to: comprehensive_seed_analysis.png")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
