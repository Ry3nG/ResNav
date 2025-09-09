#!/usr/bin/env python3
"""Test the actual retry behavior for seed 20021213.

This script tests whether seed 20021213 generates a feasible scenario
on the first try, or if it requires retries.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenario_manager import ScenarioManager
from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario


def test_seed_20021213_retry_behavior():
    """Test the actual retry behavior for seed 20021213."""

    print("Testing seed 20021213 retry behavior")
    print("=" * 50)

    # Environment configuration
    env_cfg = {
        "map": {
            "size_m": [20.0, 20.0],
            "resolution_m": 0.2,
            "corridor_width_m": [3.0, 5.0],
            "wall_thickness_m": 0.3,
            "pallet_width_m": 1.2,
            "pallet_length_m": 1.0,
            "start_x_m": 1.0,
            "goal_margin_x_m": 1.0,
            "waypoint_step_m": 0.3,
            "min_passage_m": 0.7,
            "min_pallet_x_offset_m": 1.0,
            "num_pallets_min": 3,
            "num_pallets_max": 5,
        }
    }

    # Test 1: Original scenario generation (without retry)
    print("1. Testing original scenario generation (no retry):")
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

    # Check if this original scenario is feasible
    scenario_manager = ScenarioManager(env_cfg)
    is_feasible_original = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)

    print(f"   Original seed 20021213 scenario:")
    print(f"     Feasible: {is_feasible_original}")
    print(f"     Corridor width: {info['corridor_width']:.2f}m")
    print(f"     Number of pallets: {info['num_pallets']}")

    # Test 2: With retry mechanism
    print("\n2. Testing with retry mechanism:")
    scenario_manager.set_seed(20021213)

    # Track retry attempts
    original_rng_state = scenario_manager._rng.bit_generator.state.copy()

    grid, waypoints, start_pose, goal_xy, info = scenario_manager.sample()

    # Check if RNG state changed (indicating retries)
    final_rng_state = scenario_manager._rng.bit_generator.state.copy()
    rng_changed = original_rng_state != final_rng_state

    is_feasible_retry = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)

    print(f"   With retry mechanism:")
    print(f"     Feasible: {is_feasible_retry}")
    print(f"     RNG state changed: {rng_changed}")
    print(f"     Corridor width: {info['corridor_width']:.2f}m")
    print(f"     Number of pallets: {info['num_pallets']}")

    # Test 3: Multiple runs to see retry patterns
    print("\n3. Testing multiple runs to see retry patterns:")
    retry_counts = []

    for i in range(10):
        scenario_manager.set_seed(20021213)
        original_state = scenario_manager._rng.bit_generator.state.copy()

        grid, waypoints, start_pose, goal_xy, info = scenario_manager.sample()

        final_state = scenario_manager._rng.bit_generator.state.copy()
        retry_count = 0 if original_state == final_state else 1
        retry_counts.append(retry_count)

        is_feasible = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)
        print(f"   Run {i+1}: Feasible={is_feasible}, Retry={retry_count}")

    print(f"\n   Retry statistics:")
    print(f"     Average retries: {np.mean(retry_counts):.2f}")
    print(f"     Runs requiring retry: {sum(retry_counts)}/10")

    return is_feasible_original, is_feasible_retry, retry_counts


if __name__ == "__main__":
    original_feasible, retry_feasible, retry_counts = test_seed_20021213_retry_behavior()

    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"  Original seed 20021213 feasible: {original_feasible}")
    print(f"  With retry mechanism feasible: {retry_feasible}")
    print(f"  Retry rate: {sum(retry_counts)}/10 runs")

    if not original_feasible and retry_feasible:
        print("\n✅ Seed 20021213 requires retries to generate feasible scenario")
        print("   The retry mechanism is working correctly!")
    elif original_feasible:
        print("\n✅ Seed 20021213 generates feasible scenario on first try")
        print("   No retries needed!")
    else:
        print("\n❌ Seed 20021213 still generates infeasible scenarios even with retries")
        print("   The fix may need further refinement")
