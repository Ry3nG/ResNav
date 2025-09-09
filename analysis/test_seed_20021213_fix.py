#!/usr/bin/env python3
"""Test the fix for seed 20021213 problem.

This script tests that the improved scenario manager can handle
the problematic seed 20021213 without generating infeasible maps.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenario_manager import ScenarioManager


def test_seed_20021213_fix():
    """Test that seed 20021213 now generates feasible scenarios."""

    print("Testing seed 20021213 fix")
    print("=" * 40)

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

    # Create scenario manager
    scenario_manager = ScenarioManager(env_cfg)

    # Test the problematic seed
    problematic_seed = 20021213
    scenario_manager.set_seed(problematic_seed)

    print(f"Testing seed {problematic_seed}...")

    # Generate scenario
    grid, waypoints, start_pose, goal_xy, info = scenario_manager.sample()

    print(f"Generated scenario:")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Start pose: {start_pose}")
    print(f"  Goal: {goal_xy}")
    print(f"  Corridor width: {info['corridor_width']:.2f}m")
    print(f"  Number of pallets: {info['num_pallets']}")

    # Test path feasibility
    is_feasible = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)
    print(f"  Path feasible: {is_feasible}")

    if is_feasible:
        print("✅ SUCCESS: Seed 20021213 now generates feasible scenarios!")
    else:
        print("❌ FAILURE: Seed 20021213 still generates infeasible scenarios")

    return is_feasible


def test_multiple_problematic_seeds():
    """Test multiple known problematic seeds."""

    print("\nTesting multiple problematic seeds")
    print("=" * 40)

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

    # Create scenario manager
    scenario_manager = ScenarioManager(env_cfg)

    # Test multiple seeds
    test_seeds = [20021213, 3135, 3141, 3500, 5000, 8000]
    feasible_count = 0

    for seed in test_seeds:
        scenario_manager.set_seed(seed)
        grid, waypoints, start_pose, goal_xy, info = scenario_manager.sample()

        is_feasible = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)
        print(f"Seed {seed}: {'✅ Feasible' if is_feasible else '❌ Infeasible'}")

        if is_feasible:
            feasible_count += 1

    print(f"\nSummary: {feasible_count}/{len(test_seeds)} seeds generated feasible scenarios")
    return feasible_count == len(test_seeds)


def test_retry_mechanism():
    """Test that the retry mechanism works."""

    print("\nTesting retry mechanism")
    print("=" * 40)

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

    # Create scenario manager
    scenario_manager = ScenarioManager(env_cfg)

    # Test with a seed that might need retries
    scenario_manager.set_seed(42)

    # Generate multiple scenarios to see retry behavior
    feasible_count = 0
    total_tests = 10

    for i in range(total_tests):
        grid, waypoints, start_pose, goal_xy, info = scenario_manager.sample()
        is_feasible = scenario_manager._is_path_feasible(grid, start_pose, goal_xy, 0.2)

        if is_feasible:
            feasible_count += 1

    print(f"Generated {feasible_count}/{total_tests} feasible scenarios")
    print(f"Success rate: {feasible_count/total_tests*100:.1f}%")

    return feasible_count >= total_tests * 0.8  # At least 80% success rate


if __name__ == "__main__":
    print("Testing Scenario Manager Fix for Seed 20021213")
    print("=" * 60)

    # Run tests
    test1_passed = test_seed_20021213_fix()
    test2_passed = test_multiple_problematic_seeds()
    test3_passed = test_retry_mechanism()

    print(f"\n{'='*60}")
    print("TEST RESULTS:")
    print(f"  Seed 20021213 fix: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Multiple seeds test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"  Retry mechanism: {'✅ PASSED' if test3_passed else '❌ FAILED'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
    else:
        print("\n⚠️  Some tests failed. The fix may need further refinement.")
