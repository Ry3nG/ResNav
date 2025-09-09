#!/usr/bin/env python3
"""Analyze the pallet placement logic bug that can create infeasible maps.

The issue is in the pallet placement algorithm where it doesn't guarantee
that at least one side always has sufficient passage width.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario


def analyze_pallet_placement_bug():
    """Analyze the specific bug in pallet placement logic."""

    print("Analyzing pallet placement logic bug")
    print("=" * 50)

    # Test with the problematic seed
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

    print(f"Corridor width: {info['corridor_width']:.2f}m")
    print(f"Min passage required: {cfg.min_passage_m:.2f}m")
    print(f"Number of pallets: {info['num_pallets']}")

    # Analyze each pallet's placement
    corridor_center_y = cfg.map_height_m / 2.0
    corridor_width = info['corridor_width']
    y_top = corridor_center_y + corridor_width / 2.0
    y_bot = corridor_center_y - corridor_width / 2.0

    print(f"\nCorridor bounds: top={y_top:.2f}m, bottom={y_bot:.2f}m")

    problematic_pallets = []

    for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
        print(f"\nPallet {i+1}: center=({px:.2f}, {py:.2f}), size=({pl:.2f}, {pw:.2f})")

        # Calculate actual passage widths
        pallet_top = py + pw/2
        pallet_bot = py - pw/2

        passage_above = y_top - pallet_top
        passage_below = pallet_bot - y_bot

        print(f"  Pallet bounds: top={pallet_top:.2f}m, bottom={pallet_bot:.2f}m")
        print(f"  Passage above: {passage_above:.2f}m")
        print(f"  Passage below: {passage_below:.2f}m")
        print(f"  Min required: {cfg.min_passage_m:.2f}m")

        # Check if both passages are too narrow
        if passage_above < cfg.min_passage_m and passage_below < cfg.min_passage_m:
            print(f"  ❌ PROBLEM: Both passages too narrow!")
            problematic_pallets.append(i+1)
        elif passage_above < cfg.min_passage_m:
            print(f"  ⚠️  Passage above too narrow")
        elif passage_below < cfg.min_passage_m:
            print(f"  ⚠️  Passage below too narrow")
        else:
            print(f"  ✅ Both passages sufficient")

    print(f"\nProblematic pallets: {problematic_pallets}")

    # Analyze the root cause
    print(f"\nRoot cause analysis:")
    print(f"1. Corridor width: {corridor_width:.2f}m")
    print(f"2. Min passage required: {cfg.min_passage_m:.2f}m")
    print(f"3. Available space for pallet: {corridor_width - cfg.min_passage_m:.2f}m")
    print(f"4. Max pallet width that guarantees passage: {corridor_width - cfg.min_passage_m:.2f}m")

    # Check if any pallet is too wide
    for i, ((px, py), (pl, pw)) in enumerate(zip(info['pallet_centers'], info['pallet_sizes'])):
        max_allowed_width = corridor_width - cfg.min_passage_m
        if pw > max_allowed_width:
            print(f"  Pallet {i+1} width {pw:.2f}m > max allowed {max_allowed_width:.2f}m")

    return problematic_pallets, info


def test_pallet_placement_algorithm():
    """Test the pallet placement algorithm with various seeds to find problematic cases."""

    print("\nTesting pallet placement algorithm with multiple seeds")
    print("=" * 60)

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

    problematic_seeds = []

    # Test seeds 0-1000
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

        corridor_width = info['corridor_width']
        corridor_center_y = cfg.map_height_m / 2.0
        y_top = corridor_center_y + corridor_width / 2.0
        y_bot = corridor_center_y - corridor_width / 2.0

        has_problem = False
        for (px, py), (pl, pw) in zip(info['pallet_centers'], info['pallet_sizes']):
            pallet_top = py + pw/2
            pallet_bot = py - pw/2

            passage_above = y_top - pallet_top
            passage_below = pallet_bot - y_bot

            if passage_above < cfg.min_passage_m and passage_below < cfg.min_passage_m:
                has_problem = True
                break

        if has_problem:
            problematic_seeds.append(seed)
            if len(problematic_seeds) <= 10:  # Show details for first 10
                print(f"Seed {seed}: corridor_width={corridor_width:.2f}m, pallets={len(info['pallet_centers'])}")

    print(f"\nFound {len(problematic_seeds)} problematic seeds out of 1000 tested")
    print(f"Problematic seeds: {problematic_seeds[:20]}...")  # Show first 20

    return problematic_seeds


def propose_fix():
    """Propose a fix for the pallet placement algorithm."""

    print("\nProposed fix for pallet placement algorithm")
    print("=" * 50)

    print("Current algorithm issues:")
    print("1. It calculates w_cap = max(w_min, min(pallet_w_eff, w_max_geom))")
    print("2. w_max_geom = corridor_w - T - eps")
    print("3. But it doesn't ensure that the chosen pallet width leaves enough space")
    print("4. The placement logic can still place a pallet that blocks both passages")

    print("\nProposed fix:")
    print("1. Calculate maximum pallet width that guarantees at least one passage >= min_passage_m")
    print("2. max_pallet_width = corridor_width - min_passage_m")
    print("3. Ensure pallet width <= max_pallet_width")
    print("4. When placing pallet, ensure at least one side has >= min_passage_m")

    print("\nCode changes needed in scenarios.py:")
    print("- Line 95-100: Fix w_cap calculation")
    print("- Line 105-117: Add validation that at least one passage is sufficient")
    print("- Add fallback logic if no valid placement found")


if __name__ == "__main__":
    # Analyze the specific problematic case
    problematic_pallets, info = analyze_pallet_placement_bug()

    # Test algorithm with multiple seeds
    problematic_seeds = test_pallet_placement_algorithm()

    # Propose fix
    propose_fix()

    print(f"\n{'='*60}")
    print(f"CONCLUSION: The pallet placement algorithm has a bug that can create")
    print(f"infeasible maps where both passages around a pallet are too narrow.")
    print(f"This affects approximately {len(problematic_seeds)/10:.1f}% of generated maps.")
    print(f"{'='*60}")
