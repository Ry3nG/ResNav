#!/usr/bin/env python3
"""Scenario prefiltering system for training.

This module provides a prefiltering system that:
- Validates scenarios before training
- Filters out infeasible scenarios
- Provides quality metrics for scenario selection
- Implements curriculum learning strategies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import time

from amr_env.sim.scenarios import BlockageScenarioConfig, create_blockage_scenario
from amr_env.sim.collision import inflate_grid
from improved_pathfinding import ImprovedPathfinder
from improved_scenario_generator import ImprovedScenarioGenerator, ScenarioQualityMetrics


@dataclass
class PrefilterConfig:
    """Configuration for scenario prefiltering."""
    max_retries: int = 10
    min_feasibility_rate: float = 0.95
    quality_thresholds: Dict[str, float] = None
    curriculum_enabled: bool = True
    difficulty_levels: int = 5
    cache_size: int = 1000


class ScenarioPrefilter:
    """Prefiltering system for training scenarios."""

    def __init__(self, config: PrefilterConfig, robot_radius: float = 0.25,
                 safety_margin: float = 0.1, resolution: float = 0.2):
        """Initialize the prefilter.

        Args:
            config: Prefilter configuration
            robot_radius: Robot radius in meters
            safety_margin: Safety margin in meters
            resolution: Grid resolution in meters per cell
        """
        self.config = config
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.resolution = resolution

        # Initialize components
        self.pathfinder = ImprovedPathfinder(
            robot_radius=robot_radius,
            min_turning_radius=0.5,
            resolution=resolution,
            safety_margin=safety_margin
        )

        self.generator = ImprovedScenarioGenerator(
            robot_radius=robot_radius,
            safety_margin=safety_margin,
            resolution=resolution,
            max_retries=config.max_retries
        )

        # Quality thresholds
        if config.quality_thresholds is None:
            config.quality_thresholds = {
                'min_passage_width': 0.6,
                'max_corridor_utilization': 0.4,
                'max_difficulty_score': 0.8
            }

        # Statistics
        self.stats = {
            'total_generated': 0,
            'feasible_generated': 0,
            'filtered_out': 0,
            'quality_rejected': 0,
            'generation_time': 0.0,
            'validation_time': 0.0
        }

        # Cache for validated scenarios
        self.scenario_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def generate_validated_scenario(self, cfg: BlockageScenarioConfig,
                                  rng: Optional[np.random.Generator] = None,
                                  difficulty_level: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray, Tuple[float, float, float], Tuple[float, float],
        Dict[str, float], ScenarioQualityMetrics, bool
    ]:
        """Generate a validated scenario for training.

        Args:
            cfg: Scenario configuration
            rng: Random number generator
            difficulty_level: Difficulty level for curriculum learning

        Returns:
            (grid, waypoints, start_pose, goal_xy, info, quality_metrics, is_cached)
        """

        start_time = time.time()

        # Check cache first (simplified - no caching for now)
        # if rng is not None:
        #     # Create a hashable cache key
        #     cfg_key = (cfg.map_width_m, cfg.map_height_m, cfg.corridor_width_min_m,
        #               cfg.corridor_width_max_m, cfg.num_pallets_min, cfg.num_pallets_max)
        #     rng_state = tuple(rng.bit_generator.state['state']['key'])
        #     cache_key = (cfg_key, rng_state)
        #     if cache_key in self.scenario_cache:
        #         self.cache_hits += 1
        #         return self.scenario_cache[cache_key] + (True,)

        self.cache_misses += 1

        # Generate scenario
        rng = rng or np.random.default_rng()

        # Apply curriculum learning if enabled
        if self.config.curriculum_enabled and difficulty_level is not None:
            cfg = self._adjust_config_for_difficulty(cfg, difficulty_level)

        # Generate with validation
        grid, waypoints, start_pose, goal_xy, info, quality = self.generator.generate_feasible_scenario(cfg, rng)

        # Validate quality
        is_valid = self._validate_scenario_quality(quality)

        if not is_valid:
            self.stats['quality_rejected'] += 1
            # Try to improve the scenario
            grid, waypoints, start_pose, goal_xy, info, quality = self._improve_scenario_quality(
                grid, waypoints, start_pose, goal_xy, info, quality, cfg, rng
            )
            is_valid = self._validate_scenario_quality(quality)

        # Update statistics
        self.stats['total_generated'] += 1
        if quality.feasible:
            self.stats['feasible_generated'] += 1
        else:
            self.stats['filtered_out'] += 1

        generation_time = time.time() - start_time
        self.stats['generation_time'] += generation_time

        # Cache the result (simplified - no caching for now)
        # if rng is not None:
        #     cfg_key = (cfg.map_width_m, cfg.map_height_m, cfg.corridor_width_min_m,
        #               cfg.corridor_width_max_m, cfg.num_pallets_min, cfg.num_pallets_max)
        #     rng_state = tuple(rng.bit_generator.state['state']['key'])
        #     cache_key = (cfg_key, rng_state)
        #     if len(self.scenario_cache) < self.config.cache_size:
        #         self.scenario_cache[cache_key] = (grid, waypoints, start_pose, goal_xy, info, quality)

        return grid, waypoints, start_pose, goal_xy, info, quality, False

    def _adjust_config_for_difficulty(self, cfg: BlockageScenarioConfig,
                                    difficulty_level: int) -> BlockageScenarioConfig:
        """Adjust configuration based on difficulty level."""

        # Normalize difficulty level to [0, 1]
        difficulty = difficulty_level / (self.config.difficulty_levels - 1)

        # Adjust parameters based on difficulty
        new_cfg = BlockageScenarioConfig(
            map_width_m=cfg.map_width_m,
            map_height_m=cfg.map_height_m,
            corridor_width_min_m=cfg.corridor_width_min_m + difficulty * 1.0,  # Wider corridors for higher difficulty
            corridor_width_max_m=cfg.corridor_width_max_m + difficulty * 1.0,
            wall_thickness_m=cfg.wall_thickness_m,
            pallet_width_m=cfg.pallet_width_m + difficulty * 0.3,  # Larger pallets for higher difficulty
            pallet_length_m=cfg.pallet_length_m + difficulty * 0.2,
            start_x_m=cfg.start_x_m,
            goal_margin_x_m=cfg.goal_margin_x_m,
            waypoint_step_m=cfg.waypoint_step_m,
            resolution_m=cfg.resolution_m,
            min_passage_m=cfg.min_passage_m,
            min_pallet_x_offset_m=cfg.min_pallet_x_offset_m,
            num_pallets_min=cfg.num_pallets_min + int(difficulty * 2),  # More pallets for higher difficulty
            num_pallets_max=cfg.num_pallets_max + int(difficulty * 2),
        )

        return new_cfg

    def _validate_scenario_quality(self, quality: ScenarioQualityMetrics) -> bool:
        """Validate scenario quality against thresholds."""

        thresholds = self.config.quality_thresholds

        # Check feasibility
        if not quality.feasible:
            return False

        # Check minimum passage width
        if quality.min_passage_width < thresholds['min_passage_width']:
            return False

        # Check corridor utilization
        if quality.corridor_utilization > thresholds['max_corridor_utilization']:
            return False

        # Check difficulty score
        if quality.difficulty_score > thresholds['max_difficulty_score']:
            return False

        return True

    def _improve_scenario_quality(self, grid: np.ndarray, waypoints: np.ndarray,
                                 start_pose: Tuple[float, float, float], goal_xy: Tuple[float, float],
                                 info: Dict[str, float], quality: ScenarioQualityMetrics,
                                 cfg: BlockageScenarioConfig, rng: np.random.Generator) -> Tuple[
        np.ndarray, np.ndarray, Tuple[float, float, float], Tuple[float, float],
        Dict[str, float], ScenarioQualityMetrics
    ]:
        """Try to improve scenario quality."""

        # Strategy 1: Remove pallets if corridor utilization is too high
        if quality.corridor_utilization > self.config.quality_thresholds['max_corridor_utilization']:
            # Remove some pallets
            pallet_centers = info['pallet_centers']
            pallet_sizes = info['pallet_sizes']

            if len(pallet_centers) > 1:
                # Remove the largest pallet
                largest_idx = max(range(len(pallet_sizes)), key=lambda i: pallet_sizes[i][0] * pallet_sizes[i][1])
                new_centers = pallet_centers[:largest_idx] + pallet_centers[largest_idx+1:]
                new_sizes = pallet_sizes[:largest_idx] + pallet_sizes[largest_idx+1:]

                info['pallet_centers'] = new_centers
                info['pallet_sizes'] = new_sizes
                info['num_pallets'] = len(new_centers)

                # Regenerate grid
                grid = self.generator._regenerate_grid_without_pallets(grid, new_centers, new_sizes, cfg)

        # Recalculate quality metrics
        grid_inflated = inflate_grid(grid, self.robot_radius + self.safety_margin, self.resolution)
        success, path, path_info = self.pathfinder.find_path(grid_inflated, start_pose, goal_xy)

        if success:
            quality = self.generator._calculate_quality_metrics(
                grid, grid_inflated, start_pose, goal_xy, waypoints, info, path
            )
        else:
            quality = self.generator._calculate_quality_metrics(
                grid, grid_inflated, start_pose, goal_xy, waypoints, info, None
            )

        return grid, waypoints, start_pose, goal_xy, info, quality

    def get_statistics(self) -> Dict[str, float]:
        """Get prefilter statistics."""

        stats = self.stats.copy()

        # Calculate rates
        if stats['total_generated'] > 0:
            stats['feasibility_rate'] = stats['feasible_generated'] / stats['total_generated']
            stats['avg_generation_time'] = stats['generation_time'] / stats['total_generated']
        else:
            stats['feasibility_rate'] = 0.0
            stats['avg_generation_time'] = 0.0

        # Cache statistics
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = self.cache_hits / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """Reset prefilter statistics."""
        self.stats = {
            'total_generated': 0,
            'feasible_generated': 0,
            'filtered_out': 0,
            'quality_rejected': 0,
            'generation_time': 0.0,
            'validation_time': 0.0
        }
        self.cache_hits = 0
        self.cache_misses = 0


def test_scenario_prefilter():
    """Test the scenario prefilter system."""

    print("Testing Scenario Prefilter System")
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

    # Prefilter configuration
    prefilter_config = PrefilterConfig(
        max_retries=5,
        min_feasibility_rate=0.95,
        quality_thresholds={
            'min_passage_width': 0.6,
            'max_corridor_utilization': 0.4,
            'max_difficulty_score': 0.8
        },
        curriculum_enabled=True,
        difficulty_levels=5,
        cache_size=100
    )

    prefilter = ScenarioPrefilter(prefilter_config)

    # Test different difficulty levels
    for difficulty in range(5):
        print(f"\nTesting difficulty level {difficulty}/4:")

        valid_count = 0
        total_tests = 10

        for i in range(total_tests):
            rng = np.random.default_rng(i + difficulty * 100)
            grid, waypoints, start_pose, goal_xy, info, quality, is_cached = prefilter.generate_validated_scenario(
                cfg, rng, difficulty
            )

            if quality.feasible and _validate_scenario_quality(quality, prefilter_config.quality_thresholds):
                valid_count += 1

        print(f"  Valid scenarios: {valid_count}/{total_tests} ({valid_count/total_tests*100:.1f}%)")

    # Print statistics
    stats = prefilter.get_statistics()
    print(f"\nPrefilter Statistics:")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Feasible generated: {stats['feasible_generated']}")
    print(f"  Feasibility rate: {stats['feasibility_rate']:.3f}")
    print(f"  Quality rejected: {stats['quality_rejected']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.3f}")
    print(f"  Avg generation time: {stats['avg_generation_time']:.3f}s")


def _validate_scenario_quality(quality: ScenarioQualityMetrics, thresholds: Dict[str, float]) -> bool:
    """Helper function to validate scenario quality."""

    # Check feasibility
    if not quality.feasible:
        return False

    # Check minimum passage width
    if quality.min_passage_width < thresholds['min_passage_width']:
        return False

    # Check corridor utilization
    if quality.corridor_utilization > thresholds['max_corridor_utilization']:
        return False

    # Check difficulty score
    if quality.difficulty_score > thresholds['max_difficulty_score']:
        return False

    return True


if __name__ == "__main__":
    test_scenario_prefilter()
