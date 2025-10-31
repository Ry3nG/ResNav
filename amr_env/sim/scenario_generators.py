"""Base scenario generator classes using composition pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import numpy as np


@dataclass
class BlockageScenarioConfig:
    """Config for a simple corridor with temporary blockages (pallets)."""

    map_width_m: float = 50.0
    map_height_m: float = 50.0
    corridor_width_min_m: float = 3.0
    corridor_width_max_m: float = 4.0
    wall_thickness_m: float = 0.3
    pallet_width_m: float = 1.1
    pallet_length_m: float = 0.6
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.3
    resolution_m: float = 0.2
    min_passage_m: float = 0.7  # e.g., robot_diameter + 0.2 (0.5 + 0.2)
    min_pallet_x_offset_m: float = 0.6
    num_pallets_min: int = 1
    num_pallets_max: int = 1


@dataclass
class OMCFConfig:
    map_width_m: float = 20.0
    map_height_m: float = 20.0
    corridor_width_min_m: float = 3.5
    corridor_width_max_m: float = 4.0
    wall_thickness_m: float = 0.30
    start_x_m: float = 1.0
    goal_margin_x_m: float = 1.0
    waypoint_step_m: float = 0.30
    resolution_m: float = 0.05
    pallet_width_m: float = 1.1
    pallet_length_m: float = 2.0
    num_pallets_min: int = 1
    num_pallets_max: int = 3
    min_passage_m: float = 1.3
    small_length_range_m: Tuple[float, float] = (1.0, 1.2)
    small_width_range_m: Tuple[float, float] = (1.0, 1.2)
    large_length_range_m: Tuple[float, float] = (1.8, 2.2)
    large_width_range_m: Tuple[float, float] = (1.1, 1.3)
    large_fraction: float = 0.4
    holes_enabled: bool = True
    holes_count_pairs: int = 1
    holes_x_lo_m: float = 14.0
    holes_x_hi_m: float = 17.5
    holes_open_len_m: float = 1.6
    holes_min_spacing_m: float = 1.5
    holes_pair_x_candidates: Tuple[float, ...] = ()


class BaseCorridorGenerator(ABC):
    """Base class for corridor-based scenario generators."""

    def __init__(self, cfg: Any, rng: np.random.Generator) -> None:
        """Initialize generator with config and RNG.

        Args:
            cfg: Scenario configuration dataclass
            rng: NumPy random generator
        """
        self.cfg = cfg
        self.rng = rng
        self._resolution = float(cfg.resolution_m)

    @abstractmethod
    def generate(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float, float],
        tuple[float, float],
        dict[str, Any],
    ]:
        """Generate scenario and return grid, waypoints, start, goal, info."""
        pass

    def _create_grid(self, width_m: float, height_m: float) -> np.ndarray:
        """Create empty occupancy grid."""
        H = int(round(height_m / self._resolution))
        W = int(round(width_m / self._resolution))
        return np.zeros((H, W), dtype=bool)

    def _fill_rect(
        self, grid: np.ndarray, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        """Fill rectangle in grid (in-place).

        Args:
            grid: Occupancy grid to modify
            x0, y0, x1, y1: Rectangle corners in meters
        """
        H, W = grid.shape
        i0 = int(np.clip(math.floor(y0 / self._resolution), 0, H - 1))
        i1 = int(np.clip(math.floor(y1 / self._resolution), 0, H - 1))
        j0 = int(np.clip(math.floor(x0 / self._resolution), 0, W - 1))
        j1 = int(np.clip(math.floor(x1 / self._resolution), 0, W - 1))
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        grid[i0 : i1 + 1, j0 : j1 + 1] = True

    def _draw_corridor_walls(
        self,
        grid: np.ndarray,
        y_top: float,
        y_bot: float,
        wall_thickness_m: float,
        map_width_m: float,
    ) -> None:
        """Draw solid corridor walls (in-place).

        Args:
            grid: Occupancy grid to modify
            y_top: Top edge of corridor (meters)
            y_bot: Bottom edge of corridor (meters)
            wall_thickness_m: Wall thickness (meters)
            map_width_m: Map width (meters)
        """
        self._fill_rect(grid, 0.0, y_top, map_width_m, y_top + wall_thickness_m)
        self._fill_rect(grid, 0.0, y_bot - wall_thickness_m, map_width_m, y_bot)

    @abstractmethod
    def _sample_pallet_dims(self, corridor_w: float) -> tuple[float, float]:
        """Sample pallet dimensions (length, width).

        Hook method for subclasses to customize pallet sizing.

        Args:
            corridor_w: Corridor width for constraint checking

        Returns:
            (length_m, width_m) tuple
        """
        pass

    def _place_pallets(
        self,
        grid: np.ndarray,
        corridor_w: float,
        y_bot: float,
        y_top: float,
        x_lo: float,
        x_hi: float,
        cy: float,
        num_pallets: int,
        min_passage_m: float,
    ) -> list[tuple[float, float, float, float]]:
        """Place pallets in corridor using configured sizing strategy.

        Args:
            grid: Occupancy grid to modify
            corridor_w: Corridor width (meters)
            y_bot: Bottom corridor edge (meters)
            y_top: Top corridor edge (meters)
            x_lo: Left placement bound (meters)
            x_hi: Right placement bound (meters)
            cy: Corridor centerline y (meters)
            num_pallets: Number of pallets to place
            min_passage_m: Minimum passage width (meters)

        Returns:
            List of (x_center, y_center, length, width) tuples
        """
        pallets = []
        for _ in range(num_pallets):
            length, width = self._sample_pallet_dims(corridor_w)

            # Clamp width to feasible corridor bounds
            max_width = max(0.2, corridor_w - min_passage_m - 1e-3)
            width = min(width, max_width)

            # Choose placement bias (top or bottom)
            toward_top = bool(self.rng.random() < 0.5)
            if toward_top:
                y_lo = y_bot + min_passage_m + width / 2.0
                y_hi = y_top - width / 2.0
            else:
                y_lo = y_bot + width / 2.0
                y_hi = y_top - min_passage_m - width / 2.0

            if y_hi < y_lo:
                y_center = cy
            else:
                y_center = float(self.rng.uniform(y_lo, y_hi))

            if x_hi <= x_lo:
                x_center = (x_lo + x_hi) / 2.0
            else:
                x_center = float(self.rng.uniform(x_lo, x_hi))

            self._fill_rect(
                grid,
                x_center - length / 2.0,
                y_center - width / 2.0,
                x_center + length / 2.0,
                y_center + width / 2.0,
            )
            pallets.append((x_center, y_center, length, width))

        return pallets

    def _generate_centerline_waypoints(
        self, start_x: float, goal_x: float, cy: float, waypoint_step_m: float
    ) -> np.ndarray:
        """Generate waypoints along corridor centerline.

        Args:
            start_x: Start x coordinate (meters)
            goal_x: Goal x coordinate (meters)
            cy: Centerline y coordinate (meters)
            waypoint_step_m: Step size between waypoints (meters)

        Returns:
            Array of shape (N, 2) with waypoint coordinates
        """
        n_wp = max(2, int(round((goal_x - start_x) / max(1e-6, waypoint_step_m))))
        xs = np.linspace(start_x, goal_x, n_wp)
        return np.stack([xs, np.full_like(xs, cy)], axis=1)


class BlockageGenerator(BaseCorridorGenerator):
    """Generator for blockage-only corridor scenarios."""

    def _sample_pallet_dims(self, corridor_w: float) -> tuple[float, float]:
        """Sample pallet dimensions with single-side passage constraint.

        Args:
            corridor_w: Corridor width for constraint checking

        Returns:
            (length_m, width_m) tuple
        """
        cfg = self.cfg
        pallet_w_eff = float(cfg.pallet_width_m)

        # Sample width ensuring minimum passage on one side
        T = float(cfg.min_passage_m)
        eps = 1e-3
        w_min = max(0.2, 0.4 * pallet_w_eff)
        w_max_geom = max(0.0, corridor_w - T - eps)
        w_cap = max(w_min, min(pallet_w_eff, w_max_geom))
        if w_cap <= 0.0:
            w_i = 0.2
        else:
            w_i = float(self.rng.uniform(w_min, w_cap))

        # Sample length
        l_min = max(0.3, 0.5 * cfg.pallet_length_m)
        l_i = float(self.rng.uniform(l_min, cfg.pallet_length_m))

        return (l_i, w_i)

    def generate(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float, float],
        tuple[float, float],
        dict[str, float],
    ]:
        """Generate a blockage corridor scenario.

        Returns:
            Occupancy grid, waypoints, start pose, goal xy, metadata dict
        """
        cfg = self.cfg

        # Create grid
        grid = self._create_grid(cfg.map_width_m, cfg.map_height_m)

        # Sample corridor dimensions
        corridor_w = float(
            self.rng.uniform(cfg.corridor_width_min_m, cfg.corridor_width_max_m)
        )
        cy = cfg.map_height_m / 2.0
        y_top = cy + corridor_w / 2.0
        y_bot = cy - corridor_w / 2.0

        # Draw corridor walls
        self._draw_corridor_walls(
            grid, y_top, y_bot, cfg.wall_thickness_m, cfg.map_width_m
        )

        # Placement bounds
        start_x = float(cfg.start_x_m)
        goal_x = float(cfg.map_width_m - cfg.goal_margin_x_m)
        x_lo = start_x + float(getattr(cfg, "min_pallet_x_offset_m", 0.6))
        x_hi = goal_x - 0.6

        # Sample and place pallets
        num_pallets = int(self.rng.integers(cfg.num_pallets_min, cfg.num_pallets_max + 1))
        pallets = self._place_pallets(
            grid,
            corridor_w,
            y_bot,
            y_top,
            x_lo,
            x_hi,
            cy,
            num_pallets,
            cfg.min_passage_m,
        )

        # Generate waypoints
        waypoints = self._generate_centerline_waypoints(
            start_x, goal_x, cy, cfg.waypoint_step_m
        )

        # Build info dict
        info = {
            "corridor_width": float(corridor_w),
            "corridor_width_final": float(corridor_w),
            "pallet_width_final": float(cfg.pallet_width_m),
            "num_pallets": int(num_pallets),
            "pallet_centers": [(float(px), float(py)) for (px, py, _, _) in pallets],
            "pallet_sizes": [(float(pl), float(pw)) for (_, _, pl, pw) in pallets],
        }

        start_pose = (float(start_x), float(cy), 0.0)
        goal_xy = (float(goal_x), float(cy))

        return grid, waypoints, start_pose, goal_xy, info


class OMCFGenerator(BaseCorridorGenerator):
    """Generator for occluded merge & counterflow (OMCF) scenarios."""

    def _sample_pallet_dims(self, corridor_w: float) -> tuple[float, float]:
        """Sample pallet dimensions with large/small distinction.

        Args:
            corridor_w: Corridor width for constraint checking

        Returns:
            (length_m, width_m) tuple
        """
        cfg = self.cfg

        # Decide large vs small based on configured fraction
        is_large = bool(self.rng.random() < max(0.0, min(1.0, cfg.large_fraction)))

        if is_large:
            length = float(
                self.rng.uniform(
                    min(cfg.large_length_range_m),
                    max(cfg.large_length_range_m),
                )
            )
            width = float(
                self.rng.uniform(
                    min(cfg.large_width_range_m),
                    max(cfg.large_width_range_m),
                )
            )
        else:
            length = float(
                self.rng.uniform(
                    min(cfg.small_length_range_m),
                    max(cfg.small_length_range_m),
                )
            )
            width = float(
                self.rng.uniform(
                    min(cfg.small_width_range_m),
                    max(cfg.small_width_range_m),
                )
            )

        return (length, width)

    def _sample_hole_positions(self) -> list[float]:
        """Sample x positions for wall hole pairs.

        Returns:
            List of x coordinates for hole centers
        """
        cfg = self.cfg
        holes_x = []

        if not cfg.holes_enabled or cfg.holes_count_pairs <= 0:
            return holes_x

        if cfg.holes_pair_x_candidates:
            # Use candidate positions
            candidates = list(float(x) for x in cfg.holes_pair_x_candidates)
            if candidates:
                desired = int(cfg.holes_count_pairs)
                num_pairs = min(desired, len(candidates))
                choices = self.rng.choice(
                    candidates,
                    size=num_pairs,
                    replace=False,
                )
                holes_x.extend(float(x) for x in np.atleast_1d(choices))
                # Fill remaining with random sampling
                if num_pairs < desired:
                    for _ in range(desired - num_pairs):
                        candidate = float(self.rng.uniform(cfg.holes_x_lo_m, cfg.holes_x_hi_m))
                        holes_x.append(candidate)
        else:
            # Sample with spacing constraint
            for _ in range(int(cfg.holes_count_pairs)):
                candidate = None
                for _ in range(16):
                    x_try = float(self.rng.uniform(cfg.holes_x_lo_m, cfg.holes_x_hi_m))
                    if all(
                        abs(x_try - existing) >= cfg.holes_min_spacing_m
                        for existing in holes_x
                    ):
                        candidate = x_try
                        break
                if candidate is None:
                    candidate = float(self.rng.uniform(cfg.holes_x_lo_m, cfg.holes_x_hi_m))
                holes_x.append(candidate)

        return sorted(float(x) for x in holes_x)

    def _draw_corridor_walls_with_gaps(
        self,
        grid: np.ndarray,
        y_top: float,
        y_bot: float,
        holes_x: list[float],
    ) -> None:
        """Draw corridor walls with gaps at hole positions.

        Args:
            grid: Occupancy grid to modify
            y_top: Top edge of corridor (meters)
            y_bot: Bottom edge of corridor (meters)
            holes_x: List of x coordinates for hole centers
        """
        cfg = self.cfg

        def draw_wall_with_gaps(y0: float, y1: float) -> None:
            if not holes_x:
                self._fill_rect(grid, 0.0, y0, cfg.map_width_m, y1)
                return

            for idx, x_h in enumerate(holes_x):
                half = cfg.holes_open_len_m / 2.0
                left_hi = max(0.0, x_h - half)
                right_lo = min(cfg.map_width_m, x_h + half)

                if idx == 0:
                    self._fill_rect(grid, 0.0, y0, left_hi, y1)
                else:
                    prev = holes_x[idx - 1] + half
                    self._fill_rect(grid, prev, y0, left_hi, y1)

                if idx == len(holes_x) - 1:
                    self._fill_rect(grid, right_lo, y0, cfg.map_width_m, y1)

        draw_wall_with_gaps(y_top, y_top + cfg.wall_thickness_m)
        draw_wall_with_gaps(y_bot - cfg.wall_thickness_m, y_bot)

    def generate(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float, float],
        tuple[float, float],
        dict[str, Any],
    ]:
        """Generate an OMCF corridor scenario with wall holes.

        Returns:
            Occupancy grid, waypoints, start pose, goal xy, metadata dict
        """
        cfg = self.cfg

        # Create grid
        grid = self._create_grid(cfg.map_width_m, cfg.map_height_m)

        # Sample corridor dimensions
        corridor_w = float(
            self.rng.uniform(cfg.corridor_width_min_m, cfg.corridor_width_max_m)
        )
        cy = cfg.map_height_m / 2.0
        y_top = cy + corridor_w / 2.0
        y_bot = cy - corridor_w / 2.0

        # Sample hole positions
        holes_x = self._sample_hole_positions()

        # Draw corridor walls with gaps
        self._draw_corridor_walls_with_gaps(grid, y_top, y_bot, holes_x)

        # Placement bounds
        start_x = float(cfg.start_x_m)
        goal_x = float(cfg.map_width_m - cfg.goal_margin_x_m)
        x_lo = start_x + 1.0
        x_hi = goal_x - 1.0

        # Sample and place pallets
        num_pallets = int(self.rng.integers(cfg.num_pallets_min, cfg.num_pallets_max + 1))
        pallets = self._place_pallets(
            grid,
            corridor_w,
            y_bot,
            y_top,
            x_lo,
            x_hi,
            cy,
            num_pallets,
            cfg.min_passage_m,
        )

        # Generate waypoints
        waypoints = self._generate_centerline_waypoints(
            start_x, goal_x, cy, cfg.waypoint_step_m
        )

        # Count pallet types
        small_cnt = 0
        large_cnt = 0
        for px, py, length, width in pallets:
            # Classify based on size ranges
            is_large = length >= min(cfg.large_length_range_m)
            if is_large:
                large_cnt += 1
            else:
                small_cnt += 1

        # Build info dict
        info: dict[str, Any] = {
            "corridor_width": float(corridor_w),
            "holes_x": [float(x) for x in holes_x],
            "y_top": float(y_top),
            "y_bot": float(y_bot),
            "pallets": pallets,
            "pallet_counts": {"small": int(small_cnt), "large": int(large_cnt)},
        }

        start_pose = (float(start_x), float(cy), 0.0)
        goal_xy = (float(goal_x), float(cy))

        return grid, waypoints, start_pose, goal_xy, info


def create_blockage_scenario(
    cfg: Optional[BlockageScenarioConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    np.ndarray,  # occupancy grid: True=occupied
    np.ndarray,  # waypoints: shape (N, 2)
    tuple[float, float, float],  # start pose (x, y, theta)
    tuple[float, float],  # goal xy
    dict[str, float],  # info dict with metadata
]:
    """Generate a blockage-only scenario with corridor and pallets.

    Grid convention: True = occupied; indices [row, col] = [y, x].
    """
    c = cfg or BlockageScenarioConfig()
    r = rng or np.random.default_rng()
    gen = BlockageGenerator(c, r)
    return gen.generate()


def create_omcf_scenario(
    cfg: Optional[OMCFConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[float, float, float],
    Tuple[float, float],
    Dict[str, object],
]:
    """Generate an occluded merge & counterflow corridor."""
    c = cfg or OMCFConfig()
    r = rng or np.random.default_rng()
    gen = OMCFGenerator(c, r)
    return gen.generate()
