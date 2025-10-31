"""Curriculum-aware scenario manager.

For Phase I, samples blockage-only scenarios using BlockageScenarioConfig.
"""

from __future__ import annotations

from typing import Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque

from .scenarios import BlockageScenarioConfig, create_blockage_scenario
from .scenarios_omcf import OMCFConfig, create_omcf_scenario
from .collision import inflate_grid


@dataclass
class ScenarioSample:
    grid_raw: np.ndarray
    grid_inflated: np.ndarray
    waypoints: np.ndarray
    start_pose: tuple[float, float, float]
    goal_xy: tuple[float, float]
    info: dict[str, Any]
    edt: np.ndarray | None
    edt_ms: float


class ScenarioManager:
    """Simple scenario manager with lightweight scenario selection."""

    def __init__(
        self, env_cfg: dict[str, Any], robot_radius_m: float | None = None, resolution_m: float | None = None
    ) -> None:
        """Initialize with Hydra-like env config dictionary.

        Expected keys under env_cfg:
        - map.size_m: [W, H]
        - map.resolution_m
        - corridor_width_m: [min, max]
        - wall_thickness_m, pallet_width_m, pallet_length_m (optional)
        - start_x_m, goal_margin_x_m, waypoint_step_m (optional)
        - min_passage_m, num_pallets_min, num_pallets_max (optional)
        """
        self.env_cfg = env_cfg
        self._rng = np.random.default_rng()
        self._scenario_name = str(env_cfg.get("name", "blockage")).lower()
        robot_cfg = env_cfg.get("robot", {}) if isinstance(env_cfg, dict) else {}
        if robot_radius_m is not None:
            self._robot_radius_m = float(robot_radius_m)
        else:
            self._robot_radius_m = float(robot_cfg.get("radius_m", 0.25))

        # Store resolution for EDT computation
        if resolution_m is not None:
            self._resolution = float(resolution_m)
        else:
            map_cfg = env_cfg.get("map", {}) if isinstance(env_cfg, dict) else {}
            self._resolution = float(map_cfg.get("resolution_m", 0.05))

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> ScenarioSample:
        """Sample a feasible scenario and build auxiliary products.

        Returns:
            ScenarioSample dataclass with grids, waypoints, start/goal, EDT
        """
        # Allocate buffer for reusing inflated grid
        grid_inflated_buffer = None

        # Try to generate a feasible scenario with retries
        max_retries = 5
        for _ in range(max_retries):
            grid_raw, waypoints, start_pose, goal_xy, info = self.sample_raw()

            # Create inflated grid buffer on first iteration
            if grid_inflated_buffer is None:
                grid_inflated_buffer = np.empty_like(grid_raw, dtype=bool)

            # Check path feasibility and reuse inflated grid via output parameter
            if self._is_path_feasible(
                grid_raw, start_pose, goal_xy, self._resolution, grid_inflated_buffer
            ):
                # Compute EDT on the inflated grid (reused from feasibility check)
                edt, edt_ms = self._compute_edt(grid_inflated_buffer)
                return ScenarioSample(
                    grid_raw=grid_raw,
                    grid_inflated=grid_inflated_buffer.copy(),
                    waypoints=waypoints,
                    start_pose=start_pose,
                    goal_xy=goal_xy,
                    info=info,
                    edt=edt,
                    edt_ms=edt_ms,
                )

            # If not feasible, try again with different random state
            self._rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # If all retries failed, return the last attempt (fallback)
        grid_raw, waypoints, start_pose, goal_xy, info = self.sample_raw()
        grid_inflated = inflate_grid(grid_raw, self._robot_radius_m, self._resolution)
        edt, edt_ms = self._compute_edt(grid_inflated)
        return ScenarioSample(
            grid_raw=grid_raw,
            grid_inflated=grid_inflated,
            waypoints=waypoints,
            start_pose=start_pose,
            goal_xy=goal_xy,
            info=info,
            edt=edt,
            edt_ms=edt_ms,
        )

    def _compute_edt(self, grid_inflated: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Compute Euclidean distance transform on inflated grid."""
        from .edt import compute_edt_meters

        free_mask = (~grid_inflated).astype(np.uint8)
        edt, ms = compute_edt_meters(free_mask, self._resolution)
        return edt, ms

    def sample_raw(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float, float],
        tuple[float, float],
        dict[str, Any],
    ]:
        """Sample a scenario and return raw outputs.

        Returns:
            Occupancy grid, waypoints, start pose, goal xy, metadata dict.
        """
        scenario_name = str(self.env_cfg.get("name", "blockage")).lower()
        if scenario_name == "blockage":
            cfg = self._build_blockage_cfg()
            generator = create_blockage_scenario
        elif scenario_name == "omcf":
            cfg = self._build_omcf_cfg()
            generator = create_omcf_scenario
        else:
            raise NotImplementedError(f"Unknown scenario '{scenario_name}'.")

        return generator(cfg, self._rng)

    def _build_blockage_cfg(self) -> BlockageScenarioConfig:
        msize = self.env_cfg["map"]["size_m"]
        resolution = float(self.env_cfg["map"]["resolution_m"])
        cw_min, cw_max = self.env_cfg["map"]["corridor_width_m"]
        wall_th = float(self.env_cfg["map"]["wall_thickness_m"])

        pallet_w = float(self.env_cfg["map"]["pallet_width_m"])
        pallet_l = float(self.env_cfg["map"]["pallet_length_m"])
        start_x = float(self.env_cfg["map"]["start_x_m"])
        goal_mx = float(self.env_cfg["map"]["goal_margin_x_m"])
        wp_step = float(self.env_cfg["map"]["waypoint_step_m"])
        min_pass = float(self.env_cfg["map"]["min_passage_m"])
        nmin = int(self.env_cfg["map"]["num_pallets_min"])
        nmax = int(self.env_cfg["map"]["num_pallets_max"])

        return BlockageScenarioConfig(
            map_width_m=float(msize[0]),
            map_height_m=float(msize[1]),
            corridor_width_min_m=float(cw_min),
            corridor_width_max_m=float(cw_max),
            wall_thickness_m=wall_th,
            pallet_width_m=pallet_w,
            pallet_length_m=pallet_l,
            start_x_m=start_x,
            goal_margin_x_m=goal_mx,
            waypoint_step_m=wp_step,
            resolution_m=resolution,
            min_passage_m=min_pass,
            num_pallets_min=nmin,
            num_pallets_max=nmax,
        )

    def _build_omcf_cfg(self) -> OMCFConfig:
        map_cfg = self.env_cfg["map"]
        holes_cfg = map_cfg.get("holes", {})
        small_cfg = map_cfg.get("small_static", {})
        large_cfg = map_cfg.get("large_static", {})

        def _range(cfg_section: dict[str, Any], key: str, default: Tuple[float, float]) -> Tuple[float, float]:
            values = cfg_section.get(key)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                return (float(values[0]), float(values[1]))
            return tuple(float(v) for v in default)

        return OMCFConfig(
            map_width_m=float(map_cfg["size_m"][0]),
            map_height_m=float(map_cfg["size_m"][1]),
            corridor_width_min_m=float(map_cfg["corridor_width_m"][0]),
            corridor_width_max_m=float(map_cfg["corridor_width_m"][1]),
            wall_thickness_m=float(map_cfg["wall_thickness_m"]),
            start_x_m=float(map_cfg["start_x_m"]),
            goal_margin_x_m=float(map_cfg["goal_margin_x_m"]),
            waypoint_step_m=float(map_cfg["waypoint_step_m"]),
            resolution_m=float(map_cfg["resolution_m"]),
            pallet_width_m=float(map_cfg["pallet_width_m"]),
            pallet_length_m=float(map_cfg["pallet_length_m"]),
            num_pallets_min=int(map_cfg["num_pallets_min"]),
            num_pallets_max=int(map_cfg["num_pallets_max"]),
            min_passage_m=float(map_cfg["min_passage_m"]),
            small_length_range_m=_range(
                small_cfg, "length_m", OMCFConfig.small_length_range_m
            ),
            small_width_range_m=_range(
                small_cfg, "width_m", OMCFConfig.small_width_range_m
            ),
            large_length_range_m=_range(
                large_cfg, "length_m", OMCFConfig.large_length_range_m
            ),
            large_width_range_m=_range(
                large_cfg, "width_m", OMCFConfig.large_width_range_m
            ),
            large_fraction=float(map_cfg.get("large_fraction", OMCFConfig.large_fraction)),
            holes_enabled=bool(holes_cfg.get("enabled", True)),
            holes_count_pairs=int(holes_cfg.get("count_pairs", 1)),
            holes_x_lo_m=float(holes_cfg.get("x_range_m", [15.0, 18.0])[0]),
            holes_x_hi_m=float(holes_cfg.get("x_range_m", [15.0, 18.0])[1]),
            holes_open_len_m=float(holes_cfg.get("opening_len_m", 1.6)),
            holes_min_spacing_m=float(
                holes_cfg.get("min_spacing_m", OMCFConfig.holes_min_spacing_m)
            ),
            holes_pair_x_candidates=tuple(
                float(v) for v in holes_cfg.get("pair_x_candidates_m", [])
            ),
        )

    def _is_path_feasible(
        self,
        grid: np.ndarray,
        start_pose: tuple[float, float, float],
        goal_xy: tuple[float, float],
        resolution: float,
        out_grid_inflated: np.ndarray | None = None,
    ) -> bool:
        """Path feasibility check using BFS with robot radius inflation.

        Args:
            grid: Occupancy grid (True=occupied)
            start_pose: (x, y, theta) in meters and radians
            goal_xy: (x, y) in meters
            resolution: Grid resolution in meters per cell
            out_grid_inflated: Optional output buffer to store inflated grid

        Returns:
            True if path exists, False otherwise
        """
        # Create inflated grid for robot radius
        grid_inflated = inflate_grid(grid, self._robot_radius_m, resolution)

        # Copy to output buffer if provided (for reuse)
        if out_grid_inflated is not None:
            np.copyto(out_grid_inflated, grid_inflated)

        start_x, start_y, _ = start_pose
        goal_x, goal_y = goal_xy

        # Convert to grid coordinates
        start_i = int(np.floor(start_y / resolution))
        start_j = int(np.floor(start_x / resolution))
        goal_i = int(np.floor(goal_y / resolution))
        goal_j = int(np.floor(goal_x / resolution))

        H, W = grid_inflated.shape

        # Check bounds
        if (
            start_i < 0
            or start_i >= H
            or start_j < 0
            or start_j >= W
            or goal_i < 0
            or goal_i >= H
            or goal_j < 0
            or goal_j >= W
        ):
            return False

        # Check if start/goal are in obstacles
        if grid_inflated[start_i, start_j] or grid_inflated[goal_i, goal_j]:
            return False

        # BFS to find path
        visited = np.zeros_like(grid_inflated, dtype=bool)
        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while queue:
            i, j = queue.popleft()

            if i == goal_i and j == goal_j:
                return True

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if visited[ni, nj] or grid_inflated[ni, nj]:
                    continue
                if di != 0 and dj != 0:
                    adj1_i, adj1_j = i, nj
                    adj2_i, adj2_j = ni, j
                    if grid_inflated[adj1_i, adj1_j] or grid_inflated[adj2_i, adj2_j]:
                        continue
                visited[ni, nj] = True
                queue.append((ni, nj))

        return False
