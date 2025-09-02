from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import scipy.ndimage as ndi

import gymnasium as gym
from gymnasium import spaces

from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from ..models.unicycle import UnicycleModel, UnicycleState
from ..sensors.lidar import Lidar
from ..constants import (
    DT_S,
    GRID_RESOLUTION_M,
    LIDAR_BEAMS,
    LIDAR_FOV_DEG,
    LIDAR_RANGE_M,
    ROBOT_RADIUS_M,
    EPISODE_TIME_S,
    GOAL_RADIUS_M,
)
from ..viz.pygame_render import PygameRenderer


@dataclass
class EnvConfig:
    dt: float = DT_S
    map_size: Tuple[int, int] = (300, 400)  # cells (H, W); actual meters depend on res
    res: float = GRID_RESOLUTION_M
    lidar_beams: int = LIDAR_BEAMS
    lidar_fov_deg: float = LIDAR_FOV_DEG
    lidar_range: float = LIDAR_RANGE_M
    goal_radius: float = GOAL_RADIUS_M
    episode_time_s: float = EPISODE_TIME_S
    robot_radius: float = ROBOT_RADIUS_M
    # Previously hardcoded parameters
    start_retry_attempts: int = 50
    path_preview_step: int = 5
    path_preview_count: int = 3


class UnicycleNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    """
    Gymnasium-compatible navigation environment for a unicycle robot.

    Observation vector: [lidar (N), v, omega, path_error(2), wp_preview(6), norm_time(1)]
    Action: [v_cmd, omega_cmd]
    """

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
        grid: Optional[BinaryOccupancyGrid] = None,
        path_waypoints: Optional[np.ndarray] = None,
        start_pose: Optional[Tuple[float, float, float]] = None,
        goal_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng()
        self.max_steps = int(self.cfg.episode_time_s / self.cfg.dt)
        self._step_count = 0
        self.render_mode = render_mode
        # Rendering backends (lazy init)
        self._fig = None  # matplotlib figure
        self._ax = None  # matplotlib axes
        self._pg: Optional[PygameRenderer] = None  # pygame renderer
        H, W = self.cfg.map_size
        self.grid = grid or BinaryOccupancyGrid(
            np.zeros((H, W), dtype=bool), cellsize=self.cfg.res, origin=(0, 0)
        )
        # Precompute configuration-space (C-space) occupancy by inflating obstacles
        # with the robot radius. Collision is then center-in-occupied on this grid.
        self._grid_cspace: BinaryOccupancyGrid = self._build_cspace_grid()
        self._last_collision: bool = False
        # Path/goal
        self.path_waypoints = path_waypoints  # (M,2) or None
        self.goal_xy = goal_xy
        self._start_pose = start_pose

        self.lidar = Lidar(
            n_beams=self.cfg.lidar_beams,
            fov=np.deg2rad(self.cfg.lidar_fov_deg),
            max_range=self.cfg.lidar_range,
            step=0.02,
        )
        self.robot = UnicycleModel()

        # Observation and action spaces
        # Observation vector: [lidar (N), v, omega, path_error(2), wp_preview(6), norm_time(1)]
        lidar_dim = self.cfg.lidar_beams
        kinematics_dim = 2  # v, omega
        path_error_dim = 2  # cross-track, heading error
        preview_dim = self.cfg.path_preview_count * 2  # 3 waypoints * (dx, dy)
        time_dim = 1  # normalized time

        obs_size = lidar_dim + kinematics_dim + path_error_dim + preview_dim + time_dim
        low = np.zeros((obs_size,), dtype=np.float32)
        high = np.ones((obs_size,), dtype=np.float32) * self.cfg.lidar_range

        # Velocity bounds
        v_start = lidar_dim
        high[v_start] = self.robot.v_max
        low[v_start] = self.robot.v_min

        # Angular velocity bounds
        w_start = v_start + 1
        high[w_start] = self.robot.w_max
        low[w_start] = self.robot.w_min

        # Path error bounds
        path_err_start = w_start + 1
        low[path_err_start : path_err_start + path_error_dim] = np.array(
            [-5.0, -np.pi], dtype=np.float32
        )
        high[path_err_start : path_err_start + path_error_dim] = np.array(
            [5.0, np.pi], dtype=np.float32
        )

        # Waypoint preview bounds
        preview_start = path_err_start + path_error_dim
        low[preview_start : preview_start + preview_dim] = -self.cfg.lidar_range
        high[preview_start : preview_start + preview_dim] = self.cfg.lidar_range

        # Time bounds
        time_start = preview_start + preview_dim
        low[time_start] = 0.0
        high[time_start] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.robot.v_min, self.robot.w_min], dtype=np.float32),
            high=np.array([self.robot.v_max, self.robot.w_max], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducible resets
            options: Additional reset options (unused)

        Returns:
            tuple: (observation, info) where observation is the initial state
                   and info contains reset metadata
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        if self._start_pose is not None:
            x, y, th = self._start_pose
            # ensure start not inside obstacle; walk along path if needed
            if (
                hasattr(self._grid_cspace, "isoccupied")
                and self._grid_cspace.isoccupied((x, y))
                and self.path_waypoints is not None
            ):
                for j in range(1, self.cfg.start_retry_attempts):
                    i = min(j, self.path_waypoints.shape[0] - 1)
                    x, y = float(self.path_waypoints[i, 0]), float(
                        self.path_waypoints[i, 1]
                    )
                    if not self._grid_cspace.isoccupied((x, y)):
                        break
            self.robot.reset(UnicycleState(x, y, th, 0.0, 0.0))
        else:
            # default free-space start near origin
            self.robot.reset(UnicycleState(1.0, 1.0, 0.0, 0.0, 0.0))
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Tuple[float, float]):
        """
        Execute one environment step with the given action.

        Args:
            action: Tuple of (linear_velocity, angular_velocity) commands
                   - linear_velocity: m/s, within robot velocity limits
                   - angular_velocity: rad/s, within robot angular velocity limits

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Current environment observation
                - reward: Step reward (currently 0.0 for classical control)
                - terminated: True if episode ended (collision or success)
                - truncated: True if episode exceeded time limit
                - info: Dict with 'collision' and 'success' flags

        Raises:
            ValueError: If action format is invalid or values out of range
        """
        # Input validation
        if not isinstance(action, (tuple, list, np.ndarray)) or len(action) != 2:
            raise ValueError(
                "Action must be a tuple/list/array of length 2: [v_cmd, omega_cmd]"
            )

        v_cmd, omega_cmd = float(action[0]), float(action[1])
        if not (-10.0 <= v_cmd <= 10.0) or not (-10.0 <= omega_cmd <= 10.0):
            raise ValueError(
                f"Action values out of reasonable range: v={v_cmd}, ω={omega_cmd}"
            )
        self._step_count += 1
        self.robot.step(action, dt=self.cfg.dt)
        obs = self._get_obs()
        # Geometric collision: center-in-occupied on C-space grid
        x, y, _ = self.robot.as_pose()
        collision_static = bool(self._grid_cspace.isoccupied((x, y)))
        collision_dynamic = self._check_dynamic_collisions(x, y)
        collision = collision_static or collision_dynamic
        success = False
        if self.goal_xy is not None:
            gx, gy = self.goal_xy
            if np.hypot(gx - x, gy - y) <= self.cfg.goal_radius:
                success = True
        terminated = collision or success
        truncated = self._step_count >= self.max_steps
        # Classical controller demo does not require learning rewards.
        # Return a neutral reward to satisfy Gym API without shaping.
        reward = 0.0
        self._last_collision = collision
        info = {"collision": collision, "success": success}
        return obs, reward, terminated, truncated, info

    # -------- Rendering API ---------
    def render(self):
        """
        Render the environment for visualization.

        Returns:
            None for 'human' mode, rgb_array for 'rgb_array' mode
        """
        if self.render_mode is None:
            return None
        # Choose pygame for 'human' interactive rendering; also use pygame for rgb_array
        if self.render_mode == "human":
            if self._pg is None:
                self._pg = PygameRenderer()
            # pass through; pygame manages its own window
            self._pg.draw(self)
            return None
        elif self.render_mode == "rgb_array":
            if self._pg is None:
                self._pg = PygameRenderer()
            return self._pg.draw_rgb_array(self)
        else:
            return None

    def close(self):
        """
        Clean up rendering resources and close the environment.
        """
        # Close any open figure
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except ImportError:
                pass
        self._fig, self._ax = None, None
        # Close pygame window if any
        if self._pg is not None:
            try:
                self._pg.close()
            except (AttributeError, RuntimeError):
                pass
        self._pg = None

    def _get_obs(self) -> np.ndarray:
        pose = self.robot.as_pose()
        ranges, _ = self.lidar.cast(pose, self.grid)
        rs = self.robot.get_state()
        # path error and waypoints preview if available
        if self.path_waypoints is not None and self.path_waypoints.shape[0] >= 2:
            ct_err, hdg_err, preview = self._path_features(pose)
            path_err = np.array([ct_err, hdg_err], dtype=float)
            wp_preview = preview
        else:
            path_err = np.array([0.0, 0.0], dtype=float)
            wp_preview = np.zeros((self.cfg.path_preview_count * 2,), dtype=float)
        tnorm = np.array([self._step_count / max(1, self.max_steps)], dtype=float)
        obs = np.concatenate(
            [
                ranges,
                np.array([rs.v, rs.omega], dtype=float),
                path_err,
                wp_preview,
                tnorm,
            ]
        )
        return obs.astype(np.float32)

    # -------- Grid accessors --------
    @property
    def collision_grid(self) -> BinaryOccupancyGrid:
        """Configuration-space grid used for all collision/feasibility checks."""
        return self._grid_cspace

    @property
    def sensing_grid(self) -> BinaryOccupancyGrid:
        """Raw occupancy grid used for LiDAR sensing and rendering."""
        return self.grid

    def _set_rect_obstacle(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        self.grid.set([x_min, y_min, x_max, y_max], True)

    # ------- Collision helpers --------
    def _build_cspace_grid(self) -> BinaryOccupancyGrid:
        """
        Build a configuration-space occupancy grid by dilating obstacles with
        a circular structuring element of radius equal to the robot radius.
        """
        try:
            # Build C-space via Euclidean Distance Transform (EDT) thresholding.
            # Any cell with distance to nearest obstacle <= robot_radius treated occupied.
            occ = self.grid.grid.astype(bool)
            res = float(getattr(self.grid, "_cellsize", self.cfg.res))
            if res <= 0:
                res = self.cfg.res
            # Distance in meters from free cells to nearest obstacle
            # distance_transform_edt returns distance in pixels; multiply by res
            free = ~occ
            dist_px = ndi.distance_transform_edt(free)
            dist_m = dist_px * res
            # Threshold at robot radius (optionally add safety margin if needed)
            r_eff = float(self.cfg.robot_radius)
            cspace_occ = dist_m <= r_eff
            return BinaryOccupancyGrid(cspace_occ, cellsize=res, origin=(0, 0))
        except (AttributeError, ValueError, TypeError) as e:
            # Fallback to original grid if dilation fails
            print(f"Warning: C-space grid dilation failed: {e}. Using original grid.")
            return self.grid

    def _check_dynamic_collisions(self, x: float, y: float) -> bool:
        """Placeholder for dynamic obstacle collisions (circle-circle)."""
        # TODO: Implement dynamic collision detection
        # For now, no dynamic obstacles are present
        _ = x, y  # Suppress unused parameter warnings
        return False

    # ------- Path helpers --------
    def _nearest_path_index(self, x: float, y: float) -> int:
        if self.path_waypoints is None or self.path_waypoints.shape[0] == 0:
            raise ValueError("No path waypoints available")
        wp = self.path_waypoints
        d = np.hypot(wp[:, 0] - x, wp[:, 1] - y)
        return int(np.argmin(d))

    def _path_features(
        self, pose: Tuple[float, float, float]
    ) -> Tuple[float, float, np.ndarray]:
        x, y, th = pose
        idx = self._nearest_path_index(x, y)
        wp = self.path_waypoints
        # Ensure we have at least 2 waypoints for path following
        if wp.shape[0] < 2:
            return 0.0, 0.0, np.zeros((self.cfg.path_preview_count * 2,), dtype=float)

        idx2 = min(idx + 1, wp.shape[0] - 1)
        # If idx == idx2, we're at the last waypoint
        if idx == idx2 and idx > 0:
            idx2 = idx
            idx = idx - 1

        # cross-track: signed distance to segment normal
        p = np.array([x, y])
        a = wp[idx]
        b = wp[idx2]
        ab = b - a
        ab_len = max(1e-9, np.linalg.norm(ab))
        t = np.clip(np.dot(p - a, ab) / (ab_len**2), 0.0, 1.0)
        proj = a + t * ab
        # sign via left/right of segment
        n = np.array([-ab[1], ab[0]]) / ab_len
        ct_err = float(np.dot(p - proj, n))
        path_heading = float(np.arctan2(ab[1], ab[0]))
        hdg_err = float((path_heading - th + np.pi) % (2 * np.pi) - np.pi)
        # waypoint preview: next 3 deltas
        preview_pts = []
        for k in range(1, self.cfg.path_preview_count + 1):
            j = min(idx + k * self.cfg.path_preview_step, wp.shape[0] - 1)
            dx, dy = wp[j, 0] - x, wp[j, 1] - y
            preview_pts.extend([dx, dy])
        return ct_err, hdg_err, np.array(preview_pts, dtype=float)
