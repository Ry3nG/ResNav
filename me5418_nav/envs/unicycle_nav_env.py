from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from ..models.unicycle import UnicycleModel, UnicycleState
from ..sensors.lidar import Lidar
from ..viz.plotting import draw_env
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


class UnicycleNavEnv(gym.Env if hasattr(gym, "Env") else object):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    """
    Gymnasium-compatible navigation environment for a unicycle robot.

    Observation vector: [lidar (N), v, omega, path_error(2), wp_preview(6), norm_time(1)]
    Action: [v_cmd, omega_cmd]
    """

    def __init__(
        self,
        cfg: EnvConfig | None = None,
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
        self._pg: PygameRenderer | None = None  # pygame renderer
        H, W = self.cfg.map_size
        self.grid = grid or BinaryOccupancyGrid(
            np.zeros((H, W), dtype=bool), cellsize=self.cfg.res, origin=(0, 0)
        )
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
        obs_size = self.cfg.lidar_beams + 2 + 2 + 6 + 1
        low = np.zeros((obs_size,), dtype=np.float32)
        high = np.ones((obs_size,), dtype=np.float32) * self.cfg.lidar_range
        high[self.cfg.lidar_beams] = self.robot.v_max
        low[self.cfg.lidar_beams] = self.robot.v_min
        high[self.cfg.lidar_beams + 1] = self.robot.w_max
        low[self.cfg.lidar_beams + 1] = self.robot.w_min
        low[self.cfg.lidar_beams + 2 : self.cfg.lidar_beams + 4] = np.array(
            [-5.0, -np.pi], dtype=np.float32
        )
        high[self.cfg.lidar_beams + 2 : self.cfg.lidar_beams + 4] = np.array(
            [5.0, np.pi], dtype=np.float32
        )
        low[self.cfg.lidar_beams + 4 : self.cfg.lidar_beams + 10] = (
            -self.cfg.lidar_range
        )
        high[self.cfg.lidar_beams + 4 : self.cfg.lidar_beams + 10] = (
            self.cfg.lidar_range
        )
        low[-1] = 0.0
        high[-1] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.robot.v_min, self.robot.w_min], dtype=np.float32),
            high=np.array([self.robot.v_max, self.robot.w_max], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        if self._start_pose is not None:
            x, y, th = self._start_pose
            # ensure start not inside obstacle; walk along path if needed
            if (
                hasattr(self.grid, "isoccupied")
                and self.grid.isoccupied((x, y))
                and self.path_waypoints is not None
            ):
                for j in range(1, 50):
                    i = min(j, self.path_waypoints.shape[0] - 1)
                    x, y = float(self.path_waypoints[i, 0]), float(
                        self.path_waypoints[i, 1]
                    )
                    if not self.grid.isoccupied((x, y)):
                        break
            self.robot.reset(UnicycleState(x, y, th, 0.0, 0.0))
        else:
            # default free-space start near origin
            self.robot.reset(UnicycleState(1.0, 1.0, 0.0, 0.0, 0.0))
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Tuple[float, float]):
        self._step_count += 1
        self.robot.step(action, dt=self.cfg.dt)
        obs = self._get_obs()
        collision = bool(np.min(obs[: self.cfg.lidar_beams]) < 0.05)
        success = False
        if self.goal_xy is not None:
            gx, gy = self.goal_xy
            x, y, _ = self.robot.as_pose()
            if np.hypot(gx - x, gy - y) <= self.cfg.goal_radius:
                success = True
        terminated = collision or success
        truncated = self._step_count >= self.max_steps
        from ..constants import REWARD_STEP, REWARD_COLLISION, REWARD_SUCCESS

        reward = (
            REWARD_STEP
            + (REWARD_COLLISION if collision else 0.0)
            + (REWARD_SUCCESS if success else 0.0)
        )
        info = {"collision": collision, "success": success}
        return obs, reward, terminated, truncated, info

    # -------- Rendering API ---------
    def render(self):
        if self.render_mode is None:
            return None
        # Choose pygame for 'human' interactive rendering; matplotlib for rgb_array
        if self.render_mode == "human":
            if self._pg is None:
                self._pg = PygameRenderer()
            # pass through; pygame manages its own window
            self._pg.draw(self)
            return None
        elif self.render_mode == "rgb_array":
            # lazy-create matplotlib figure/axes
            if self._fig is None or self._ax is None:
                import matplotlib.pyplot as plt

                self._fig, self._ax = plt.subplots(figsize=(6, 6))
            # draw current state using Matplotlib
            status = getattr(self, "_debug_status", None)
            draw_env(self, self._ax, status)
            # convert canvas to RGB array; use renderer dims to handle HiDPI
            canvas = self._fig.canvas
            canvas.draw()
            import numpy as _np

            renderer = canvas.get_renderer()
            w = int(getattr(renderer, "width", canvas.get_width_height()[0]))
            h = int(getattr(renderer, "height", canvas.get_width_height()[1]))
            buf = _np.frombuffer(canvas.tostring_rgb(), dtype=_np.uint8)
            return buf.reshape(h, w, 3)
        else:
            return None

    def close(self):
        # Close any open figure
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception:
                pass
        self._fig, self._ax = None, None
        # Close pygame window if any
        if self._pg is not None:
            try:
                self._pg.close()
            except Exception:
                pass
        self._pg = None

    def _get_obs(self) -> np.ndarray:
        pose = self.robot.as_pose()
        ranges, _ = self.lidar.cast(pose, self.grid)
        rs = self.robot._last_state
        # path error and waypoints preview if available
        if self.path_waypoints is not None and self.path_waypoints.shape[0] >= 2:
            ct_err, hdg_err, preview = self._path_features(pose)
            path_err = np.array([ct_err, hdg_err], dtype=float)
            wp_preview = preview
        else:
            path_err = np.array([0.0, 0.0], dtype=float)
            wp_preview = np.zeros((6,), dtype=float)
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

    def _set_rect_obstacle(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        self.grid.set([x_min, y_min, x_max, y_max], True)

    # ------- Path helpers --------
    def _nearest_path_index(self, x: float, y: float) -> int:
        wp = self.path_waypoints
        d = np.hypot(wp[:, 0] - x, wp[:, 1] - y)
        return int(np.argmin(d))

    def _path_features(
        self, pose: Tuple[float, float, float]
    ) -> Tuple[float, float, np.ndarray]:
        x, y, th = pose
        idx = self._nearest_path_index(x, y)
        wp = self.path_waypoints
        idx2 = min(idx + 1, wp.shape[0] - 1)
        # cross-track: signed distance to segment normal
        p = np.array([x, y])
        a = wp[idx]
        b = wp[idx2]
        ab = b - a
        ab_len = max(1e-6, np.linalg.norm(ab))
        t = np.clip(np.dot(p - a, ab) / (ab_len**2), 0.0, 1.0)
        proj = a + t * ab
        # sign via left/right of segment
        n = np.array([-ab[1], ab[0]]) / ab_len
        ct_err = float(np.dot(p - proj, n))
        path_heading = float(np.arctan2(ab[1], ab[0]))
        hdg_err = float((path_heading - th + np.pi) % (2 * np.pi) - np.pi)
        # waypoint preview: next 3 deltas
        preview_pts = []
        for k in range(1, 4):
            j = min(idx + k * 5, wp.shape[0] - 1)
            dx, dy = wp[j, 0] - x, wp[j, 1] - y
            preview_pts.extend([dx, dy])
        return ct_err, hdg_err, np.array(preview_pts, dtype=float)
