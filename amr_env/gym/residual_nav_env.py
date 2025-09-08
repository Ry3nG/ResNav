"""Residual navigation Gymnasium environment (Phase I core).

Observation Dict keys:
- lidar: 1×N array of distances (no stacking here; wrapper stacks later)
- kin: (v_t, ω_t, v_{t-1}, ω_{t-1})
- path: (d_lat, θ_err, x1, y1, x2, y2, x3, y3) in robot frame

Action: residual Δu added to Pure Pursuit u_track, clipped to robot limits.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from amr_env.sim.dynamics import UnicycleModel, UnicycleState
from amr_env.sim.scenario_manager import ScenarioManager
from amr_env.sim.lidar import GridLidar
from amr_env.sim.collision import inflate_grid
from control.pure_pursuit import compute_u_track
from .path_utils import compute_path_context


class ResidualNavEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_cfg: Dict[str, Any], robot_cfg: Dict[str, Any], reward_cfg: Dict[str, Any], run_cfg: Dict[str, Any]):
        super().__init__()
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        self.reward_cfg = reward_cfg
        self.run_cfg = run_cfg

        # Scenario manager
        self.scenarios = ScenarioManager(env_cfg)

        # Robot limits
        self.v_max = float(robot_cfg.get("v_max", 1.5))
        self.w_max = float(robot_cfg.get("w_max", 2.0))
        self.radius_m = float(robot_cfg.get("radius_m", 0.25))

        # Time step
        self.dt = float(run_cfg.get("dt", 0.1))

        # LiDAR parameters
        lidar_cfg = env_cfg.get("lidar", {})
        map_cfg = env_cfg.get("map", {})
        self.resolution_m = float(map_cfg.get("resolution_m", 0.2))
        self.lidar = GridLidar(
            beams=int(lidar_cfg.get("beams", 24)),
            fov_deg=float(lidar_cfg.get("fov_deg", 240.0)),
            max_range_m=float(lidar_cfg.get("max_range_m", 4.0)),
            noise_std_m=float(lidar_cfg.get("noise_std_m", 0.03)),
            noise_enable=bool(lidar_cfg.get("noise_enable", True)),
            resolution_m=self.resolution_m,
        )

        # Observation and action spaces
        n_beams = int(lidar_cfg.get("beams", 24))
        obs_lidar = spaces.Box(low=0.0, high=float(self.lidar.max_range), shape=(n_beams,), dtype=np.float32)
        obs_kin = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Path space bounds: reasonable physical limits
        d_lat_bound = float(self.env_cfg.get("map", {}).get("corridor_width_m", [3.0, 4.0])[1])
        theta_bound = float(np.pi)
        preview_bound = 10.0
        low_path = np.array([-d_lat_bound, -theta_bound, -preview_bound, -preview_bound, -preview_bound, -preview_bound, -preview_bound, -preview_bound], dtype=np.float32)
        high_path = -low_path
        obs_path = spaces.Box(low=low_path, high=high_path, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "lidar": obs_lidar,
            "kin": obs_kin,
            "path": obs_path,
        })
        self.action_space = spaces.Box(low=np.array([-self.v_max, -self.w_max], dtype=np.float32), high=np.array([self.v_max, self.w_max], dtype=np.float32), dtype=np.float32)

        # Internal state
        self._grid_raw = None
        self._grid_inflated = None
        self._waypoints = None
        self._start_pose = None
        self._goal_xy = None
        self._info = {}
        self._model = None
        self._last_u = (0.0, 0.0)
        self._prev_u = (0.0, 0.0)
        self._steps = 0
        self._max_steps = int(run_cfg.get("max_steps", 600))

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.scenarios.set_seed(seed)
            self.lidar.set_seed(seed)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Sample scenario and ensure start pose is free in inflated grid
        safety_margin = float(self.robot_cfg.get("safety_margin_m", 0.0))
        max_tries = 20
        for _ in range(max_tries):
            self._grid_raw, self._waypoints, self._start_pose, self._goal_xy, self._info = self.scenarios.sample()
            self._grid_inflated = inflate_grid(self._grid_raw, self.radius_m + safety_margin, self.resolution_m)
            if not self._point_in_inflated(self._start_pose[0], self._start_pose[1]):
                break
        else:
            # Fallback: slide start along centerline to nearest free cell
            map_h = float(self.env_cfg.get("map", {}).get("size_m", [50.0, 50.0])[1])
            y_center = map_h / 2.0
            H, W = self._grid_inflated.shape
            i_center = int(np.clip(np.floor(y_center / self.resolution_m), 0, H - 1))
            j0 = int(np.clip(np.floor(self._start_pose[0] / self.resolution_m), 0, W - 1))
            j_free = j0
            while j_free < W and self._grid_inflated[i_center, j_free]:
                j_free += 1
            x_free = (min(j_free, W - 1) + 0.5) * self.resolution_m
            self._start_pose = (x_free, y_center, self._start_pose[2])

        # Init dynamics
        self._model = UnicycleModel(v_max=self.v_max, w_max=self.w_max)
        x0, y0, th0 = self._start_pose
        self._model.reset(UnicycleState(x0, y0, th0, 0.0, 0.0))

        self._prev_u = (0.0, 0.0)
        self._last_u = (0.0, 0.0)
        self._steps = 0

        obs = self._get_obs()
        # Do not set SB3's episode info; RecordEpisodeStatistics will populate it
        return obs, {}

    def step(self, action: np.ndarray):
        # Residual action
        dv, dw = float(action[0]), float(action[1])

        # Base tracker command
        v_track, w_track = compute_u_track(self._model.as_pose(), self._waypoints, self.robot_cfg.get("controller", {}).get("lookahead_m", 1.2), self.robot_cfg.get("controller", {}).get("speed_nominal", 1.0))
        v_cmd = v_track + dv
        w_cmd = w_track + dw
        # Clip to robot limits
        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

        self._prev_u = self._last_u
        self._last_u = (v_cmd, w_cmd)

        state = self._model.step((v_cmd, w_cmd), self.dt)

        done = False
        terminated = False
        truncated = False
        reward = 0.0

        # Collision check on inflated grid
        if self._is_collision():
            terminated = True
            done = True

        # Timeout
        self._steps += 1
        if self._steps >= self._max_steps and not done:
            truncated = True
            done = True

        # Goal reached
        goal_dist = self._dist_to_goal()
        if goal_dist < 0.5 and not done:
            terminated = True
            done = True

        # Reward
        reward = self._compute_reward(goal_dist, terminated)

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        # Provide success flag on termination for eval metrics
        if terminated or truncated:
            info["is_success"] = bool(terminated and goal_dist < 0.5)
        return obs, reward, terminated, truncated, info

    # Debug/visualization helper to avoid poking private fields externally
    def get_render_payload(self) -> Dict[str, Any]:
        pose = self._model.as_pose()
        payload = {
            "raw_grid": self._grid_raw,
            "inflated_grid": self._grid_inflated,
            "pose": pose,
            "radius_m": self.radius_m,
            "lidar": {
                "beams": self.lidar.beams,
                "fov_rad": self.lidar.fov_rad,
                "max_range": self.lidar.max_range,
            },
            "waypoints": self._waypoints,
            "last_u": self._last_u,
            "prev_u": self._prev_u,
            "obs": self._get_obs(),  # returns current lidar/kin/path as well
        }
        return payload

    def _get_obs(self) -> Dict[str, np.ndarray]:
        x, y, th = self._model.as_pose()
        lidar = self.lidar.sense(self._grid_raw, (x, y, th)).astype(np.float32)

        kin = np.array([self._last_u[0], self._last_u[1], self._prev_u[0], self._prev_u[1]], dtype=np.float32)

        # Cache path context to avoid recomputation in reward
        self._last_ctx = compute_path_context((x, y, th), self._waypoints, (1.0, 2.0, 3.0))
        path = np.array([self._last_ctx.d_lat, self._last_ctx.theta_err, *self._last_ctx.previews_robot.flatten().tolist()], dtype=np.float32)

        return {"lidar": lidar, "kin": kin, "path": path}

    def _is_collision(self) -> bool:
        x, y, _ = self._model.as_pose()
        return self._point_in_inflated(x, y)

    def _point_in_inflated(self, x: float, y: float) -> bool:
        i = int(np.floor(y / self.resolution_m))
        j = int(np.floor(x / self.resolution_m))
        H, W = self._grid_inflated.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return bool(self._grid_inflated[i, j])

    def _dist_to_goal(self) -> float:
        x, y, _ = self._model.as_pose()
        gx, gy = self._goal_xy
        return float(np.hypot(gx - x, gy - y))

    def _compute_reward(self, goal_dist_t: float, terminated: bool) -> float:
        # Sparse components
        sparse = 0.0
        if terminated:
            if goal_dist_t < 0.5:
                sparse += float(self.reward_cfg.get("sparse", {}).get("goal", 200.0))
            else:
                sparse += float(self.reward_cfg.get("sparse", {}).get("collision", -200.0))

        # Progress shaping: use previous goal distance cached on env
        d_prev = getattr(self, "_prev_goal_dist", goal_dist_t)
        progress = d_prev - goal_dist_t
        self._prev_goal_dist = goal_dist_t

        # Path penalty
        ctx = getattr(self, "_last_ctx", compute_path_context(self._model.as_pose(), self._waypoints, (1.0, 2.0, 3.0)))
        path_pen = -(
            float(self.reward_cfg.get("path_penalty", {}).get("lateral_weight", 1.0)) * abs(ctx.d_lat)
            + float(self.reward_cfg.get("path_penalty", {}).get("heading_weight", 0.5)) * abs(ctx.theta_err)
        )

        # Effort penalty on residual (difference from tracker)
        v_track, w_track = compute_u_track(self._model.as_pose(), self._waypoints, self.robot_cfg.get("controller", {}).get("lookahead_m", 1.2), self.robot_cfg.get("controller", {}).get("speed_nominal", 1.0))
        dv = self._last_u[0] - v_track
        dw = self._last_u[1] - w_track
        lam = self.reward_cfg.get("effort_penalty", {"lambda_v": 1.0, "lambda_w": 1.0, "lambda_jerk": 0.05})
        effort = -(
            float(lam.get("lambda_v", 1.0)) * abs(dv)
            + float(lam.get("lambda_w", 1.0)) * abs(dw)
            + float(lam.get("lambda_jerk", 0.05)) * (abs(self._last_u[0] - self._prev_u[0]) + abs(self._last_u[1] - self._prev_u[1]))
        )

        wts = self.reward_cfg.get("weights", {"progress": 1.0, "path": 0.2, "effort": 0.01, "sparse": 1.0})
        R = (
            float(wts.get("progress", 1.0)) * progress
            + float(wts.get("path", 0.2)) * path_pen
            + float(wts.get("effort", 0.01)) * effort
            + float(wts.get("sparse", 1.0)) * sparse
        )
        return float(R)
