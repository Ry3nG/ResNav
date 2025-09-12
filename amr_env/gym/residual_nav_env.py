"""Residual navigation Gymnasium environment (Phase I core).

Observation Dict keys:
- lidar: 1xN array of distances (no stacking here; wrapper stacks later)
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
from amr_env.reward import compute_terms, apply_weights, to_breakdown_dict, RewardTerms


class ResidualNavEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        robot_cfg: Dict[str, Any],
        reward_cfg: Dict[str, Any],
        run_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        self.reward_cfg = reward_cfg
        self.run_cfg = run_cfg

        # Scenario manager
        self.scenarios = ScenarioManager(env_cfg)

        # Robot limits
        self.v_max = float(robot_cfg["v_max"])
        self.w_max = float(robot_cfg["w_max"])
        self.v_min = float(robot_cfg["v_min"])
        self.radius_m = float(robot_cfg["radius_m"])

        # Time step
        self.dt = float(run_cfg["dt"])

        # LiDAR parameters
        lidar_cfg = env_cfg["lidar"]
        map_cfg = env_cfg["map"]
        self.resolution_m = float(map_cfg["resolution_m"])
        self.lidar = GridLidar(
            beams=int(lidar_cfg["beams"]),
            fov_deg=float(lidar_cfg["fov_deg"]),
            max_range_m=float(lidar_cfg["max_range_m"]),
            noise_std_m=float(lidar_cfg["noise_std_m"]),
            noise_enable=bool(lidar_cfg["noise_enable"]),
            resolution_m=self.resolution_m,
        )

        # Observation and action spaces
        n_beams = int(lidar_cfg["beams"])
        obs_lidar = spaces.Box(
            low=0.0,
            high=float(self.lidar.max_range),
            shape=(n_beams,),
            dtype=np.float32,
        )
        obs_kin = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Path space bounds: reasonable physical limits
        d_lat_bound = float(
            self.env_cfg["map"]["corridor_width_m"][1]
        )
        theta_bound = float(np.pi)
        preview_bound = 10.0
        low_path = np.array(
            [
                -d_lat_bound,
                -theta_bound,
                -preview_bound,
                -preview_bound,
                -preview_bound,
                -preview_bound,
                -preview_bound,
                -preview_bound,
            ],
            dtype=np.float32,
        )
        high_path = -low_path
        obs_path = spaces.Box(
            low=low_path, high=high_path, shape=(8,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "lidar": obs_lidar,
                "kin": obs_kin,
                "path": obs_path,
            }
        )
        self.action_space = spaces.Box(
            low=np.array([-self.v_max, -self.w_max], dtype=np.float32),
            high=np.array([self.v_max, self.w_max], dtype=np.float32),
            dtype=np.float32,
        )

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
        self._max_steps = int(run_cfg["max_steps"])
        # Reward bookkeeping for renderer/debug
        self._last_reward_terms: Dict[str, Any] = {}
        self._prev_goal_dist: float | None = None

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.scenarios.set_seed(seed)
            self.lidar.set_seed(seed)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Sample scenario and ensure start pose is free in inflated grid
        safety_margin = float(self.robot_cfg["safety_margin_m"])
        max_tries = 20
        for _ in range(max_tries):
            (
                self._grid_raw,
                self._waypoints,
                self._start_pose,
                self._goal_xy,
                self._info,
            ) = self.scenarios.sample()
            self._grid_inflated = inflate_grid(
                self._grid_raw, self.radius_m + safety_margin, self.resolution_m
            )
            if not self._point_in_inflated(self._start_pose[0], self._start_pose[1]):
                break

        # Init dynamics
        self._model = UnicycleModel(
            v_max=self.v_max, w_max=self.w_max, v_min=self.v_min
        )
        x0, y0, th0 = self._start_pose
        self._model.reset(UnicycleState(x0, y0, th0, 0.0, 0.0))

        self._prev_u = (0.0, 0.0)
        self._last_u = (0.0, 0.0)
        self._steps = 0
        self._prev_goal_dist = None
        self._last_reward_terms = {}

        obs = self._get_obs()
        # Do not set SB3's episode info; RecordEpisodeStatistics will populate it
        return obs, {}

    def step(self, action: np.ndarray):
        # Residual action
        dv, dw = float(action[0]), float(action[1])

        # Base tracker command
        v_track, w_track = compute_u_track(
            self._model.as_pose(),
            self._waypoints,
            self.robot_cfg["controller"]["lookahead_m"],
            self.robot_cfg["controller"]["speed_nominal"],
        )
        v_cmd = v_track + dv
        w_cmd = w_track + dw
        # Clip to robot limits
        v_cmd = float(np.clip(v_cmd, self.v_min, self.v_max))
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

        # Compute path context once for this new state (reuse in reward + obs)
        try:
            x, y, th = self._model.as_pose()
            from .path_utils import compute_path_context

            self._last_ctx = compute_path_context(
                (x, y, th), self._waypoints, (1.0, 2.0, 3.0)
            )
        except Exception:
            self._last_ctx = None

        # Reward
        reward = self._compute_reward(goal_dist, terminated, truncated)

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        # Provide success flag on termination for eval metrics
        if terminated or truncated:
            info["is_success"] = bool(terminated and goal_dist < 0.5)
            # Include reward breakdown at terminal steps for logging
            if self._last_reward_terms:
                info["reward_terms"] = self._last_reward_terms
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
            "reward_terms": dict(getattr(self, "_last_reward_terms", {})),
        }
        return payload

    def _get_obs(self) -> Dict[str, np.ndarray]:
        x, y, th = self._model.as_pose()
        lidar = self.lidar.sense(self._grid_raw, (x, y, th)).astype(np.float32)

        kin = np.array(
            [self._last_u[0], self._last_u[1], self._prev_u[0], self._prev_u[1]],
            dtype=np.float32,
        )

        # Path context: reuse if already computed this step, else compute now
        if getattr(self, "_last_ctx", None) is None:
            from .path_utils import compute_path_context

            self._last_ctx = compute_path_context(
                (x, y, th), self._waypoints, (1.0, 2.0, 3.0)
            )
        path = np.array(
            [
                self._last_ctx.d_lat,
                self._last_ctx.theta_err,
                *self._last_ctx.previews_robot.flatten().tolist(),
            ],
            dtype=np.float32,
        )

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

    def _compute_reward(
        self, goal_dist_t: float, terminated: bool, truncated: bool = False
    ) -> float:
        # Compute raw terms using the reward module (includes sparse decision)
        # Pass cached context via reward_cfg to avoid recomputing inside reward module
        reward_cfg_with_ctx = dict(self.reward_cfg)
        reward_cfg_with_ctx["_ctx"] = getattr(self, "_last_ctx", None)
        terms, new_prev_goal = compute_terms(
            self._model.as_pose(),
            self._waypoints,
            self._prev_goal_dist,
            self._last_u,
            self._prev_u,
            self.robot_cfg,
            reward_cfg_with_ctx,
            terminated,
            truncated,
        )
        self._prev_goal_dist = new_prev_goal

        weights = self.reward_cfg["weights"]
        total, contrib = apply_weights(terms, weights)
        # Pack for renderer/logging
        self._last_reward_terms = to_breakdown_dict(terms, weights, total, contrib)
        return float(total)
