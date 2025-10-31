"""Residual navigation Gymnasium environment (Phase I core).

Observation Dict keys:
- lidar: 1xN array of distances (no stacking here; wrapper stacks later)
- kin: (v_t, ω_t, v_{t-1}, ω_{t-1})
- path: (d_lat, θ_err, x1, y1, x2, y2, x3, y3) in robot frame

Action: residual Δu added to Pure Pursuit u_track, clipped to robot limits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from amr_env.sim.dynamics import UnicycleModel, UnicycleState
from amr_env.sim.lidar import GridLidar
from amr_env.control.pure_pursuit import compute_u_track
from amr_env.sim import movers

from .observation_builder import ObservationBuilder, ObservationData
from .reward_manager import RewardManager
from amr_env.sim.scenario_manager import ScenarioManager, ScenarioSample


class ResidualNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_cfg: dict[str, Any],
        robot_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        run_cfg: dict[str, Any],
    ):
        super().__init__()
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg
        self.reward_cfg = reward_cfg
        self.run_cfg = run_cfg

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

        # Helper components
        self._scenario_manager = ScenarioManager(
            env_cfg, robot_radius_m=self.radius_m, resolution_m=self.resolution_m
        )
        self._obs_builder = ObservationBuilder(self.lidar)
        self._reward_manager = RewardManager(robot_cfg, reward_cfg)

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
        d_lat_bound = float(self.env_cfg["map"]["corridor_width_m"][1])
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
        self._scenario: ScenarioSample | None = None
        self._model: UnicycleModel | None = None
        self._last_u = (0.0, 0.0)
        self._prev_u = (0.0, 0.0)
        self._steps = 0
        self._max_steps = int(run_cfg["max_steps"])
        self._last_reward_terms: dict[str, Any] = {}
        self._last_obs: dict[str, np.ndarray] = {}
        self._grid_dyn_raw: np.ndarray | None = None
        self._grid_dyn_infl: np.ndarray | None = None
        self._grid_raw_curr: np.ndarray | None = None
        self._grid_infl_curr: np.ndarray | None = None
        self._active_movers: list[movers.DiscMover] = []
        self._next_lateral_spawn_t = 0.0
        self._next_longitudinal_spawn_t = 0.0
        self._t = 0.0

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._scenario_manager.set_seed(seed)
            self.lidar.set_seed(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self._scenario = self._scenario_manager.sample()
        self._reward_manager.reset()
        self._obs_builder.reset()

        self._model = UnicycleModel(
            v_max=self.v_max, w_max=self.w_max, v_min=self.v_min
        )
        x0, y0, th0 = self._scenario.start_pose
        self._model.reset(UnicycleState(x0, y0, th0, 0.0, 0.0))

        self._prev_u = (0.0, 0.0)
        self._last_u = (0.0, 0.0)
        self._steps = 0
        self._last_reward_terms = {}
        self._last_obs = {}
        self._t = 0.0

        H, W = self._scenario.grid_raw.shape
        self._grid_dyn_raw = np.zeros((H, W), dtype=bool)
        self._grid_dyn_infl = np.zeros((H, W), dtype=bool)
        self._init_spawner()
        if self.env_cfg.get("dynamic_movers", {}).get("enabled", False):
            self._update_dynamic_grids(0.0)
        self._compose_current_grids()

        obs = self._build_observation().obs
        # Do not set SB3's episode info; RecordEpisodeStatistics will populate it
        return obs, {}

    def _init_spawner(self) -> None:
        """Initialize Poisson spawner with timers set in the future."""
        self._active_movers = []

        dyn_cfg = self.env_cfg.get("dynamic_movers", {})
        if not dyn_cfg.get("enabled", False):
            return

        scenario_name = str(self.env_cfg.get("name", "blockage")).lower()
        if scenario_name != "omcf":
            return

        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        # Read spawn rates
        lat_cfg = dyn_cfg.get("lateral", {})
        lon_cfg = dyn_cfg.get("longitudinal", {})
        rate_lat = max(1e-6, float(lat_cfg.get("spawn_rate_hz", 0.15)))
        rate_lon = max(1e-6, float(lon_cfg.get("spawn_rate_hz", 0.10)))

        # Initialize timers to future (critical: clean start on reset)
        self._next_lateral_spawn_t = self._t + float(rng.exponential(1.0 / rate_lat))
        self._next_longitudinal_spawn_t = self._t + float(
            rng.exponential(1.0 / rate_lon)
        )

    def _update_dynamic_grids(self, dt: float) -> None:
        """Four-phase update: spawn → step → cull → rasterize."""
        if self._grid_dyn_raw is None or self._grid_dyn_infl is None:
            return

        dyn_cfg = self.env_cfg.get("dynamic_movers", {})
        if not dyn_cfg.get("enabled", False):
            return

        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        # Read guardrail parameters
        max_concurrent = int(dyn_cfg.get("max_concurrent", 30))
        kill_margin = float(dyn_cfg.get("kill_margin_m", 0.5))
        reflect_walls = bool(dyn_cfg.get("reflect_walls", True))
        v_lo, v_hi = dyn_cfg.get("velocity_range_mps", [0.4, 1.5])

        map_width = float(self.env_cfg["map"]["size_m"][0])
        map_height = float(self.env_cfg["map"]["size_m"][1])
        y_top = float(self._scenario.info.get("y_top", map_height / 2 + 2))
        y_bot = float(self._scenario.info.get("y_bot", map_height / 2 - 2))

        lat_cfg = dyn_cfg.get("lateral", {})
        lon_cfg = dyn_cfg.get("longitudinal", {})
        rate_lat = max(1e-6, float(lat_cfg.get("spawn_rate_hz", 0.15)))
        rate_lon = max(1e-6, float(lon_cfg.get("spawn_rate_hz", 0.10)))

        # === Phase 1: Spawn new movers ===
        agent_pose = self._model.as_pose()

        # Lateral spawning
        while self._t >= self._next_lateral_spawn_t:
            if len(self._active_movers) < max_concurrent:
                new_mover = movers.spawn_lateral_mover(
                    self.env_cfg,
                    self._scenario.info,
                    self._scenario.grid_inflated,
                    self.resolution_m,
                    agent_pose,
                    self._t,
                    rng,
                )
                if new_mover is not None:
                    self._active_movers.append(new_mover)
            # Advance timer regardless of success/failure (avoid starvation)
            self._next_lateral_spawn_t += float(rng.exponential(1.0 / rate_lat))

        # Longitudinal spawning
        if lon_cfg.get("spawn_from_holes", True):
            while self._t >= self._next_longitudinal_spawn_t:
                if len(self._active_movers) < max_concurrent:
                    new_mover = movers.spawn_longitudinal_mover(
                        self.env_cfg,
                        self._scenario.info,
                        self._scenario.grid_inflated,
                        self.resolution_m,
                        agent_pose,
                        self._t,
                        rng,
                    )
                    if new_mover is not None:
                        self._active_movers.append(new_mover)
                self._next_longitudinal_spawn_t += float(rng.exponential(1.0 / rate_lon))

        # === Phase 2: Step all movers ===
        for mover in self._active_movers:
            mover.step(dt, self._t, v_lo, v_hi, y_bot, y_top, reflect_walls)

        # === Phase 3: Cull expired/out-of-bounds movers ===
        for mover in self._active_movers:
            # TTL already handled in mover.step(), check OOB here
            if (
                mover.x < -kill_margin
                or mover.x > map_width + kill_margin
                or mover.y < -kill_margin
                or mover.y > map_height + kill_margin
            ):
                mover.active = False

        self._active_movers = [m for m in self._active_movers if m.active]

        # === Phase 4: Rasterize to grids ===
        self._grid_dyn_raw.fill(False)
        self._grid_dyn_infl.fill(False)
        inflate_extra = self.radius_m

        for mover in self._active_movers:
            if self._t < mover.spawn_t:  # Double-check spawn time
                continue
            movers.rasterize_disc(
                self._grid_dyn_raw, mover.x, mover.y, mover.radius_m, self.resolution_m
            )
            movers.rasterize_disc(
                self._grid_dyn_infl,
                mover.x,
                mover.y,
                mover.radius_m + inflate_extra,
                self.resolution_m,
            )

    def _compose_current_grids(self) -> None:
        scenario = self._scenario
        if scenario is None:
            self._grid_raw_curr = None
            self._grid_infl_curr = None
            return
        if self._grid_dyn_raw is not None:
            self._grid_raw_curr = np.logical_or(scenario.grid_raw, self._grid_dyn_raw)
        else:
            self._grid_raw_curr = scenario.grid_raw
        if self._grid_dyn_infl is not None:
            self._grid_infl_curr = np.logical_or(
                scenario.grid_inflated, self._grid_dyn_infl
            )
        else:
            self._grid_infl_curr = scenario.grid_inflated

    def _clearance_iso(self) -> float:
        """Deterministic clearance via bilinear sampling over the EDT grid."""
        scenario = self._scenario
        if scenario is None or scenario.edt is None:
            return 1e9

        x, y, _ = self._model.as_pose()
        i_f = y / self.resolution_m
        j_f = x / self.resolution_m

        edt = scenario.edt
        H, W = edt.shape
        if i_f < 0.0 or j_f < 0.0 or i_f >= H - 1 or j_f >= W - 1:
            return 0.0

        i0 = int(np.floor(i_f))
        j0 = int(np.floor(j_f))
        di = float(i_f - i0)
        dj = float(j_f - j0)

        v00 = float(edt[i0, j0])
        v01 = float(edt[i0, j0 + 1])
        v10 = float(edt[i0 + 1, j0])
        v11 = float(edt[i0 + 1, j0 + 1])

        v0 = v00 * (1.0 - dj) + v01 * dj
        v1 = v10 * (1.0 - dj) + v11 * dj
        return float(v0 * (1.0 - di) + v1 * di)

    def step(self, action: np.ndarray):
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before stepping")

        dv, dw = float(action[0]), float(action[1])

        lookahead = self.robot_cfg["controller"]["lookahead_m"]
        v_nom = self.robot_cfg["controller"]["speed_nominal"]
        v_track, w_track = compute_u_track(
            self._model.as_pose(), self._scenario.waypoints, lookahead, v_nom
        )
        v_cmd = float(np.clip(v_track + dv, self.v_min, self.v_max))
        w_cmd = float(np.clip(w_track + dw, -self.w_max, self.w_max))

        self._prev_u = self._last_u
        self._last_u = (v_cmd, w_cmd)
        self._model.step((v_cmd, w_cmd), self.dt)
        self._t += self.dt

        dyn_cfg = self.env_cfg.get("dynamic_movers", {})
        scenario_name = str(self.env_cfg.get("name", "blockage")).lower()
        dyn_enabled = bool(dyn_cfg.get("enabled", False)) and scenario_name == "omcf"
        if dyn_enabled:
            self._update_dynamic_grids(self.dt)
            self._compose_current_grids()
        else:
            self._grid_raw_curr = self._scenario.grid_raw
            self._grid_infl_curr = self._scenario.grid_inflated

        terminated = False
        truncated = False

        if self._is_collision():
            terminated = True

        self._steps += 1
        if self._steps >= self._max_steps and not terminated:
            truncated = True

        goal_dist = self._dist_to_goal()
        if goal_dist < 0.5 and not (terminated or truncated):
            terminated = True

        obs_data = self._build_observation()

        safety_cfg = self.reward_cfg.get("safety") or {}
        if not isinstance(safety_cfg, dict):
            safety_cfg = {}
        use_map_barrier = str(safety_cfg.get("source", "")).lower() == "map"
        clearance = None
        if use_map_barrier:
            true_ranges = self.lidar.sense(
                self._grid_raw_curr, self._model.as_pose(), noise=False
            )
            static_clear = self._clearance_iso()
            clearance = min(static_clear, float(np.min(true_ranges)))

        reward_result = self._reward_manager.compute(
            self._model.as_pose(),
            self._scenario.waypoints,
            self._last_u,
            self._prev_u,
            terminated,
            truncated,
            obs_data.context,
            clearance,
        )

        breakdown = dict(reward_result.breakdown)
        breakdown.setdefault("metrics", {})
        breakdown["metrics"]["edt_ms"] = float(self._scenario.edt_ms)
        self._last_reward_terms = breakdown
        self._reward_manager.update_last_breakdown(breakdown)
        reward = float(reward_result.total)

        obs = obs_data.obs
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["is_success"] = bool(terminated and goal_dist < 0.5)
            if breakdown:
                info["reward_terms"] = breakdown
                metrics = breakdown.get("metrics", {})
                if metrics:
                    info.setdefault("metrics", {}).update(metrics)
        return obs, reward, terminated, truncated, info

    # Debug/visualization helper to avoid poking private fields externally
    def get_render_payload(self) -> dict[str, Any]:
        pose = self._model.as_pose()
        scenario = self._scenario
        payload = {
            "raw_grid": None if scenario is None else self._grid_raw_curr,
            "inflated_grid": None if scenario is None else self._grid_infl_curr,
            "pose": pose,
            "radius_m": self.radius_m,
            "lidar": {
                "beams": self.lidar.beams,
                "fov_rad": self.lidar.fov_rad,
                "max_range": self.lidar.max_range,
            },
            "waypoints": None if scenario is None else scenario.waypoints,
            "last_u": self._last_u,
            "prev_u": self._prev_u,
            "obs": self._last_obs if self._last_obs else self._get_obs(),
            "reward_terms": dict(self._last_reward_terms),
            "scenario_info": {} if scenario is None else dict(scenario.info),
            "active_movers_count": len(self._active_movers),
            "next_spawn_eta_lat": max(0.0, self._next_lateral_spawn_t - self._t),
            "next_spawn_eta_lon": max(0.0, self._next_longitudinal_spawn_t - self._t),
            "movers": [m for m in self._active_movers if m.active and self._t >= m.spawn_t],  # Active visible movers
        }
        return payload

    def _get_obs(self) -> dict[str, np.ndarray]:
        return self._build_observation().obs

    def _is_collision(self) -> bool:
        x, y, _ = self._model.as_pose()
        return self._point_in_inflated(x, y)

    def _point_in_inflated(self, x: float, y: float) -> bool:
        if self._grid_infl_curr is None:
            return False
        grid = self._grid_infl_curr
        i = int(np.floor(y / self.resolution_m))
        j = int(np.floor(x / self.resolution_m))
        H, W = grid.shape
        if i < 0 or i >= H or j < 0 or j >= W:
            return True
        return bool(grid[i, j])

    def _dist_to_goal(self) -> float:
        x, y, _ = self._model.as_pose()
        gx, gy = self._scenario.goal_xy
        return float(np.hypot(gx - x, gy - y))

    def _build_observation(self) -> ObservationData:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before building observations")
        data = self._obs_builder.build(
            self._grid_raw_curr,
            self._model.as_pose(),
            self._scenario.waypoints,
            self._last_u,
            self._prev_u,
        )
        self._last_obs = data.obs
        return data
