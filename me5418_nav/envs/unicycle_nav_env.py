from __future__ import annotations

from typing import Tuple, Dict, Optional
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from me5418_nav.models.unicycle import UnicycleModel, UnicycleState
from me5418_nav.constants import (
    GRID_RESOLUTION_M,
    ROBOT_DIAMETER_M,
    ROBOT_COLLISION_RADIUS_M,
    GOAL_TOLERANCE_M,
    ROBOT_V_MIN_MPS,
)
from me5418_nav.navigation.path_tracker import PathTracker
from me5418_nav.sensors.lidar import Lidar
from me5418_nav.scenarios.blockage import (
    BlockageScenarioConfig,
    create_blockage_scenario,
)
from me5418_nav.visualization.pygame_viewer import PygameViewer
from me5418_nav.config import EnvConfig
from me5418_nav.observation import Observation
from me5418_nav.config import CurriculumManager
from scipy.ndimage import binary_dilation
from me5418_nav.utils.connectivity import reachable_inflated
from me5418_nav.rewards import build_reward


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _make_disk_selem(radius_cells: int) -> np.ndarray:
    r = int(max(0, radius_cells))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


class UnicycleNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human", "none"]}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # Accept either a raw dict or an EnvConfig instance
        if isinstance(config, EnvConfig):
            self.conf = config
        else:
            self.conf = EnvConfig.from_dict(config)
        self.dt = self.conf.dynamics.dt
        self.v_max = self.conf.dynamics.v_max
        self.w_max = self.conf.dynamics.w_max
        self.render_mode = self.conf.render_mode
        self._default_scenario = self.conf.scenario
        self._default_scenario_kwargs = dict(self.conf.scenario_kwargs)

        self.lidar = Lidar(
            self.conf.lidar.beams,
            self.conf.lidar.fov_deg,
            self.conf.lidar.max_range_m,
            self.conf.lidar.step_m,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = self.conf.lidar.beams + 2 + 2 + 2 * self.conf.preview.K
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self._veh = UnicycleModel()
        self._grid_raw = None
        self._grid_infl = None
        self._map_w_m = 0.0
        self._map_h_m = 0.0
        self._tracker: Optional[PathTracker] = None
        self._goal_xy = (0.0, 0.0)
        self._steps = 0
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_lidar_ranges = None
        # Build reward implementation from config dict (select by name if provided)
        rparams = dict(self.conf.reward or {})
        rname = str(rparams.pop("name", rparams.pop("impl", "v1")))
        self._reward = build_reward(
            name=rname,
            params=rparams,
            dt=self.dt,
            v_max=self.v_max,
            w_max=self.w_max,
            lidar_max_range=self.lidar.max_range,
        )
        self._viewer: Optional[PygameViewer] = None

        # Curriculum learning system
        self._curriculum = (
            CurriculumManager(self.conf.curriculum)
            if self.conf.curriculum.enabled
            else None
        )
        self._episode_count = 0

    def _inflate_grid(self, grid: np.ndarray) -> np.ndarray:
        r_cells = int(math.ceil(ROBOT_COLLISION_RADIUS_M / GRID_RESOLUTION_M))
        selem = _make_disk_selem(r_cells)
        return binary_dilation(grid, structure=selem)

    def _pose(self) -> Tuple[float, float, float]:
        s = self._veh.get_state()
        return float(s.x), float(s.y), float(s.theta)

    def _observe(
        self,
        lidar_data: Optional[Tuple[np.ndarray, float]] = None,
        path_errors: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        x, y, th = self._pose()

        # Use provided LiDAR data if available, otherwise compute fresh
        if lidar_data is not None:
            ranges, _ = lidar_data
        else:
            ranges, _ = self.lidar.sense(
                self._grid_raw,
                (x, y, th),
                GRID_RESOLUTION_M,
                self._map_w_m,
                self._map_h_m,
            )
        lidar_norm = np.clip(ranges / self.conf.lidar.max_range_m, 0.0, 1.0)

        # Use provided path errors if available, otherwise compute fresh
        if path_errors is not None:
            e_lat, e_head = path_errors
        else:
            e_lat, e_head = self._tracker.errors((x, y, th))
        e_lat_n = float(np.clip(e_lat, -1.0, 1.0))
        e_head_n = float(e_head / math.pi)  # Remove duplicate wrapping
        previews = self._tracker.preview_points(
            self.conf.preview.K, self.conf.preview.ds
        )
        rel = previews - np.array([x, y])[None, :]
        c = math.cos(th)
        s = math.sin(th)
        # Transform world->robot for row-vectors: use R(-theta)^T = [[c, -s], [s, c]]
        # This makes +y in robot frame point to the robot's left, consistent with e_lat sign.
        rot_T = np.array([[c, -s], [s, c]])
        rel_robot = rel @ rot_T
        rel_robot = np.clip(rel_robot / self.conf.preview.range_m, -1.0, 1.0)
        v = self._veh.get_state().v
        w = self._veh.get_state().omega
        # Allow symmetric normalization to reflect limited reverse capability
        v_n = float(np.clip(v / self.v_max, -1.0, 1.0))
        w_n = float(np.clip(w / self.w_max, -1.0, 1.0))
        obs_struct = Observation(
            lidar=lidar_norm,
            kinematics=np.array([v_n, w_n], dtype=np.float32),
            path_errors=np.array([e_lat_n, e_head_n], dtype=np.float32),
            preview=rel_robot.astype(np.float32),
        )
        return obs_struct.flatten()

    # Reward logic is delegated to me5418_nav.rewards; no local reward function.

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        scenario = (options or {}).get("scenario", self._default_scenario)
        kwargs = (options or {}).get("kwargs", self._default_scenario_kwargs)

        # Apply curriculum learning if enabled
        if self._curriculum:
            stage = self._curriculum.get_stage_config(self._episode_count)
            # Check if stage changed for logging
            if self._curriculum.should_update_stage(self._episode_count):
                print(
                    f"[CURRICULUM] Episode {self._episode_count}: {self._curriculum.get_stage_name(self._episode_count)}"
                )
            # Create curriculum-controlled configuration
            cfg = BlockageScenarioConfig.from_curriculum_stage(stage)
        else:
            # Use static configuration with single pallet count
            num_pallets = int(kwargs.get("num_pallets", 1))
            cfg = BlockageScenarioConfig(
                num_pallets_min=num_pallets, num_pallets_max=num_pallets
            )

        resample_count = 0
        while True:
            if scenario == "blockage":
                grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(
                    cfg, rng
                )
            else:
                raise ValueError("unknown scenario")
            grid_infl = self._inflate_grid(grid)
            start_xy = (float(start_pose[0]), float(start_pose[1]))
            ok = reachable_inflated(grid_infl, start_xy, goal_xy, GRID_RESOLUTION_M)
            if ok:
                break
            resample_count += 1

        self._grid_raw = grid
        self._grid_infl = grid_infl
        self._map_h_m = grid.shape[0] * GRID_RESOLUTION_M
        self._map_w_m = grid.shape[1] * GRID_RESOLUTION_M
        self._goal_xy = goal_xy
        self._tracker = PathTracker(waypoints)
        self._veh.reset(
            UnicycleState(start_pose[0], start_pose[1], start_pose[2], 0.0, 0.0)
        )
        self._steps = 0
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_lidar_ranges = None
        self._viewer = None

        # Increment episode count for curriculum learning
        self._episode_count += 1

        obs = self._observe()
        info_out = {"scenario": scenario, "resample_count": int(resample_count), **info}
        return obs, info_out

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=float)
        a_v = float(np.clip(a[0], -1.0, 1.0))
        a_w = float(np.clip(a[1], -1.0, 1.0))
        # Allow limited reverse: clamp by global min speed from constants
        v_cmd = float(np.clip(a_v * self.v_max, ROBOT_V_MIN_MPS, self.v_max))
        w_cmd = a_w * self.w_max
        v_prev = self._veh.get_state().v
        w_prev = self._veh.get_state().omega
        self._veh.step((v_cmd, w_cmd), self.dt)
        x, y, th = self._pose()
        _, s_ptr, ds, q, t_hat, _ = self._tracker.update_progress(
            np.array([x, y], dtype=float)
        )
        # Compute errors directly without calling tracker.errors() to avoid duplicate projection
        n_hat = np.array([-t_hat[1], t_hat[0]])
        e_lat = float(np.dot(np.array([x, y]) - q, n_hat))
        ang_t = math.atan2(t_hat[1], t_hat[0])
        e_head = _wrap_pi(ang_t - th)
        ranges, min_lidar = self.lidar.sense(
            self._grid_raw, (x, y, th), GRID_RESOLUTION_M, self._map_w_m, self._map_h_m
        )
        self._last_lidar_ranges = ranges
        ii = int(np.clip(y / GRID_RESOLUTION_M, 0, self._grid_infl.shape[0] - 1))
        jj = int(np.clip(x / GRID_RESOLUTION_M, 0, self._grid_infl.shape[1] - 1))
        collision = bool(self._grid_infl[ii, jj])
        oob = bool(x < 0.0 or y < 0.0 or x >= self._map_w_m or y >= self._map_h_m)
        goal = self._tracker.goal_reached(
            np.array([x, y], dtype=float), self._goal_xy, GOAL_TOLERANCE_M
        )
        self._steps += 1
        timeout = bool(self._steps >= int(self.conf.dynamics.max_steps))
        terminated = bool(collision or goal)
        truncated = bool(timeout or oob)
        # Delegate reward computation
        rew, dbg = self._reward.compute(
            ds=ds,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            v_prev=v_prev,
            w_prev=w_prev,
            ranges=ranges,
            angles=self.lidar.rel_angles,
            min_lidar=min_lidar,
            e_lat=e_lat,
            e_head=e_head,
            goal=goal,
            collision=collision,
            timeout=timeout,
            truncated=truncated,
        )
        obs = self._observe(lidar_data=(ranges, min_lidar), path_errors=(e_lat, e_head))
        info = {
            "progress": float(ds),
            "s_ptr": float(s_ptr),
            "e_lat": float(e_lat),
            "e_head": float(e_head),
            "min_lidar": float(min_lidar),
            **{k: float(v) for k, v in (dbg or {}).items()},
            "collision": collision,
            "goal": goal,
            "timeout": timeout,
        }
        if self.render_mode == "human":
            try:
                self.render("human")
            except Exception:
                pass
        return obs, float(rew), terminated, truncated, info

    def render(self, mode: str = "human"):
        if self._grid_raw is None or self._tracker is None:
            return None
        if self._viewer is None:
            try:
                self._viewer = PygameViewer(self._map_w_m, self._map_h_m)
            except Exception:
                return None
        x, y, th = self._pose()
        previews = self._tracker.preview_points(
            self.conf.preview.K, self.conf.preview.ds
        )
        ranges = self._last_lidar_ranges
        if ranges is None:
            ranges, _ = self.lidar.sense(
                self._grid_raw,
                (x, y, th),
                GRID_RESOLUTION_M,
                self._map_w_m,
                self._map_h_m,
            )
        frame = self._viewer.draw(
            self._grid_raw,
            self._tracker.P,
            previews,
            (x, y, th),
            ranges,
            self.lidar.rel_angles,
            self._goal_xy,
            # Draw using the effective collision radius (includes safety margin)
            ROBOT_COLLISION_RADIUS_M,
            capture=(mode == "rgb_array"),
        )
        if mode == "rgb_array":
            return frame
        return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        return None


__all__ = ["UnicycleNavEnv"]
