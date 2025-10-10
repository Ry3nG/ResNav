# Architecture Overview

## Stack Layers
- **Simulation (`amr_env.sim`)**: occupancy-grid map builders, inflation, EDT, LiDAR, unicycle dynamics.
- **Environment (`amr_env.gym`)**: wraps the simulator into `ResidualNavEnv`, handles observations, rewards, curriculum.
- **Training (`training/`)**: Stable-Baselines3 factories, Hydra configs, callbacks, rollout helpers.
- **Visualization (`amr_env.viz`)**: pygame renderer plus `training/rollout.py` video export tooling.

Data flows sim → env → policy → renderer/loggers, always in meters/radians.

## Frames & Conventions
- Occupancy grid index order `[i, j] = [y, x]`, `True` denotes occupied cells.
- World frame is 2D `(x, y)` in meters; robot pose is `(x, y, θ)` with θ in radians.
- Distances and velocities use SI units; time step defaults to `0.1 s`.
- LiDAR beams sweep counter-clockwise with beam 0 at `θ - fov/2`.

## Observations & Actions
- Observation dict:
  - `lidar`: stacked range scans (by default 24 beams × 4 frames).
  - `kin`: current linear/angular velocities plus previous step.
  - `path`: lateral error, heading error, and waypoint preview in robot frame.
- Action: residual twist `(Δv, Δω)` added to the Pure Pursuit tracker output and clipped to robot limits.

## Adding a New Map
1. **Scenario generator**: create `amr_env/sim/scenarios_<name>.py` with a dataclass config and a `create_<name>` function returning `(grid, waypoints, start_pose, goal_xy, info)`. See `scenarios_tjunction.py` for a 50-line template.
2. **Scenario manager hook**: extend `ScenarioManager.sample` with a new branch keyed on `env.name`, instantiate your config from the Hydra map block, and set `generator = create_<name>`.
3. **Config file**: add `configs/env/<name>.yaml` with `name: <name>` and whatever map keys your generator expects. Reuse existing sections (wrappers, lidar, viz) to stay consistent.
4. **Train or roll out**: point `python training/train_ppo.py env=<name>` or `python training/rollout.py --env_cfg configs/env/<name>.yaml` once the config is in place.

## Reproducing Results
1. `conda env create -f environment.yml && conda activate amr-nav`
2. `pip install -e .`
3. `python training/train_ppo.py` (defaults to blockage) or `python training/train_ppo.py env=t_junction` for the new scenario.
4. Use `training/rollout.py --render` or `--record out.mp4` to inspect trajectories.

## Code Conventions
- **Type hints**: Uses Python 3.10+ built-in generics (`tuple[...]`, `dict[...]`) with `from __future__ import annotations` for forward compatibility.
- **Grid indexing**: `[i, j]` maps to `[y, x]` in world coordinates; `True` denotes occupied cells.
- **Angle wrapping**: All heading angles wrapped to `[-π, π]` via `wrap_to_pi` in dynamics module.
