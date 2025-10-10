# Unified AMR Navigation via Residual RL

Residual policy + conventional tracker for AMR local navigation in a 2D factory map. The RL agent outputs a residual action on top of Pure Pursuit. The stack is Gymnasium + Stable‑Baselines3 (PPO) with Hydra configs, VecEnv, and Weights & Biases logging. Baseline includes Pure Pursuit.

## Key Features
- Residual control: `u_final = clip(u_track + Δu)`
- Fast 2D simulator: unicycle dynamics, occupancy grid, DDA LiDAR (24 beams, 240°)
- Reward HUD & logging: total + per-term breakdown; logs to TB/W&B
- Reproducibility: Hydra configs, resolved snapshots, VecNormalize stats
- Phase I scenarios: temporary blockage in narrow corridors

## Quickstart

### 1) Setup (one-time)
```bash
make setup
```
This creates/updates the conda env `amr-nav` from `environment.yml` and runs `pip install -e .` inside it.

### 2) Launch
```bash
make amr
```
The interactive launcher guides you to train and render models with sensible defaults.

## Repository Layout

```
amr_env/                 # installable package (pip install -e .)
├── control/             # classical controllers (pure pursuit, etc.)
├── gym/                 # Gymnasium-facing wrappers and services
├── planning/            # path geometry utilities (projection, previews)
├── sim/                 # map/scenario generation, dynamics, LiDAR, EDT
├── utils/               # config helpers, run discovery
├── viz/                 # pygame renderer + video export
└── reward.py            # reward math used by the env & logger

training/                # SB3/Hydra training utilities (env factory, callbacks, feature extractors)
tools/                   # CLI launcher and misc scripts
configs/                 # Hydra config groups
```

> The `training/` and `tools/` folders stay top-level so Hydra and SB3 discover configs and scripts without extra package plumbing. Everything that defines the environment lives under `amr_env`.

## Configuration

| Component | Config File | Description |
|-----------|-------------|-------------|
| Environment | `configs/env/blockage.yaml` | Map, LiDAR, wrappers |
| Robot | `configs/robot/default.yaml` | Limits (`v_min`, `v_max`, `w_max`), controller, safety margin |
| Reward | `configs/reward/default.yaml` | Sparse, progress, path, effort |
| PPO | `configs/algo/ppo.yaml` | Learning rate, batch size, etc. |
| Network | `configs/network/default.yaml` | MLP sizes, activations |
| WandB | `configs/wandb/default.yaml` | Project, mode, tags |

### Adding New Maps Quickly
- `ScenarioManager` now switches on `env.name`; `name: t_junction` is provided in `configs/env/t_junction.yaml`.
- For a step-by-step recipe (scenario generator + config glue), read `docs/ARCHITECTURE.md#adding-a-new-map`.

## Design Overview

### Observation (Dict)
- lidar: 24 distances (stacked by wrapper)
- kin: `(v_t, w_t, v_{t-1}, w_{t-1})`
- path: `(d_lat, θ_err, 3× preview waypoints in robot frame)`

### Reward Components (`amr_env/reward.py`)
- Progress: `d_{t-1} - d_t`
- Path penalty: `-|d_lat| - 0.5|θ_err|`
- Effort: `-λ_v|Δv| - λ_ω|Δω| - λ_jerk(|Δv|+|Δω|)`
- Sparse: goal +200, collision -200, timeout -50

Reward schema exposed per step (for HUD/logging):
- raw: unweighted terms
- weights: config weights (keys match contrib)
- contrib: weighted contributions
- total: scalar sum

## Tips & Troubleshooting
- VecNormalize files are saved as:
  - best/: `best_model.zip` + `vecnorm_best.pkl`
  - final/: `final_model.zip` + `vecnorm_final.pkl`
  - checkpoints/ckpt_step_N/: `model.zip` + `vecnorm.pkl`
- When rendering, the loader prefers `vecnorm_best.pkl` or `vecnorm_final.pkl`; otherwise falls back to `vecnorm.pkl` in checkpoints.
- Renderer shows geometry overlays in physical units (meters/radians). Module path: `amr_env.viz.pygame_renderer.Renderer`.
- Reward logs stream to TensorBoard by default and mirror to W&B when enabled.

## Testing
- Lightweight regression tests live under `tests/` and exercise pure pursuit, path projection, LiDAR, inflation, and EDT helpers.
- Run `pytest` (after `pip install -e .`) to ensure math kernels stay stable across refactors.

## Development Notes
- **Python 3.10+ required**: The codebase uses modern type hints (`tuple[...]`, `dict[...]`, `list[...]`) via `from __future__ import annotations`.
- **Pre-commit hooks**: Run `pre-commit install` to enable automatic linting (ruff, black, mypy) on commit.
- **Package structure**: Installable via `pip install -e .` for clean imports in Hydra subprocess workers.
