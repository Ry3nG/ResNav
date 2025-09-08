# Unified AMR Navigation via Residual RL

Residual policy + conventional tracker for AMR local navigation in a 2D factory map. The RL agent outputs a residual action on top of Pure Pursuit. The stack is Gymnasium + Stable‑Baselines3 (PPO) with Hydra configs, VecEnv, and Weights & Biases logging. Baselines include Pure Pursuit and a lightweight DWA.

## Key Features
- **Residual control architecture**: `u_final = clip(u_track + Δu)`
- **Fast 2D simulator**: unicycle dynamics, occupancy grid, DDA LiDAR (24 beams, 240°)
- **Phase I scenarios**: temporary blockage in narrow corridors (domain randomization)
- **Benchmarks and visualizations**: rollout renderer (Pygame), CSV summaries, videos
- **Reproducibility**: Hydra configs, VecNormalize stats saved, seeds per env

## Quickstart

### 1. Install Dependencies
```bash
# Conda environment recommended
conda env create -f environment.yml
# install project as a package
pip install -e .
```


### 2. Training
```bash
# Smoke test (2 envs, 5k steps)
make train-smoke

# Full Phase I training (16 envs, 200k steps)
make train-full
```

### 3. Model Management
```bash
# List available trained models with copy-paste commands
make list-models

# Copy-paste the 📊/🎬 commands from the output above
```

### 4. Visualization
```bash
# Baselines (no training needed)
make render-pp        # Pure Pursuit
make render-dwa       # DWA baseline

# Trained PPO (after training)
make list-models      # Shows copy-paste commands
# Then copy-paste: make render-model MODEL=... VECNORM=... SEED=42

# Render from a checkpoint directory
# make render-ckpt CKPT_DIR=runs/TIMESTAMP/checkpoints/ckpt_step_50000 SEED=42
```

### 5. Benchmarking (先别用)
```bash
# Baselines only
make benchmark-all

# PPO (specify model from list-models)
# Prefer pairing best model with best vecnorm stats
make benchmark-ppo MODEL=runs/TIMESTAMP/best/best_model.zip VECNORM=runs/TIMESTAMP/best/vecnorm_best.pkl
# For final model, pair with final vecnorm stats
# make benchmark-ppo MODEL=runs/TIMESTAMP/final_model.zip VECNORM=runs/TIMESTAMP/vecnorm.pkl

# Checkpoint examples
# make benchmark-ppo MODEL=runs/TIMESTAMP/checkpoints/ckpt_step_50000/model.zip \
#                    VECNORM=runs/TIMESTAMP/checkpoints/ckpt_step_50000/vecnorm.pkl
```

## Configuration

| Component | Config File | Description |
|-----------|-------------|-------------|
| Environment | `configs/env/blockage.yaml` | Map, LiDAR, wrappers |
| Robot | `configs/robot/default.yaml` | Limits, controller, safety margin |
| Reward | `configs/reward/default.yaml` | Sparse, progress, path, effort |
| PPO | `configs/algo/ppo.yaml` | Learning rate, batch size, etc. |
| Policy | `configs/policy/default.yaml` | MLP sizes, activations |
| DWA | `configs/control/dwa.yaml` | Weights, lattice, horizon |
| WandB | `configs/wandb/default.yaml` | Project, mode, tags |

## Make Targets

| Target | Description |
|--------|-------------|
| `make train-smoke` | 5k steps sanity run |
| `make train-full` | 200k steps training |
| `make list-models` | List available trained models with copy-paste commands |
| `make eval-model MODEL=... VECNORM=...` | Evaluate specific model |
| `make render-model MODEL=... VECNORM=...` | Record demo from specific model |
| `make benchmark-ppo MODEL=... VECNORM=...` | Benchmark specific PPO model |
| `make render-pp` | Visualize Pure Pursuit |
| `make render-dwa` | Visualize DWA |
| `make test` | Unit tests |

## Design Overview

### Environment Observation (Dict)
- **lidar**: 24 distances (meters); stacked by Vec wrapper (default K=4 → 96)
- **kin**: `(v_t, w_t, v_{t-1}, w_{t-1})`
- **path**: `(d_lat, θ_err, 3 preview waypoints in robot frame)`

### Reward Components
- **Progress**: `d_{t-1} - d_t`
- **Path penalty**: `-|d_lat| - 0.5|θ_err|`
- **Effort**: `-λ_v|Δv| - λ_ω|Δω|`
- **Sparse**: goal +200, collision -200

### Technical Details
- **LiDAR**: DDA raycasting on raw occupancy; collision uses inflated grid
- **DWA**: lattice forward-sim (2s), obstacle-safe cost, path dead-zone, speed bias

## Tips & Troubleshooting

- **Model management**: Use `make list-models` for copy-paste ready commands
- **VecNormalize**: Playback requires matching stats from training
- **Renderer**: Shows raw meters/radians (not normalized) for geometry overlays
- **PPO shape errors**: Ensure frame stack (K) matches training config
- **DWA tuning**: Adjust `configs/control/dwa.yaml` (path weight, dead-zone, d_free/safe)

## Status

✅ **Phase I Complete**: Baselines (PP, DWA), PPO residual, benchmarks, videos
🚧 **Next phases**: Counter-flow, occlusions will extend env + curriculum

