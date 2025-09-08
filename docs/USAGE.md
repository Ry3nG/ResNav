# Usage Cheat Sheet

## Setup
```bash
pip install -r requirements.txt
```
Optional: Set WandB mode in `configs/wandb/default.yaml` (online/offline/disabled)

## Training
```bash
make train-smoke    # Quick test (2 envs, 5k steps)
make train-full     # Full training (16 envs, 200k steps)
```

### Enable periodic checkpoints (optional)
Edit `configs/algo/ppo.yaml`:
```yaml
checkpoint:
  enabled: true
  every_steps: 50000   # save frequency in timesteps
  keep_last_k: 3       # keep the most recent K
  prefix: ckpt
  to_wandb: false      # set true to upload each checkpoint
```

## Model Management
```bash
make list-models    # Shows copy-paste ready commands with 📊/🎬 icons
```
**Copy-paste workflow**: Run `make list-models`, then copy the command you want

## Visualization
```bash
# Baselines (no training needed)
make render-pp      # Pure Pursuit
make render-dwa     # DWA baseline

# Trained PPO
make list-models    # Get copy-paste commands
# Then use: make render-model MODEL=... VECNORM=... SEED=42
# If using best model, prefer VECNORM=runs/TIMESTAMP/best/vecnorm_best.pkl
# Otherwise, use VECNORM=runs/TIMESTAMP/vecnorm.pkl (final stats)
```

## Benchmarking
```bash
# Baselines
python eval/benchmark.py --agent pp --episodes 100
python eval/benchmark.py --agent dwa --episodes 100 --dwa_cfg configs/control/dwa.yaml

# PPO (use copy-paste from list-models)
make benchmark-ppo MODEL=runs/TIMESTAMP/best/best_model.zip VECNORM=runs/TIMESTAMP/best/vecnorm_best.pkl
# Alternatively, for final model:
# make benchmark-ppo MODEL=runs/TIMESTAMP/final_model.zip VECNORM=runs/TIMESTAMP/vecnorm.pkl

# Benchmark a specific checkpoint (example):
# make benchmark-ppo MODEL=runs/TIMESTAMP/checkpoints/ckpt_step_50000/model.zip \
#                      VECNORM=runs/TIMESTAMP/checkpoints/ckpt_step_50000/vecnorm.pkl
```

## Configuration Files

| Component | File | Key Settings |
|-----------|------|--------------|
| Environment | `configs/env/blockage.yaml` | Map size, LiDAR params, wrappers |
| Robot | `configs/robot/default.yaml` | Speed limits, safety margins |
| Rewards | `configs/reward/default.yaml` | Reward weights, sparse values |
| PPO | `configs/algo/ppo.yaml` | Learning rate, batch size, epochs |
| Policy | `configs/policy/default.yaml` | Network architecture |
| DWA | `configs/control/dwa.yaml` | Weights, lattice, horizon |

## Notes
- Each training run creates timestamped directory: `runs/YYYYMMDD_HHMMSS/`
- Use `make list-models` to see available trained models with full paths
- Renderer uses raw env observations for geometry overlays
- Disable LiDAR noise in config for cleaner demo videos
- Evaluation CSVs written to `runs/bench_*.csv`
- VecNormalize: `best_model.zip` pairs with `best/vecnorm_best.pkl`; `final_model.zip` pairs with `vecnorm.pkl`
- Checkpoints: use `.../checkpoints/ckpt_step_XXXXXX/model.zip` with matching `vecnorm.pkl`
