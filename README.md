# ME5418 AMR Bottleneck Navigation

A reinforcement learning environment for Autonomous Mobile Robot (AMR) navigation in industrial bottleneck scenarios. This project focuses on learning continuous control policies that can handle temporary blockages, occluded intersections, and narrow counter-flows without online global replanning.

## Key Features

- **Continuous Control**: Learns (v, ω) velocities for differential-drive robots
- **Sparse Sensing**: Uses only local 2D LiDAR (24 beams, 240° FOV, 4m range)
- **Path Tracking**: Maintains continuous progress along pre-computed global paths
- **Industrial Scenarios**: Specialized for warehouse/factory bottleneck navigation
- **Modular Design**: Clean architecture with configurable components

## Quick Start

### Installation

```bash
git clone <repository-url>
cd ME5418-Project
conda env create -f environment.yml
conda activate me5418-nav
```

### Train a Policy

```bash
python scripts/train.py --config configs/ppo_default.yaml --outdir runs/ppo
```

### Evaluate and Generate GIF

```bash
# Evaluate the most recent training run and save a GIF
LATEST_RUN=$(ls -td runs/ppo/* | head -n1)
python scripts/eval.py \
  --model "$LATEST_RUN/best_model.zip" \
  --episodes 50 \
  --gif runs/eval/ppo_eval.gif
```

## Environment Details

### Observation Space (structured)
- Shape (flattened): `L + 2 + 2 + 2K`
- Semantic fields:
  - **LiDAR** `(L,)`: normalized distances [0,1]
  - **Kinematics** `(2,)`: [v_norm, ω_norm]
  - **Path errors** `(2,)`: [e_lat_norm, e_head_norm]
  - **Preview** `(K,2)`: future waypoints in robot frame (clipped and normalized)

### Action Space (2-dim)
- **Linear velocity**: v ∈ [0, 1.5] m/s (mapped from [-1,1])
- **Angular velocity**: ω ∈ [-2, 2] rad/s (direct mapping)

### Reward Function
```python
reward = w_prog * progress +
         - w_lat * |lateral_error| +
         - w_head * |heading_error| +
         - w_clear * exp(-min_lidar_dist/safe_dist) +
         - w_smoothness * |velocity_changes| +
         + terminal_rewards
```

## Programmatic Usage

### Basic Environment

```python
from me5418_nav.envs import UnicycleNavEnv

env = UnicycleNavEnv()
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Custom Configuration

```python
from me5418_nav.config import EnvConfig, RewardConfig, LidarConfig, PathPreviewConfig
from me5418_nav.envs import UnicycleNavEnv

# Create custom config
config = EnvConfig(
    reward=RewardConfig(w_prog=1.5, w_lat=0.1, clearance_safe_m=0.5),
    lidar=LidarConfig(beams=36, fov_deg=270, max_range_m=4.0),
    preview=PathPreviewConfig(K=5, ds=0.6, range_m=3.0),
    scenario="blockage",
    scenario_kwargs={"num_pallets": 2}  # or drive via curriculum (see YAML)
)

env = UnicycleNavEnv(config)
```

### Training with Stable-Baselines3 (quick demo)

```python
from stable_baselines3 import PPO
from me5418_nav.envs import UnicycleNavEnv

env = UnicycleNavEnv()  # headless by default; call env.render('rgb_array') if needed
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_amr_navigation")
```

## Configuration

All parameters can be configured via YAML files. See `configs/ppo_default.yaml`:

```yaml
env:
  dt: 0.1
  max_steps: 400
  render_mode: null
  scenario: blockage
  scenario_kwargs:
    num_pallets: 1
  reward:
    w_prog: 1.0
    w_lat: 0.2
    w_head: 0.1
    w_clear: 0.4
    w_dv: 0.05
    w_dw: 0.02
    R_goal: 50.0
    R_collide: 50.0
    R_timeout: 10.0
    clearance_safe_m: 0.5
  lidar:
    beams: 24
    fov_deg: 240
    max_range_m: 4.0
    step_m: 0.025
  preview:
    K: 5
    ds: 0.6
    range_m: 3.0
  # Optional curriculum (example)
  curriculum:
    enabled: false
    stages:
      - episode_range: [0, 20000]
        num_pallets_range: [1, 1]
        corridor_width_range: [2.6, 3.0]
      - episode_range: [20000, 60000]
        num_pallets_range: [1, 2]
        corridor_width_range: [2.4, 2.8]
      - episode_range: [60000, 100000]
        num_pallets_range: [2, 3]
        corridor_width_range: [2.2, 2.6]
```

## Architecture

```
me5418_nav/
├── envs/           # Gym environment interface
├── models/         # Robot dynamics (UnicycleModel)
├── navigation/     # Path tracking algorithms
├── sensors/        # LiDAR simulation
├── scenarios/      # Environment generation (blockage, etc.)
├── visualization/  # Pygame renderer
└── config.py       # Configuration system
```

## Scenarios

### Temporary Blockage
- Corridor with 1-3 pallets partially blocking the path
- Robot must navigate through remaining gaps
- Corridor width: 2.2-3.0m, robot diameter: 0.5m

## Performance Metrics

The environment tracks key metrics during evaluation:
- **Success rate**: Episodes reaching the goal
- **Collision rate**: Episodes ending in collision
- **Timeout rate**: Episodes exceeding time limit
- **Mean completion time**: Average time for successful episodes
- **Path following length**: Progress along the reference path

## Research Applications

This environment is designed for studying:
- Continuous control in constrained spaces
- Path tracking vs. obstacle avoidance trade-offs
- Local sensing for navigation (no global mapping)
- Industrial robotics scenarios
- Comparison with classical methods (DWA, Pure Pursuit + APF)
