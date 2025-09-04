# ME5418 Project: Learning When to Yield and When to Pass for Unified AMR Navigation

## Overview
### Grid usage convention

We distinguish two grids throughout the codebase:

- Sensing grid (raw occupancy): the world obstacle geometry. Used for LiDAR
  simulation and rendering. It matches what sensors "see".
- Collision grid (C-space): the sensing grid inflated by the robot radius and
  safety margin. Used by the environment for feasibility and collision checks
  (reset feasibility and step() collision). Controllers may optionally consult
  C-space for trajectory feasibility; the provided DWA is sensor-driven and
  uses only the sensing grid for LiDAR and approximate clearance.

Controllers should:
- Use the sensing grid when calling LiDAR.cast for realistic ranges
- Use the collision grid when performing any trajectory collision checks

This separation avoids "looks free" vs "collides in env" inconsistencies and
matches common robotics practice.

This project focuses on training a reinforcement learning policy for a single differential-drive Autonomous Mobile Robot (AMR) to navigate 2D factory-style environments with dynamic bottlenecks. The core challenge is to learn **when to yield and wait for a clear path, and when to confidently pass** using only local sensing, without online global replanning.

## Problem Statement

Classical navigation stacks frequently exhibit undesirable behaviors like oscillation, freezing, or deadlock in industrial scenarios such as:
- **Temporary aisle blockages**
- **Occluded intersections**
- **Narrow counter-flows with non-cooperative agents**

This project proposes a unified continuous control policy that integrates path tracking with intelligent, temporal decision-making to navigate these common industrial "pain points" more efficiently than classical methods.

## Key Constraints & Objectives

- **Fixed Global Path**: A* planned path computed once at task start - no online replanning
- **Local Sensing**: Sparse 2D LiDAR only (24–36 beams, 240° FOV, 4 m range)
- **Non-cooperative Environment**: Static obstacles + dynamic movers that don't yield
- **Continuous Control**: Direct velocity commands (v, ω) output

## Target Scenarios

The policy is specifically designed to handle three challenging micro-benchmarks:

1. **Temporary Blockage**: Navigate through narrow gaps around partial obstructions
2. **Occluded Merge**: Safely cross intersections with limited visibility
3. **Narrow Counterflow**: Pass oncoming agents in constrained aisles


## Installation

### Using Conda
```bash
conda env create -f environment.yml
conda activate me5418-nav
pip install -e .
```

### Weights & Biases Setup (Optional)
For experiment tracking and visualization:
```bash
conda install -n me5418-nav wandb
wandb login
```

## Quick Start

### Run Baseline Evaluation
```bash
python scripts/blockage_demo.py --controller dwa
python scripts/blockage_demo.py --controller ppapf
```

### Training & Evaluation (PPO on Blockage)

Train a PPO agent on randomized blockage maps. The training script ships with safety‑oriented reward shaping and sensible defaults: SDE on, observation/reward normalization on, lower clip range, light entropy regularization, periodic evaluation and checkpoints, and W&B in offline mode by default.

Recommended large run (8 envs, 3M steps):

```bash
python scripts/train_blockage_ppo.py \
  --timesteps 3000000 \
  --num-envs 8 \
  --run-name risk10_margin12_sde_clip01_seed0 \
  --seed 0
```

Notes:
- Defaults: `--use-sde` on, `--clip-range 0.1`, `--ent-coef 0.01`, `--norm-obs/--norm-reward` on, `--wandb-mode offline`.
- The script saves best model, periodic checkpoints, TB logs, and `config.yaml` under `logs/ppo_blockage/<run_name>/`.
- Success/collision/timeout rates are logged during training (TensorBoard/W&B).

Evaluate a trained model (headless metrics):

```bash
python scripts/eval_blockage_ppo.py \
  --model logs/ppo_blockage/<run_name>/ppo_blockage.zip \
  --vecnorm logs/ppo_blockage/<run_name>/vecnormalize.pkl \
  --episodes 200 \
  --bins "0.0,0.4,0.6,1.0,10.0"
```

Evaluate visually (on‑screen rendering):

```bash
python scripts/eval_blockage_ppo.py \
  --model logs/ppo_blockage/<run_name>/ppo_blockage.zip \
  --vecnorm logs/ppo_blockage/<run_name>/vecnormalize.pkl \
  --episodes 5 --render
```

Resume training from a checkpoint:

```bash
python scripts/train_blockage_ppo.py \
  --resume-from logs/ppo_blockage/<run_name>/checkpoints/ppo_blockage_500000_steps.zip \
  --resume-vecnorm logs/ppo_blockage/<run_name>/vecnormalize.pkl \
  --run-name <run_name> \
  --timesteps 5000000 --num-envs 8
```

Notes:
- The environment regenerates a new blockage scenario each episode.
- Two-grid convention applies: sensing grid for LiDAR; C-space grid for collisions.
- Rewards (simplified) emphasize goal‑distance progress, clearance‑based risk penalty, smoothness, and a gentle no‑progress penalty; terminal collision penalty is stronger to disincentivize “fast crash”.


## RL Formulation

- **State Space**: LiDAR readings + robot kinematics + path context
- **Action Space**: Continuous velocity commands [v, ω]
  - v ∈ [0.0, 1.5] m/s (linear velocity)
  - ω ∈ [-2.0, 2.0] rad/s (angular velocity)
- **Algorithm**: Proximal Policy Optimization (PPO) with tanh-squashed Gaussian policy
- **Rewards**: Path progress + collision avoidance + smoothness + goal reaching

## Baselines

- Pure Pursuit + Artificial Potential Fields (PP+APF)
- Dynamic Window Approach (DWA)

## Evaluation Metrics

- Success Rate
- Collision Rate
- Deadlock Rate
- Task Completion Time
- Path Following Accuracy

## Training Script Options (Summary)
- Run & device: `--timesteps`, `--seed`, `--device {auto,cpu,cuda}`, `--torch-deterministic`, `--run-name`
- Envs & wrappers: `--num-envs`, `--norm-obs`, `--norm-reward`, `--clip-obs`
- PPO core: `--n-steps` (total rollout across envs), `--batch-size`, `--gamma`, `--gae-lambda`, `--n-epochs`, `--learning-rate`, `--clip-range`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`, `--target-kl`, `--use-sde`, `--sde-sample-freq`, `--policy-arch`
- Schedules: `--lr-schedule {constant,linear}`, `--clip-schedule {constant,linear}`
- Eval/checkpoints: `--eval-freq`, `--eval-episodes`, `--checkpoint-freq`, `--early-stop-patience`
- W&B: `--no-wandb`, `--wandb-project`, `--wandb-group`, `--wandb-mode` (e.g., `offline`)
- Resume: `--resume-from`, `--resume-vecnorm`

See `python scripts/train_blockage_ppo.py --help` for full details.

## Logs & Artifacts
- Base dir: `logs/ppo_blockage/<run_name>/`
- Files/dirs:
  - `ppo_blockage.zip`: final policy; `best_model/` contains best eval policy
  - `vecnormalize.pkl`: VecNormalize stats when normalization is enabled
  - `checkpoints/`: periodic checkpoints by step count
  - `monitor/`: per‑env CSVs of episodic returns/lengths and outcome flags
  - `eval/`: evaluation logs and summaries
  - `tb/`: TensorBoard logs; W&B runs if enabled
  - `config.yaml`: saved CLI config for reproducibility
