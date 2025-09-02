# ME5418 Project: Learning When to Yield and When to Pass for Unified AMR Navigation

## Overview

This project focuses on training a reinforcement learning policy for a single differential-drive Autonomous Mobile Robot (AMR) to navigate 2D factory-style environments with dynamic bottlenecks. The core challenge is to learn **when to yield and wait for a clear path, and when to confidently pass** using only local sensing, without online global replanning.

## Problem Statement

Classical navigation stacks frequently exhibit undesirable behaviors like oscillation, freezing, or deadlock in industrial scenarios such as:
- **Temporary aisle blockages**
- **Occluded intersections**
- **Narrow counter-flows with non-cooperative agents**

This project proposes a unified continuous control policy that integrates path tracking with intelligent, temporal decision-making to navigate these common industrial "pain points" more efficiently than classical methods.

## Key Constraints & Objectives

- **Fixed Global Path**: A* planned path computed once at task start - no online replanning
- **Local Sensing**: Sparse 2D LiDAR only (24-36 beams, 270° FOV, 4m range)
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

## Quick Start

### Run Baseline Evaluation
```bash
python scripts/blockage_demo.py
```


## RL Formulation

- **State Space**: LiDAR readings + robot kinematics + path context
- **Action Space**: Continuous velocity commands [v, ω]
  - v ∈ [0.0, 1.5] m/s (linear velocity)
  - ω ∈ [-2.0, 2.0] rad/s (angular velocity)
- **Algorithm**: Proximal Policy Optimization (PPO) with tanh-squashed Gaussian policy
- **Rewards**: Path progress + collision avoidance + smoothness + goal reaching

## Baselines

- Pure Pursuit + Artificial Potential Fields (PP+APF)
- Dynamic Window Approach (planned)

## Evaluation Metrics

- Success Rate
- Collision Rate
- Deadlock Rate
- Task Completion Time
- Path Following Accuracy

