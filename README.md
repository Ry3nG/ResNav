# Unified AMR Navigation via Residual RL

Residual policy + conventional tracker for AMR local navigation in a 2D factory map. The RL agent outputs a residual action on top of Pure Pursuit.

## Assignment 3 Demo

### 1) Launch the standard training run
Now ships a one-file launcher.

```bash
python train_demo.py
```

By default this executes:

```
python training/train_sac.py env=omcf robot=allow_reverse reward=lower_w_path \
    algo=sac network=lidar_cnn wandb=default \
    run.vec_envs=20 run.total_timesteps=200000 run.seed=20030413
```

Add overrides such as `--override run.vec_envs=10` when needed.

### 2) Run inference on a saved model
For playback or grading, point `inference_demo.py` at any run directory (e.g. `runs/demo_1031/best`, `runs/demo_1107/final`):

```bash
python inference_demo.py --path /home/gong-zerui/code/ResNav/runs/demo_1031/best --render --deterministic
```

Recording defaults to `demo.mp4` in the project root; pass `--record none` to disable or set a custom path.


## Quickstart

### 1) Setup
```bash
make setup
```
This creates/updates the conda env `amr-nav` from `environment.yml` and runs `pip install -e .` inside it.

### 2) Launch
```bash
make amr
```
The interactive launcher guides you to train and render models. For demo, you can simply use the trained model in the ***runs*** folder.
### 3) Render

Enter the parameters like below
```bash
Select [0-2] (default 0): 1

Examples:
runs/demo_1017/best
runs/demo_1031/best
Model directory (best/final/ckpt_step_N) (default ''): runs/demo_1031/best
```
This model's **TRAINING CONFIGURATION** will then be displayed.
Then choose the render parameters. A example is shown below.
```bash
Steps (default '300'): 200
Seed (default '42'): 567
Output MP4 filename (saves to run/outputs/) (default 'demo'): try
```
Use different **seed** can show the path of mobile robot in different maps.

Then a mp4 video can be found in model's file. For example, if input same as above, the video will in */runs/demo_1031/outputs*


## Package Organization

## Project Structure
- **Simulation Core (`amr_env/sim`)**: Lightweight 2D world. Generates corridor scenarios with randomized static obstacles (`scenarios.py`), simulates unicycle kinematics (`dynamics.py`), performs collision checking (`collision.py`), and models a 24‑beam LiDAR sensor (`lidar.py`).
- **Gymnasium Environment (`amr_env/gym`)**: Home of `ResidualNavEnv` (standard Gymnasium API). Orchestrates the simulation, builds multi‑modal observations (`ObservationBuilder`), computes rewards (`RewardManager`), and interprets actions as residuals to the baseline controller.
- **Control and Planning (`amr_env/control`, `amr_env/planning`)**: Classical robotics stack. `pure_pursuit.py` implements the baseline path tracker; planning helpers compute path‑relative metrics (e.g., lateral and heading error) for agent observations.
- **Training Framework (`training`)**: Scripts for training/evaluation. `train_sac.py` (entry point) uses `env_factory.py` for vectorized envs. `feature_extractors.py` defines a custom 1D CNN for stacked LiDAR scans. `rollout.py` visualizes and records trained agents.
- **Configuration (`configs/`)**: YAML‑driven configuration. Modify environment dimensions and obstacle density (`env/blockage.yaml`), robot limits (`robot/allow_reverse.yaml`), reward weights (`reward/lower_w_path.yaml`), and network architecture (`network/lidar_cnn.yaml`) without changing source code.

## Development Notes
- **Python 3.10+ required**: The codebase uses modern type hints (`tuple[...]`, `dict[...]`, `list[...]`) via `from __future__ import annotations`.
