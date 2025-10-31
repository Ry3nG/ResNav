# Unified AMR Navigation via Residual RL

Residual policy + conventional tracker for AMR local navigation in a 2D factory map. The RL agent outputs a residual action on top of Pure Pursuit.

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
The interactive launcher guides you to train and render models.

### 3) Launch
```bash
make amr
```
The interactive launcher guides you to train and render models. For demo, you can simply use the trained model in the ***runs*** folder.
### 4) Render

Enter the parameters like below
```bash
Select [0-2] (default 0): 1

Examples:
runs/demo_1017/best
Model directory (best/final/ckpt_step_N) (default ''): runs/demo_1017/best
```
This model's **TRAINING CONFIGURATION** will then be displayed.
Then choose the render parameters. A example is shown below.
```bash
Steps (default '300'): 200
Seed (default '42'): 567
Output MP4 filename (saves to run/outputs/) (default 'demo'): try
```
Use different **seed** can show the path of mobile robot in different maps.

Then a mp4 video can be found in model's file. For example, if input same as above, the video will in */runs/demo_1017/outputs*


## Package Organization

## Project Structure
- **Simulation Core (`amr_env/sim`)**: Lightweight 2D world. Generates corridor scenarios with randomized static obstacles (`scenarios.py`), simulates unicycle kinematics (`dynamics.py`), performs collision checking (`collision.py`), and models a 24‑beam LiDAR sensor (`lidar.py`).
- **Gymnasium Environment (`amr_env/gym`)**: Home of `ResidualNavEnv` (standard Gymnasium API). Orchestrates the simulation, builds multi‑modal observations (`ObservationBuilder`), computes rewards (`RewardManager`), and interprets actions as residuals to the baseline controller.
- **Control and Planning (`amr_env/control`, `amr_env/planning`)**: Classical robotics stack. `pure_pursuit.py` implements the baseline path tracker; planning helpers compute path‑relative metrics (e.g., lateral and heading error) for agent observations.
- **Training Framework (`training`)**: Scripts for training/evaluation. `train_sac.py` (entry point) uses `env_factory.py` for vectorized envs. `feature_extractors.py` defines a custom 1D CNN for stacked LiDAR scans. `rollout.py` visualizes and records trained agents.
- **Configuration (`configs/`)**: YAML‑driven configuration. Modify environment dimensions and obstacle density (`env/blockage.yaml`), robot limits (`robot/allow_reverse.yaml`), reward weights (`reward/lower_w_path.yaml`), and network architecture (`network/lidar_cnn.yaml`) without changing source code.

## Development Notes
- **Python 3.10+ required**: The codebase uses modern type hints (`tuple[...]`, `dict[...]`, `list[...]`) via `from __future__ import annotations`.
