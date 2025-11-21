#!/usr/bin/env python
"""Debug train-test mismatch: compare observations between training and testing."""
import numpy as np
from pathlib import Path

from training.rollout import load_config_dict, resolve_model_and_vecnorm, load_resolved_run_config
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec

model_path = '/home/gong-zerui/code/ResNav/runs/overnight_hard/best'

print("="*60)
print("DEBUG: Train-Test Mismatch Investigation")
print("="*60)

# Resolve model
model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(model_path)
print(f"Model: {model_zip}")
print(f"VecNorm: {vecnorm_pkl}")

# Load training config (what model was trained on)
train_cfg = load_resolved_run_config(run_dir)
train_env_cfg = train_cfg["env"]
robot_cfg = train_cfg["robot"]
reward_cfg = train_cfg["reward"]
run_cfg = train_cfg.get("run", {"dt": 0.1, "max_steps": 600})

print(f"\n=== Training Environment Config ===")
print(f"Scenario: {train_env_cfg.get('name')}")
print(f"Dynamic movers enabled: {train_env_cfg.get('dynamic_movers', {}).get('enabled')}")
print(f"LiDAR beams: {train_env_cfg.get('lidar', {}).get('beams')}")

# Load test config
test_env_cfg = load_config_dict('configs/env/eval_basic.yaml')
print(f"\n=== Test Environment Config ===")
print(f"Scenario: {test_env_cfg.get('name')}")
print(f"Dynamic movers enabled: {test_env_cfg.get('dynamic_movers', {}).get('enabled')}")
print(f"LiDAR beams: {test_env_cfg.get('lidar', {}).get('beams')}")

# Create environment with TRAINING config
def make_train_env():
    return ResidualNavEnv(train_env_cfg, robot_cfg, reward_cfg, run_cfg)

def make_test_env():
    return ResidualNavEnv(test_env_cfg, robot_cfg, reward_cfg, run_cfg)

# Test with training config
print("\n=== Testing with TRAINING config (should work) ===")
venv = DummyVecEnv([make_train_env])
k = int(train_env_cfg["wrappers"]["frame_stack"]["k"])
if k > 1:
    venv = LidarFrameStackVec(venv, k=k)

venv = VecNormalize.load(vecnorm_pkl, venv)
venv.training = False
venv.norm_reward = False

model = SAC.load(model_zip, env=venv)

# Run episode with training config
obs = venv.reset()
for i in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    if done[0]:
        print(f"Episode ended at step {i+1}")
        print(f"Success: {info[0].get('is_success', False)}")
        break
else:
    print("Completed 300 steps")

# Test with test config
print("\n=== Testing with TEST config (eval_basic) ===")
venv2 = DummyVecEnv([make_test_env])
k2 = int(test_env_cfg["wrappers"]["frame_stack"]["k"])
if k2 > 1:
    venv2 = LidarFrameStackVec(venv2, k=k2)

venv2 = VecNormalize.load(vecnorm_pkl, venv2)
venv2.training = False
venv2.norm_reward = False

model2 = SAC.load(model_zip, env=venv2)

obs2 = venv2.reset()
for i in range(300):
    action, _ = model2.predict(obs2, deterministic=True)
    obs2, reward, done, info = venv2.step(action)
    if done[0]:
        print(f"Episode ended at step {i+1}")
        print(f"Success: {info[0].get('is_success', False)}")
        break
else:
    print("Completed 300 steps")

# Compare observation distributions
print("\n=== Observation Analysis ===")
print("Training VecNormalize obs_rms:")
for key, rms in venv.obs_rms.items():
    print(f"  {key}: mean_range=[{rms.mean.min():.3f}, {rms.mean.max():.3f}], var_range=[{rms.var.min():.3f}, {rms.var.max():.3f}]")

# Check raw observations before normalization
print("\nRaw observations comparison:")
train_base = venv.venv.venv if hasattr(venv.venv, 'venv') else venv.venv
test_base = venv2.venv.venv if hasattr(venv2.venv, 'venv') else venv2.venv

train_raw_obs = train_base.envs[0]._get_obs()
test_raw_obs = test_base.envs[0]._get_obs()

for key in train_raw_obs:
    train_arr = train_raw_obs[key]
    test_arr = test_raw_obs[key]
    print(f"  {key}:")
    print(f"    Train: range=[{train_arr.min():.3f}, {train_arr.max():.3f}], mean={train_arr.mean():.3f}")
    print(f"    Test:  range=[{test_arr.min():.3f}, {test_arr.max():.3f}], mean={test_arr.mean():.3f}")
