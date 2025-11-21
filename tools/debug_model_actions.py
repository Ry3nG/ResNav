#!/usr/bin/env python
"""Deep debug: Check if model is actually producing valid actions."""
import numpy as np
from pathlib import Path

from training.rollout import load_config_dict, resolve_model_and_vecnorm, load_resolved_run_config
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec

model_path = '/home/gong-zerui/code/ResNav/runs/overnight_hard/best'

# Resolve model
model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(model_path)
train_cfg = load_resolved_run_config(run_dir)
train_env_cfg = train_cfg["env"]
robot_cfg = train_cfg["robot"]
reward_cfg = train_cfg["reward"]
run_cfg = train_cfg.get("run", {"dt": 0.1, "max_steps": 600})

def make_train_env():
    return ResidualNavEnv(train_env_cfg, robot_cfg, reward_cfg, run_cfg)

venv = DummyVecEnv([make_train_env])
k = int(train_env_cfg["wrappers"]["frame_stack"]["k"])
if k > 1:
    venv = LidarFrameStackVec(venv, k=k)

venv = VecNormalize.load(vecnorm_pkl, venv)
venv.training = False
venv.norm_reward = False

model = SAC.load(model_zip, env=venv)

print("="*60)
print("Testing STOCHASTIC policy (like training)")
print("="*60)

obs = venv.reset()
for i in range(50):
    action, _ = model.predict(obs, deterministic=False)  # Stochastic!
    obs, reward, done, info = venv.step(action)

    # Get raw action values
    base_env = venv.venv.venv.envs[0] if hasattr(venv.venv, 'venv') else venv.venv.envs[0]
    v_cmd, w_cmd = base_env._last_u

    if i < 10 or done[0]:
        print(f"Step {i+1}: action=[{action[0][0]:.3f}, {action[0][1]:.3f}], "
              f"cmd=[v={v_cmd:.2f}, w={w_cmd:.2f}], reward={reward[0]:.2f}")

    if done[0]:
        print(f"\nEpisode ended at step {i+1}")
        print(f"Success: {info[0].get('is_success', False)}")
        print(f"Reward terms: {info[0].get('reward_terms', {})}")
        break

print("\n" + "="*60)
print("Testing DETERMINISTIC policy")
print("="*60)

obs = venv.reset()
for i in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)

    base_env = venv.venv.venv.envs[0] if hasattr(venv.venv, 'venv') else venv.venv.envs[0]
    v_cmd, w_cmd = base_env._last_u

    if i < 10 or done[0]:
        print(f"Step {i+1}: action=[{action[0][0]:.3f}, {action[0][1]:.3f}], "
              f"cmd=[v={v_cmd:.2f}, w={w_cmd:.2f}], reward={reward[0]:.2f}")

    if done[0]:
        print(f"\nEpisode ended at step {i+1}")
        print(f"Success: {info[0].get('is_success', False)}")
        break

print("\n" + "="*60)
print("Check model internals")
print("="*60)

# Check actor network output
print(f"Model actor log_std: {model.actor.log_std if hasattr(model.actor, 'log_std') else 'N/A'}")
print(f"Model action_scale: {model.actor.action_scale if hasattr(model.actor, 'action_scale') else 'N/A'}")

# Get some random observations and check action distribution
obs = venv.reset()
import torch
with torch.no_grad():
    obs_tensor = {k: torch.as_tensor(v).to(model.device) for k, v in obs.items()}
    mean_actions, log_std, kwargs = model.actor.get_action_dist_params(obs_tensor)
    print(f"\nAction mean: {mean_actions.cpu().numpy()}")
    print(f"Action log_std: {log_std.cpu().numpy()}")
