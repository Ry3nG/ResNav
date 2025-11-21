"""Baseline comparison evaluation framework for APF vs RL controllers."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add ResNav to path
sys.path.insert(0, '/home/gong-zerui/code/ResNav')

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.control.apf import compute_apf_command
from tools.controller_io import absolute_to_residual


def create_env(seed: int, env_cfg: dict, robot_cfg: dict, reward_cfg: dict, run_cfg: dict) -> gym.Env:
    """Create environment with fixed seed for reproducible evaluation."""
    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=seed)
    return env


def extract_lidar_angles(env: ResidualNavEnv) -> np.ndarray:
    """Extract LiDAR beam angles from environment configuration."""
    n_beams = env.lidar.beams
    fov_rad = env.lidar.fov_rad  # Already in radians
    angles = np.linspace(-fov_rad/2, fov_rad/2, n_beams)
    return angles


def run_apf_episode(env: ResidualNavEnv, seed: int, params: dict) -> dict:
    """Run one episode with APF controller."""
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False

    lidar_angles = extract_lidar_angles(env)

    # Metrics
    steps = 0
    total_distance = 0.0
    min_clearance = float('inf')
    velocities = []
    accelerations = []
    jerks = []
    compute_times = []

    prev_v, prev_omega = 0.0, 0.0
    prev_a, prev_alpha = 0.0, 0.0
    prev_pos = None

    while not (done or truncated):
        # Get current state
        lidar_scan = obs['lidar'] if isinstance(obs, dict) else obs[:24]
        pose = (env._model._last_state.x, env._model._last_state.y, env._model._last_state.theta)
        goal = env._scenario.goal_xy

        # Compute APF command
        t_start = time.perf_counter()
        v, omega = compute_apf_command(
            pose=pose,
            goal=goal,
            lidar_scan=lidar_scan,
            lidar_angles=lidar_angles,
            **params
        )
        compute_time = (time.perf_counter() - t_start) * 1000  # ms
        compute_times.append(compute_time)

        # Step environment using residualized command
        action = absolute_to_residual(env, v, omega)
        obs, reward, done, truncated, info = env.step(action)
        executed_v, executed_omega = env._last_u

        # Collect metrics
        steps += 1
        velocities.append(executed_v)

        # Track distance traveled
        if prev_pos is not None:
            dist = np.sqrt((pose[0] - prev_pos[0])**2 + (pose[1] - prev_pos[1])**2)
            total_distance += dist
        prev_pos = pose

        # Track min clearance
        valid_ranges = lidar_scan[lidar_scan < 3.9]
        if len(valid_ranges) > 0:
            min_clearance = min(min_clearance, np.min(valid_ranges))

        # Track acceleration and jerk
        a = (executed_v - prev_v) / env.dt
        alpha = (executed_omega - prev_omega) / env.dt
        accelerations.append(a)

        jerk = (a - prev_a) / env.dt
        jerks.append(abs(jerk))

        prev_v, prev_omega = executed_v, executed_omega
        prev_a, prev_alpha = a, alpha

        if steps >= env.run_cfg['max_steps']:
            break

    # Compute final metrics
    final_dist_to_goal = np.sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)
    success = final_dist_to_goal < 0.5 and not info.get('collision', False)

    return {
        'success': success,
        'collision': info.get('collision', False),
        'timeout': steps >= env.run_cfg['max_steps'] and not success,
        'episode_length': steps,
        'time_to_goal': steps * env.dt,
        'final_distance_to_goal': final_dist_to_goal,
        'path_length': total_distance,
        'min_clearance': min_clearance if min_clearance != float('inf') else 0.0,
        'avg_velocity': np.mean(velocities) if velocities else 0.0,
        'max_velocity': np.max(velocities) if velocities else 0.0,
        'avg_acceleration': np.mean(np.abs(accelerations)) if accelerations else 0.0,
        'max_acceleration': np.max(np.abs(accelerations)) if accelerations else 0.0,
        'avg_jerk': np.mean(jerks) if jerks else 0.0,
        'max_jerk': np.max(jerks) if jerks else 0.0,
        'avg_compute_time_ms': np.mean(compute_times) if compute_times else 0.0,
        'max_compute_time_ms': np.max(compute_times) if compute_times else 0.0,
    }


def run_rl_episode(env: ResidualNavEnv, model: SAC, seed: int) -> dict:
    """Run one episode with trained RL model."""
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False

    # Metrics
    steps = 0
    total_distance = 0.0
    min_clearance = float('inf')
    velocities = []
    accelerations = []
    jerks = []
    compute_times = []

    prev_v, prev_omega = 0.0, 0.0
    prev_a = 0.0
    prev_pos = None

    while not (done or truncated):
        pose = (env._model._last_state.x, env._model._last_state.y, env._model._last_state.theta)
        goal = env._scenario.goal_xy

        # Get RL action
        t_start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        compute_time = (time.perf_counter() - t_start) * 1000  # ms
        compute_times.append(compute_time)

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Get actual velocity from action (residual + baseline)
        v = env._model._last_state.v
        omega = env._model._last_state.omega

        # Collect metrics
        steps += 1
        velocities.append(v)

        # Track distance traveled
        if prev_pos is not None:
            dist = np.sqrt((pose[0] - prev_pos[0])**2 + (pose[1] - prev_pos[1])**2)
            total_distance += dist
        prev_pos = pose

        # Track min clearance
        lidar_scan = obs['lidar'] if isinstance(obs, dict) else obs[:24]
        valid_ranges = lidar_scan[lidar_scan < 3.9]
        if len(valid_ranges) > 0:
            min_clearance = min(min_clearance, np.min(valid_ranges))

        # Track acceleration and jerk
        a = (v - prev_v) / env.dt
        accelerations.append(a)

        jerk = (a - prev_a) / env.dt
        jerks.append(abs(jerk))

        prev_v, prev_omega = v, omega
        prev_a = a

        if steps >= env.run_cfg['max_steps']:
            break

    # Compute final metrics
    final_dist_to_goal = np.sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)
    success = final_dist_to_goal < 0.5 and not info.get('collision', False)

    return {
        'success': success,
        'collision': info.get('collision', False),
        'timeout': steps >= env.run_cfg['max_steps'] and not success,
        'episode_length': steps,
        'time_to_goal': steps * env.dt,
        'final_distance_to_goal': final_dist_to_goal,
        'path_length': total_distance,
        'min_clearance': min_clearance if min_clearance != float('inf') else 0.0,
        'avg_velocity': np.mean(velocities) if velocities else 0.0,
        'max_velocity': np.max(velocities) if velocities else 0.0,
        'avg_acceleration': np.mean(np.abs(accelerations)) if accelerations else 0.0,
        'max_acceleration': np.max(np.abs(accelerations)) if accelerations else 0.0,
        'avg_jerk': np.mean(jerks) if jerks else 0.0,
        'max_jerk': np.max(jerks) if jerks else 0.0,
        'avg_compute_time_ms': np.mean(compute_times) if compute_times else 0.0,
        'max_compute_time_ms': np.max(compute_times) if compute_times else 0.0,
    }


if __name__ == '__main__':
    print("Baseline comparison evaluation framework loaded.")
    print("Use compare_all_controllers() to run full comparison.")
