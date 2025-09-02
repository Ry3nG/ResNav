from __future__ import annotations

import os
import csv
import time
import numpy as np

import argparse
from me5418_nav.envs import UnicycleNavEnv, EnvConfig


def run_random(episodes: int = 3, steps: int = 200, log_dir: str = "logs", render: bool = False):
    os.makedirs(log_dir, exist_ok=True)
    env = UnicycleNavEnv(EnvConfig(), render_mode='human' if render else None)
    for ep in range(episodes):
        obs, info = env.reset()
        prev_action = None
        path = os.path.join(log_dir, f"random_ep{ep:03d}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step","x","y","theta","v","omega","a_v","a_w",
                "min_lidar","reward","terminated","truncated","collision"
            ])
            if render:
                env.render()
            for t in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                s = env.robot._last_state
                min_lidar = float(np.min(obs[:env.cfg.lidar_beams]))
                w.writerow([t, s.x, s.y, s.theta, s.v, s.omega, action[0], action[1], min_lidar, reward, int(terminated), int(truncated), int(info.get('collision', False))])
                if render:
                    env.render()
                if terminated or truncated:
                    break
        print(f"Episode {ep} -> steps: {t+1}, log: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    run_random(episodes=args.episodes, steps=args.steps, log_dir=args.log_dir, render=args.render)
