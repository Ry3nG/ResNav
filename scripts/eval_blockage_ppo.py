#!/usr/bin/env python3
"""
Evaluate a trained PPO policy on blockage maps.

Supports headless metrics collection and optional on-screen visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from me5418_nav.envs.unicycle_nav_env import UnicycleNavEnv, EnvConfig
from me5418_nav.envs.rl_wrappers import BlockageRLWrapper, RewardConfig
from me5418_nav.maps import BlockageScenarioConfig
from me5418_nav.constants import GRID_RESOLUTION_M, DT_S


def make_env(seed: int, scen_cfg: BlockageScenarioConfig, render: bool):
    def _thunk():
        # 10x10m map at 0.05m -> 200x200 cells
        cfg = EnvConfig(dt=DT_S, map_size=(200, 200), res=GRID_RESOLUTION_M)
        env = UnicycleNavEnv(cfg=cfg, render_mode=("human" if render else None))
        env = BlockageRLWrapper(
            env, scenario_cfg=scen_cfg, reward_cfg=RewardConfig(), seed=seed
        )
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO on blockage maps")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip")
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize pickle (vecnormalize.pkl)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render", action="store_true", help="Enable on-screen rendering"
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="0.0,0.4,0.6,1.0,10.0",
        help="Comma-separated clearance bin edges in meters (min_clearance)",
    )
    # Scenario knobs
    parser.add_argument("--num-pallets", type=str, default="1,3")
    parser.add_argument("--pallet-width", type=str, default="0.5,1.1")
    parser.add_argument("--pallet-length", type=str, default="0.3,0.6")

    args = parser.parse_args()

    # Scenario config
    num_pallets_range = tuple(map(int, args.num_pallets.split(",")))
    pallet_width_range = tuple(map(float, args.pallet_width.split(",")))
    pallet_length_range = tuple(map(float, args.pallet_length.split(",")))
    scen_cfg = BlockageScenarioConfig(
        num_pallets_range=num_pallets_range,
        pallet_width_range=pallet_width_range,
        pallet_length_range=pallet_length_range,
    )

    rng = np.random.default_rng(args.seed)
    env_fns = [make_env(int(rng.integers(0, 2**31 - 1)), scen_cfg, args.render)]
    vec = DummyVecEnv(env_fns)

    # Load VecNormalize stats if provided
    if args.vecnorm:
        vec = VecNormalize.load(args.vecnorm, vec)
        vec.training = False
        vec.norm_reward = False

    model = PPO.load(args.model, device="auto")

    results = {"success": 0, "collision": 0, "timeout": 0}
    # Binning by scenario_min_clearance
    edges = [float(x) for x in args.bins.split(",")]
    if edges[0] > -1e-9:  # ensure starts at 0 for clarity
        edges = [0.0] + edges
    edges = sorted(list(dict.fromkeys(edges)))  # unique + sorted preserving order
    bin_stats = [
        {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lo": edges[i], "hi": edges[i + 1]}
        for i in range(len(edges) - 1)
    ]

    for ep in range(args.episodes):
        obs = vec.reset()
        done = False
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec.step(action)
            ep_steps += 1
            done = bool(dones[0])
            if args.render:
                try:
                    # Access underlying base env
                    wrapper = vec.envs[0]
                    base_env = getattr(wrapper, "env", wrapper)
                    # Compute live clearance and progress from base env state
                    pose = base_env.robot.as_pose()
                    ranges, _ = base_env.lidar.cast(pose, base_env.grid)
                    min_range = float(np.min(ranges)) if ranges.size > 0 else 0.0
                    # Progress percent via nearest waypoint index
                    idx = base_env._nearest_path_index(pose[0], pose[1])
                    total_wp = (
                        base_env.path_waypoints.shape[0]
                        if base_env.path_waypoints is not None
                        else 1
                    )
                    progress_pct = int(round(100 * idx / max(1, total_wp - 1)))
                    base_env._debug_status = {
                        "episode": ep + 1,
                        "step": ep_steps,
                        "reward": (
                            float(reward[0])
                            if isinstance(reward, np.ndarray)
                            else float(reward)
                        ),
                        "clearance": f"{min_range:.2f}m",
                        "action": f"v={action[0][0]:.2f}, ω={action[0][1]:.2f}",
                        "progress": f"{progress_pct}%",
                        "min_range": min_range,
                        "v_cmd": action[0][0],
                        "w_cmd": action[0][1],
                    }
                    base_env.render()
                except Exception:
                    pass
        info = infos[0]
        if info.get("success", False):
            results["success"] += 1
            print(f"Episode {ep+1}: SUCCESS in {ep_steps} steps")
        elif info.get("collision", False):
            results["collision"] += 1
            print(f"Episode {ep+1}: COLLISION at {ep_steps} steps")
        else:
            results["timeout"] += 1
            print(f"Episode {ep+1}: TIMEOUT at {ep_steps} steps")

        # Update bin stats
        mclear = float(info.get("scenario_min_clearance", -1.0))
        for b in bin_stats:
            if b["lo"] <= mclear < b["hi"]:
                b["n"] += 1
                if info.get("success", False):
                    b["success"] += 1
                elif info.get("collision", False):
                    b["collision"] += 1
                else:
                    b["timeout"] += 1
                break

    total = max(1, args.episodes)
    print("\n=== Evaluation Summary ===")
    print(
        f"Success:  {results['success']}/{total} ({results['success']/total*100:.1f}%)"
    )
    print(
        f"Collision:{results['collision']}/{total} ({results['collision']/total*100:.1f}%)"
    )
    print(
        f"Timeout:  {results['timeout']}/{total} ({results['timeout']/total*100:.1f}%)"
    )

    # Per-bin summary
    print("\n=== By Clearance Bin (scenario_min_clearance, meters) ===")
    for b in bin_stats:
        n = max(1, b["n"])  # avoid div zero
        print(
            f"[{b['lo']:.2f}, {b['hi']:.2f}): n={b['n']}, success={b['success']/n:.2f}, "
            f"collision={b['collision']/n:.2f}, timeout={b['timeout']/n:.2f}"
        )


if __name__ == "__main__":
    main()
