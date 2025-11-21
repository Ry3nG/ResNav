#!/usr/bin/env python
"""Test all overnight training stages to identify where training broke."""
import numpy as np
from pathlib import Path

from training.rollout import load_config_dict, resolve_model_and_vecnorm, load_resolved_run_config
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.gym.wrappers import LidarFrameStackVec

def test_model(model_name, model_path, test_env_cfg_name, n_episodes=5):
    """Test a model on a specific environment config."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} on {test_env_cfg_name}")
    print(f"{'='*60}")

    try:
        model_zip, vecnorm_pkl, run_dir = resolve_model_and_vecnorm(model_path)
    except Exception as e:
        print(f"[SKIP] Could not load model: {e}")
        return None

    train_cfg = load_resolved_run_config(run_dir)
    robot_cfg = train_cfg["robot"]
    reward_cfg = train_cfg["reward"]
    run_cfg = train_cfg.get("run", {"dt": 0.1, "max_steps": 600})

    test_env_cfg = load_config_dict(f'configs/env/{test_env_cfg_name}.yaml')

    def make_env():
        return ResidualNavEnv(test_env_cfg, robot_cfg, reward_cfg, run_cfg)

    venv = DummyVecEnv([make_env])
    k = int(test_env_cfg["wrappers"]["frame_stack"]["k"])
    if k > 1:
        venv = LidarFrameStackVec(venv, k=k)

    if vecnorm_pkl and Path(vecnorm_pkl).exists():
        venv = VecNormalize.load(vecnorm_pkl, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("[WARN] No VecNormalize found")

    model = SAC.load(model_zip, env=venv)

    successes = 0
    total_steps = []

    for ep in range(n_episodes):
        obs = venv.reset()
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            if done[0]:
                success = info[0].get('is_success', False)
                if success:
                    successes += 1
                total_steps.append(step + 1)
                break
        else:
            total_steps.append(300)

    success_rate = successes / n_episodes * 100
    avg_steps = np.mean(total_steps)
    print(f"Results: {successes}/{n_episodes} success ({success_rate:.0f}%)")
    print(f"Avg steps: {avg_steps:.0f}")

    return success_rate

print("="*60)
print("OVERNIGHT TRAINING STAGE ANALYSIS")
print("="*60)

# Test each stage on appropriate difficulty
test_results = []

# Stage 1: overnight_basic - should work on basic
r1 = test_model("overnight_basic", "runs/overnight_basic/best", "eval_basic", 5)
test_results.append(("Stage 1 (basic) on basic", r1))

# Stage 2: overnight_medium - should work on basic and medium
r2a = test_model("overnight_medium", "runs/overnight_medium/best", "eval_basic", 5)
test_results.append(("Stage 2 (medium) on basic", r2a))
r2b = test_model("overnight_medium", "runs/overnight_medium/best", "eval_medium", 5)
test_results.append(("Stage 2 (medium) on medium", r2b))

# Stage 3: overnight_hard - should work on all
r3a = test_model("overnight_hard", "runs/overnight_hard/best", "eval_basic", 5)
test_results.append(("Stage 3 (hard) on basic", r3a))
r3b = test_model("overnight_hard", "runs/overnight_hard/best", "eval_medium", 5)
test_results.append(("Stage 3 (hard) on medium", r3b))
r3c = test_model("overnight_hard", "runs/overnight_hard/best", "eval_hard", 5)
test_results.append(("Stage 3 (hard) on hard", r3c))

# Compare with demo_1031 (known working)
r_demo_basic = test_model("demo_1031", "runs/demo_1031/best", "eval_basic", 5)
test_results.append(("demo_1031 on basic", r_demo_basic))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, rate in test_results:
    if rate is not None:
        status = "✓" if rate > 50 else "✗"
        print(f"{status} {name}: {rate:.0f}%")
    else:
        print(f"? {name}: SKIPPED")
