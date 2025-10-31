import numpy as np

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.sim.movers import DiscMover


def test_env_detects_dynamic_collision(monkeypatch):
    env_cfg = {
        "name": "omcf",
        "map": {
            "size_m": [10.0, 10.0],
            "resolution_m": 0.1,
            "corridor_width_m": [3.5, 3.5],
            "wall_thickness_m": 0.3,
            "start_x_m": 1.0,
            "goal_margin_x_m": 1.0,
            "waypoint_step_m": 0.3,
            "pallet_width_m": 1.0,
            "pallet_length_m": 1.0,
            "min_passage_m": 1.3,
            "num_pallets_min": 0,
            "num_pallets_max": 0,
            "holes": {
                "enabled": True,
                "count_pairs": 1,
                "x_range_m": [7.0, 7.0],
                "opening_len_m": 1.5,
            },
        },
        "wrappers": {
            "frame_stack": {"enabled": True, "keys": ["lidar"], "k": 1, "flatten": True}
        },
        "lidar": {
            "beams": 8,
            "fov_deg": 120,
            "max_range_m": 5.0,
            "noise_std_m": 0.0,
            "noise_enable": False,
        },
        "dynamic_movers": {
            "enabled": True,
            "radius_m": 0.45,
            "velocity_range_mps": [0.0, 0.0],
            "acceleration_range_mps2": [0.0, 0.0],
            "lifetime_range_s": [10.0, 10.0],
            "lateral": {
                "spawn_rate_hz": 0.0,
                "safe_spawn_distance_m": 0.0,
                "lanes": ["center"],
            },
            "longitudinal": {"spawn_rate_hz": 0.0, "spawn_from_holes": False},
            "kill_margin_m": 0.5,
            "max_concurrent": 5,
            "reflect_walls": False,
        },
        "viz": {"show_inflated": False, "show_lidar": False, "show_actions": False, "fps": 20},
    }
    robot_cfg = {
        "v_max": 1.0,
        "w_max": 2.0,
        "v_min": 0.0,
        "radius_m": 0.45,
        "controller": {"lookahead_m": 1.0, "speed_nominal": 0.5},
    }
    reward_cfg = {
        "sparse": {"goal": 0.0, "collision": 0.0, "timeout": 0.0},
        "weights": {"progress": 0.0, "path": 0.0, "effort": 0.0, "sparse": 0.0},
        "path_penalty": {"lateral_weight": 0.0, "heading_weight": 0.0},
        "effort_penalty": {"lambda_v": 0.0, "lambda_w": 0.0, "lambda_jerk": 0.0},
    }
    run_cfg = {"dt": 0.1, "max_steps": 5, "seed": 123}

    # Manually inject a mover at agent start position
    original_init_spawner = ResidualNavEnv._init_spawner

    def _patched_init_spawner(self):
        original_init_spawner(self)
        # Add blocking mover
        start_x = float(env_cfg["map"]["start_x_m"])
        cy = 5.0  # Center of 10m map
        self._active_movers.append(
            DiscMover(x=start_x, y=cy, vx=0.0, vy=0.0, radius_m=0.45, spawn_t=0.0, lifetime_s=10.0)
        )

    monkeypatch.setattr(ResidualNavEnv, "_init_spawner", _patched_init_spawner)

    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert bool(terminated), "Should detect collision with static mover at start"


def test_continuous_spawning_statistics():
    """Statistical test: verify continuous spawning prevents permanent empty periods."""
    env_cfg = {
        "name": "omcf",
        "map": {
            "size_m": [20.0, 20.0],
            "resolution_m": 0.05,
            "corridor_width_m": [3.5, 4.0],
            "wall_thickness_m": 0.30,
            "start_x_m": 1.0,
            "goal_margin_x_m": 1.0,
            "waypoint_step_m": 0.30,
            "pallet_width_m": 1.1,
            "pallet_length_m": 2.0,
            "min_passage_m": 1.3,
            "num_pallets_min": 1,
            "num_pallets_max": 2,
            "holes": {
                "enabled": True,
                "count_pairs": 2,
                "x_range_m": [5.0, 15.0],
                "opening_len_m": 1.6,
                "min_spacing_m": 2.0,
            },
        },
        "wrappers": {
            "frame_stack": {"enabled": True, "keys": ["lidar"], "k": 4, "flatten": True}
        },
        "lidar": {
            "beams": 24,
            "fov_deg": 240,
            "max_range_m": 4.0,
            "noise_std_m": 0.03,
            "noise_enable": False,  # Disable noise for determinism
        },
        "dynamic_movers": {
            "enabled": True,
            "radius_m": 0.45,
            "velocity_range_mps": [0.4, 1.5],
            "acceleration_range_mps2": [-0.15, 0.15],
            "lifetime_range_s": [6.0, 10.0],
            "lateral": {
                "spawn_rate_hz": 0.20,  # Higher rate for faster testing
                "spawn_anywhere": True,
                "safe_spawn_distance_m": 3.0,
                "lanes": ["center", "up", "down"],
            },
            "longitudinal": {
                "spawn_rate_hz": 0.15,
                "spawn_from_holes": True,
                "safe_spawn_distance_m": 3.0,
            },
            "lane_offset_m": 0.8,
            "kill_margin_m": 0.5,
            "max_concurrent": 30,
            "reflect_walls": True,
        },
        "viz": {"show_inflated": False, "show_lidar": False, "show_actions": False, "fps": 20},
    }
    robot_cfg = {
        "v_max": 1.0,
        "w_max": 2.0,
        "v_min": 0.0,
        "radius_m": 0.45,
        "controller": {"lookahead_m": 1.0, "speed_nominal": 0.5},
    }
    reward_cfg = {
        "sparse": {"goal": 10.0, "collision": -5.0, "timeout": -1.0},
        "weights": {"progress": 1.0, "path": 0.5, "effort": 0.1, "sparse": 1.0},
        "path_penalty": {"lateral_weight": 0.5, "heading_weight": 0.3},
        "effort_penalty": {"lambda_v": 0.1, "lambda_w": 0.05, "lambda_jerk": 0.02},
    }
    run_cfg = {"dt": 0.1, "max_steps": 300, "seed": 42}

    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=42)

    mover_counts = []
    for step in range(200):  # Run for 20s @ 0.1s/step
        env.step(np.zeros(2, dtype=np.float32))
        mover_counts.append(len(env._active_movers))

    # Statistical assertions
    # 1. Movers exist for a significant portion of the episode (not 100%, due to Poisson gaps)
    #    Note: Safe spawning checks + static obstacles reduce success rate
    has_movers_ratio = sum(c > 0 for c in mover_counts) / len(mover_counts)
    assert has_movers_ratio >= 0.35, (
        f"Movers should exist for >=35% of steps (got {has_movers_ratio:.2%}). "
        "This prevents 'wait-and-pass' exploit compared to finite-count (0% after ~15s)."
    )

    # 2. Average concurrent movers should match expected value within tolerance
    # Expected: (λ_lat + λ_lon) * E[TTL] = (0.20 + 0.15) * 8.0 = 2.8
    # Note: Safe spawning significantly reduces actual spawn success rate
    expected_concurrent = (0.20 + 0.15) * 8.0
    actual_mean = np.mean(mover_counts)
    assert 0.2 * expected_concurrent <= actual_mean <= 2.0 * expected_concurrent, (
        f"Mean concurrent movers {actual_mean:.2f} outside expected range "
        f"[{0.2*expected_concurrent:.2f}, {2.0*expected_concurrent:.2f}]"
    )

    # 3. Max concurrent should respect configured limit
    max_concurrent = max(mover_counts)
    assert max_concurrent <= 30, f"Max concurrent {max_concurrent} exceeds limit 30"

    print(f"✓ Continuous spawning stats: {has_movers_ratio:.1%} coverage, "
          f"mean={actual_mean:.2f}, max={max_concurrent}")
