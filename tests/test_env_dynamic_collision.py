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
                "occluders": {"enabled": False},
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
            "speed_mps": [0.0, 0.0],
            "lane_offset_m": 0.1,
            "spawn_jitter_m": 0.0,
            "from_right": {"count_min": 0, "count_max": 0, "lanes": ["center"]},
            "from_holes": {"enabled": False, "count_min": 0, "count_max": 0},
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

    def _stub_sample_movers(env_cfg_inner, scenario_info, rng):
        start_x = float(env_cfg_inner["map"]["start_x_m"])
        cy = 0.5 * (float(scenario_info["y_top"]) + float(scenario_info["y_bot"]))
        return [DiscMover(x=start_x, y=cy, vx=0.0, vy=0.0, radius_m=0.45)]

    monkeypatch.setattr(
        "amr_env.sim.movers.sample_movers_for_omcf",
        _stub_sample_movers,
    )

    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert bool(terminated)
