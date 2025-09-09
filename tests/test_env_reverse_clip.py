import numpy as np

from amr_env.gym.residual_nav_env import ResidualNavEnv


def test_env_clips_to_negative_vmin():
    env_cfg = {
        "map": {
            "size_m": [10.0, 10.0],
            "resolution_m": 0.5,
            "corridor_width_m": [3.0, 3.0],
            "wall_thickness_m": 0.3,
            "pallet_width_m": 1.0,
            "pallet_length_m": 0.6,
            "start_x_m": 1.0,
            "goal_margin_x_m": 1.0,
            "waypoint_step_m": 0.5,
            "min_passage_m": 0.7,
            "num_pallets_min": 1,
            "num_pallets_max": 1,
        },
        "lidar": {
            "beams": 8,
            "fov_deg": 180,
            "max_range_m": 4.0,
            "noise_enable": False,
        },
    }
    robot_cfg = {
        "v_max": 1.5,
        "w_max": 2.0,
        "v_min": -0.25,
        "radius_m": 0.25,
        "controller": {"lookahead_m": 1.0, "speed_nominal": 0.2},
    }
    reward_cfg = {
        "sparse": {"goal": 200.0, "collision": -200.0, "timeout": -50.0},
        "weights": {"progress": 1.0, "path": 0.2, "effort": 0.01, "sparse": 1.0},
        "path_penalty": {"lateral_weight": 1.0, "heading_weight": 0.5},
        "effort_penalty": {"lambda_v": 1.0, "lambda_w": 1.0, "lambda_jerk": 0.05},
    }
    run_cfg = {"dt": 0.1, "max_steps": 10}
    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=0)

    # Force negative residual large enough so v_cmd would go below v_min without clipping
    residual = np.array([-1.0, 0.0], dtype=np.float32)
    _, _, _, _, _ = env.step(residual)
    last_v = env.get_render_payload()["last_u"][0]
    assert last_v >= robot_cfg["v_min"] - 1e-6
    assert last_v <= 0.0 + 1e-6  # should be non-positive under strong negative residual
