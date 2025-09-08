import numpy as np

from amr_env.gym.residual_nav_env import ResidualNavEnv
from amr_env.sim.dynamics import UnicycleState


def make_cfgs():
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
        "lidar": {"beams": 8, "fov_deg": 180, "max_range_m": 4.0, "noise_enable": False},
    }
    robot_cfg = {"v_max": 1.5, "w_max": 2.0, "radius_m": 0.25, "controller": {"lookahead_m": 1.0, "speed_nominal": 0.5}}
    reward_cfg = {
        "sparse": {"goal": 200.0, "collision": -200.0, "timeout": -50.0},
        "weights": {"progress": 1.0, "path": 0.2, "effort": 0.01, "sparse": 1.0},
        "path_penalty": {"lateral_weight": 1.0, "heading_weight": 0.5},
        "effort_penalty": {"lambda_v": 1.0, "lambda_w": 1.0, "lambda_jerk": 0.05},
    }
    run_cfg = {"dt": 0.1, "max_steps": 50}
    return env_cfg, robot_cfg, reward_cfg, run_cfg


def test_env_reset_step_shapes():
    env_cfg, robot_cfg, reward_cfg, run_cfg = make_cfgs()
    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"lidar", "kin", "path"}
    assert obs["lidar"].shape[0] == env_cfg["lidar"]["beams"]
    assert obs["kin"].shape == (4,)
    assert obs["path"].shape == (8,)

    a = np.array([0.0, 0.0], dtype=np.float32)
    o2, r, term, trunc, info = env.step(a)
    assert set(o2.keys()) == {"lidar", "kin", "path"}


def test_start_inside_obstacle_terminates():
    env_cfg, robot_cfg, reward_cfg, run_cfg = make_cfgs()
    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=0)
    # Force robot into an occupied inflated cell
    occ = np.argwhere(env._grid_inflated)
    assert occ.size > 0
    i, j = occ[0]
    x = (j + 0.5) * env.resolution_m
    y = (i + 0.5) * env.resolution_m
    env._model.reset(UnicycleState(x, y, 0.0, 0.0, 0.0))
    o2, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert term is True


def test_goal_reach_gives_sparse_reward():
    env_cfg, robot_cfg, reward_cfg, run_cfg = make_cfgs()
    env = ResidualNavEnv(env_cfg, robot_cfg, reward_cfg, run_cfg)
    env.reset(seed=0)
    gx, gy = env._goal_xy
    env._model.reset(UnicycleState(gx - 0.1, gy, 0.0, 0.0, 0.0))
    _, r, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert term is True
    # Goal reward positive
    assert r >= reward_cfg["sparse"]["goal"] - 1.0
