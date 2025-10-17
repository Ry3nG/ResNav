import numpy as np

from amr_env.sim.movers import DiscMover, rasterize_disc, sample_movers_for_omcf


def test_rasterize_disc_marks_cells():
    grid = np.zeros((100, 100), dtype=bool)
    mover = DiscMover(x=5.0, y=5.0, vx=0.0, vy=0.0, radius_m=0.5)
    rasterize_disc(grid, mover.x, mover.y, mover.radius_m, res=0.1)
    assert grid.sum() > 0


def test_hole_movers_travel_vertically():
    env_cfg = {
        "map": {"size_m": [20.0, 20.0]},
        "dynamic_movers": {
            "enabled": True,
            "radius_m": 0.45,
            "speed_mps": [0.8, 0.8],
            "lane_offset_m": 0.8,
            "spawn_jitter_m": 0.0,
            "spawn_time_range_s": [0.0, 0.0],
            "from_right": {"count_min": 0, "count_max": 0},
            "from_holes": {
                "enabled": True,
                "count_min": 1,
                "count_max": 1,
                "hole_choice": ["top"],
                "spawn_time_range_s": [0.0, 0.0],
            },
        },
    }
    scenario_info = {"y_top": 11.0, "y_bot": 9.0, "holes_x": [15.0]}
    movers = sample_movers_for_omcf(env_cfg, scenario_info, np.random.default_rng(0))
    assert movers, "Expected at least one hole mover"
    for mover in movers:
        if abs(mover.vy) > 1e-6:
            assert abs(mover.vx) < 1e-6


def test_spawn_time_range_respected():
    env_cfg = {
        "map": {"size_m": [20.0, 20.0]},
        "dynamic_movers": {
            "enabled": True,
            "radius_m": 0.45,
            "speed_mps": [0.5, 0.5],
            "lane_offset_m": 0.5,
            "spawn_jitter_m": 0.0,
            "spawn_time_range_s": [1.0, 1.0],
            "from_right": {"count_min": 1, "count_max": 1, "lanes": ["center"], "spawn_time_range_s": [2.0, 2.0]},
            "from_holes": {"enabled": False, "count_min": 0, "count_max": 0},
        },
    }
    scenario_info = {"y_top": 11.0, "y_bot": 9.0, "holes_x": []}
    movers = sample_movers_for_omcf(env_cfg, scenario_info, np.random.default_rng(0))
    assert movers
    for mover in movers:
        assert abs(mover.spawn_t - 2.0) < 1e-6
