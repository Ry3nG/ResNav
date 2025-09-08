import numpy as np

from amr_env.sim.scenarios import (
    BlockageScenarioConfig,
    create_blockage_scenario,
)
from amr_env.sim.collision import inflate_grid


def test_blockage_basic_shapes_and_ranges():
    cfg = BlockageScenarioConfig(
        map_width_m=10.0,
        map_height_m=10.0,
        corridor_width_min_m=3.0,
        corridor_width_max_m=3.0,
        resolution_m=0.5,
        num_pallets_min=1,
        num_pallets_max=1,
        min_passage_m=0.7,
    )
    rng = np.random.default_rng(0)
    grid, waypoints, start_pose, goal_xy, info = create_blockage_scenario(cfg, rng)

    assert grid.ndim == 2
    assert waypoints.shape[1] == 2
    assert isinstance(start_pose, tuple) and len(start_pose) == 3
    assert isinstance(goal_xy, tuple) and len(goal_xy) == 2

    # Corridor width matches config (since min=max)
    assert abs(info["corridor_width"] - 3.0) < 1e-9

    # Start and goal within map bounds
    w, h = cfg.map_width_m, cfg.map_height_m
    sx, sy, _ = start_pose
    gx, gy = goal_xy
    for x, y in [(sx, sy), (gx, gy)]:
        assert 0.0 <= x <= w
        assert 0.0 <= y <= h


def test_inflation_circle_subset_of_square():
    cfg = BlockageScenarioConfig(map_width_m=4.0, map_height_m=4.0, resolution_m=0.5, num_pallets_min=0, num_pallets_max=0)
    rng = np.random.default_rng(1)
    grid, _, _, _, _ = create_blockage_scenario(cfg, rng)
    # Add a single obstacle pixel in center
    H, W = grid.shape
    grid[H // 2, W // 2] = True
    # Inflate with circle (SciPy path if available, else our fallback square)
    infl_circ = inflate_grid(grid, radius_m=0.6, resolution_m=cfg.resolution_m)
    # Compute explicit square dilation for comparison
    r_cells = int(np.ceil(0.6 / cfg.resolution_m))
    out_sq = np.zeros_like(grid)
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            i_src0 = max(0, -di)
            i_src1 = min(grid.shape[0], grid.shape[0] - di)
            j_src0 = max(0, -dj)
            j_src1 = min(grid.shape[1], grid.shape[1] - dj)
            i_dst0 = max(0, di)
            i_dst1 = i_dst0 + (i_src1 - i_src0)
            j_dst0 = max(0, dj)
            j_dst1 = j_dst0 + (j_src1 - j_src0)
            if i_src0 < i_src1 and j_src0 < j_src1:
                out_sq[i_dst0:i_dst1, j_dst0:j_dst1] |= grid[i_src0:i_src1, j_src0:j_src1]
    # Circle inflation should be a subset of square dilation
    assert np.all(infl_circ <= out_sq)
