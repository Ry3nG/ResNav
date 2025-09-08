import math
import numpy as np

from amr_env.sim.lidar import GridLidar


def make_grid(w_m=10.0, h_m=10.0, res=1.0):
    W = int(round(w_m / res))
    H = int(round(h_m / res))
    return np.zeros((H, W), dtype=bool), W, H


def test_empty_grid_returns_max_range():
    grid, W, H = make_grid(10.0, 10.0, 1.0)
    lidar = GridLidar(beams=5, fov_deg=180, max_range_m=4.0, noise_enable=False, resolution_m=1.0)
    d = lidar.sense(grid, (5.0, 5.0, 0.0))
    assert d.shape == (5,)
    assert np.allclose(d, 4.0)


def test_wall_hit_distance():
    grid, W, H = make_grid(10.0, 10.0, 1.0)
    # Horizontal wall at y in [5,6): row index i=5
    grid[5, :] = True
    lidar = GridLidar(beams=1, fov_deg=0.0, max_range_m=10.0, noise_enable=False, resolution_m=1.0)
    # Robot at (5,2), facing up (pi/2). Distance to y=5 boundary is 3.0
    d = lidar.sense(grid, (5.0, 2.0, math.pi / 2))
    assert d.shape == (1,)
    assert abs(d[0] - 3.0) < 1e-9


def test_inside_obstacle_returns_zero():
    grid, W, H = make_grid(10.0, 10.0, 1.0)
    grid[5, 5] = True
    lidar = GridLidar(beams=3, fov_deg=90.0, max_range_m=10.0, noise_enable=False, resolution_m=1.0)
    # Inside occupied cell (x in [5,6), y in [5,6)) -> all zeros
    d = lidar.sense(grid, (5.5, 5.5, 0.0))
    assert np.allclose(d, 0.0)


def test_boundary_exit_distance():
    grid, W, H = make_grid(10.0, 10.0, 1.0)
    lidar = GridLidar(beams=1, fov_deg=0.0, max_range_m=5.0, noise_enable=False, resolution_m=1.0)
    # Near top boundary at y=9.5, facing up -> boundary at y=10.0 => distance 0.5
    d = lidar.sense(grid, (5.0, 9.5, math.pi / 2))
    assert abs(d[0] - 0.5) < 1e-9


def test_determinism_with_seed_and_noise():
    grid, W, H = make_grid(10.0, 10.0, 1.0)
    # Add a simple obstacle to avoid trivial boundary-only case
    grid[7, 5] = True
    l1 = GridLidar(beams=5, fov_deg=120.0, max_range_m=10.0, noise_enable=True, noise_std_m=0.05, resolution_m=1.0)
    l2 = GridLidar(beams=5, fov_deg=120.0, max_range_m=10.0, noise_enable=True, noise_std_m=0.05, resolution_m=1.0)
    l1.set_seed(123)
    l2.set_seed(123)
    p = (5.0, 5.0, 0.0)
    d1 = l1.sense(grid, p)
    d2 = l2.sense(grid, p)
    assert np.allclose(d1, d2)

