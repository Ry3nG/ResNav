import numpy as np

from amr_env.sim.lidar import GridLidar


def test_lidar_empty_returns_max() -> None:
    grid = np.zeros((100, 100), dtype=bool)
    lidar = GridLidar(
        beams=8,
        fov_deg=90,
        max_range_m=4.0,
        noise_enable=False,
        resolution_m=0.1,
    )
    distances = lidar.sense(grid, (5.0, 5.0, 0.0), noise=False)
    assert np.allclose(distances, 4.0)
