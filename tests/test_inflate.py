import numpy as np

from amr_env.sim.collision import inflate_grid


def test_inflate_increases_occupied() -> None:
    grid = np.zeros((20, 20), dtype=bool)
    grid[10, 10] = True
    inflated = inflate_grid(grid, radius_m=0.5, resolution_m=0.1)
    assert inflated.sum() > 1
