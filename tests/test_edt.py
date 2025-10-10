import numpy as np

from amr_env.sim.edt import compute_edt_meters


def test_edt_zero_on_obstacle() -> None:
    free = np.ones((10, 10), dtype=np.uint8)
    free[5, 5] = 0
    edt, _ = compute_edt_meters(free, 0.1)
    assert edt[5, 5] == 0.0
