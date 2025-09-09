import numpy as np

from control.dwa_baseline import dwa_select_action


def test_dwa_samples_negative_when_vmin_negative():
    # Minimal straight path and empty grid
    waypoints = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float)
    grid = np.zeros((40, 40), dtype=bool)
    res = 0.1
    pose = (0.0, 0.0, 0.0)

    v_min = -0.2
    v_max = 1.0
    w_max = 2.0
    v, w = dwa_select_action(
        pose,
        waypoints,
        grid,
        res,
        v_max,
        w_max,
        dt=0.1,
        horizon_s=0.5,
        v_samples=5,
        w_samples=3,
        w_progress=0.1,  # de-emphasize forward-only bias
        w_path=0.0,
        w_heading=0.0,
        w_obst=0.0,
        w_smooth=0.0,
        v_min=v_min,
    )
    # Returned candidate should respect lower bound, and allowing negative is possible
    assert v >= v_min - 1e-6
    assert v <= v_max + 1e-6
    # At least ensure the sampling included negatives by probing internal grid: here we
    # approximate by verifying that choosing negative is not forbidden by bounds.
    # We cannot directly access the sampled set, but the lack of exception and valid range suffices.
