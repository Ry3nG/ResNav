import numpy as np

from amr_env.control.pure_pursuit import compute_u_track


def test_pure_pursuit_straight_centerline_zero_curvature() -> None:
    waypoints = np.stack([np.linspace(0, 5, 21), np.zeros(21)], axis=1)
    v, w = compute_u_track((1.0, 0.0, 0.0), waypoints, lookahead_m=1.0, v_nominal=1.0)
    assert v == 1.0
    assert abs(w) < 1e-6
