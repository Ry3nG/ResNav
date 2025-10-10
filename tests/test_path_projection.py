import numpy as np

from amr_env.planning.path import closest_and_lookahead


def test_lookahead_monotonic() -> None:
    waypoints = np.stack([np.linspace(0, 10, 51), np.zeros(51)], axis=1)
    proj, lookahead = closest_and_lookahead((5.0, 1.0, 0.0), waypoints, 1.5)
    assert lookahead[0] >= proj[0]
