"""Pure Pursuit tracker (minimal implementation).

API: compute_u_track(pose, waypoints, lookahead_m, v_nominal) -> (v, w)
Pose is (x, y, theta). Waypoints is (N, 2) polyline in world frame.
"""

from __future__ import annotations

from math import cos, sin
from typing import Tuple

import numpy as np

from amr_env.gym.path_utils import closest_and_lookahead


def compute_u_track(
    pose: Tuple[float, float, float], waypoints: np.ndarray, lookahead_m: float, v_nominal: float
) -> Tuple[float, float]:
    """Compute tracker command (v, w) using Pure Pursuit geometry."""
    x, y, th = pose
    _, target = closest_and_lookahead(pose, waypoints, lookahead_m)

    # Transform target into robot frame
    dx = float(target[0] - x)
    dy = float(target[1] - y)
    xr = cos(-th) * dx - sin(-th) * dy
    yr = sin(-th) * dx + cos(-th) * dy

    # Curvature kappa = 2*yr / L^2
    L2 = xr * xr + yr * yr
    # Robustness: avoid excessive curvature when target is extremely close
    if L2 <= (0.05 ** 2):
        kappa = 0.0
    else:
        kappa = 2.0 * yr / L2

    v = float(v_nominal)
    w = float(v * kappa)
    return v, w
