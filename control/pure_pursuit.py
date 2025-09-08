"""Pure Pursuit tracker (minimal implementation).

API: compute_u_track(pose, waypoints, lookahead_m, v_nominal) -> (v, w)
Pose is (x, y, theta). Waypoints is (N, 2) polyline in world frame.
"""

from __future__ import annotations

from math import atan2, cos, sin
from typing import Tuple

import numpy as np


def _closest_point_and_lookahead(
    pose: Tuple[float, float, float], waypoints: np.ndarray, lookahead_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return closest point on path and lookahead target point along polyline."""
    x, y, _ = pose
    pts = waypoints
    # Find closest segment by projection
    best_dist2 = float("inf")
    best_proj = pts[0]
    best_s = 0.0
    s_acc = 0.0
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        L2 = float(np.dot(d, d))
        if L2 == 0.0:
            continue
        t = max(0.0, min(1.0, float(np.dot(np.array([x, y]) - p0, d) / L2)))
        proj = p0 + t * d
        dist2 = float(np.sum((proj - np.array([x, y])) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_proj = proj
            best_s = s_acc + t * float(np.sqrt(L2))
        s_acc += float(np.sqrt(L2))

    # March forward along polyline to reach lookahead distance
    target_s = best_s + float(lookahead_m)
    s_acc = 0.0
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if s_acc + seg_len < target_s:
            s_acc += seg_len
            continue
        # Inside this segment
        remain = target_s - s_acc
        alpha = 0.0 if seg_len == 0.0 else remain / seg_len
        alpha = max(0.0, min(1.0, alpha))
        look = p0 + alpha * (p1 - p0)
        return best_proj, look
    # If beyond end, use final point
    return best_proj, pts[-1]


def compute_u_track(
    pose: Tuple[float, float, float], waypoints: np.ndarray, lookahead_m: float, v_nominal: float
) -> Tuple[float, float]:
    """Compute tracker command (v, w) using Pure Pursuit geometry."""
    x, y, th = pose
    closest, target = _closest_point_and_lookahead(pose, waypoints, lookahead_m)

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
