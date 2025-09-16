"""Path utilities: projection, errors, and preview points."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Tuple

import numpy as np


@dataclass
class PathContext:
    d_lat: float
    theta_err: float
    previews_robot: np.ndarray  # shape (3, 2)


def _polyline_arclength(pts: np.ndarray) -> np.ndarray:
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    return s


def _project_to_polyline(pts: np.ndarray, p: np.ndarray) -> Tuple[float, np.ndarray, int, float]:
    """Project point p onto polyline pts.

    Returns (s_proj, proj_point, seg_idx, t) where s_proj is arclength position,
    seg_idx is segment index, and t in [0,1] along that segment.
    """
    best = (float("inf"), np.array([pts[0, 0], pts[0, 1]]), 0, 0.0, 0.0)
    s_acc = 0.0
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        L2 = float(np.dot(d, d))
        if L2 == 0.0:
            s_acc += 0.0
            continue
        t = float(np.dot(p - p0, d) / L2)
        t_clamped = max(0.0, min(1.0, t))
        proj = p0 + t_clamped * d
        dist2 = float(np.sum((proj - p) ** 2))
        if dist2 < best[0]:
            best = (dist2, proj, i, t_clamped, s_acc + t_clamped * float(np.sqrt(L2)))
        s_acc += float(np.sqrt(L2))
    _, proj, idx, t_clamped, s_proj = best
    return s_proj, proj, idx, t_clamped


def _heading_of_segment(pts: np.ndarray, idx: int) -> float:
    if idx >= len(pts) - 1:
        idx = len(pts) - 2
    d = pts[idx + 1] - pts[idx]
    return float(atan2(d[1], d[0]))


def _point_at_arclength(pts: np.ndarray, s: np.ndarray, s_target: float) -> np.ndarray:
    if s_target <= s[0]:
        return pts[0]
    if s_target >= s[-1]:
        return pts[-1]
    i = int(np.searchsorted(s, s_target) - 1)
    seg_len = s[i + 1] - s[i]
    if seg_len <= 0.0:
        return pts[i]
    alpha = (s_target - s[i]) / seg_len
    return pts[i] + alpha * (pts[i + 1] - pts[i])


def compute_path_context(
    pose: Tuple[float, float, float], waypoints: np.ndarray, preview_ds: Tuple[float, float, float]
) -> PathContext:
    """Compute lateral/heading errors and 3 preview points in robot frame."""
    x, y, th = pose
    p = np.array([x, y], dtype=float)
    s = _polyline_arclength(waypoints)
    s_proj, proj, idx, t = _project_to_polyline(waypoints, p)
    seg_theta = _heading_of_segment(waypoints, idx)

    # Lateral error sign by left/right of segment direction
    v = waypoints[idx + 1] - waypoints[idx]
    n = np.array([-v[1], v[0]])  # left normal
    sign = np.sign(np.dot(n, p - proj))
    d_lat = float(sign * np.linalg.norm(p - proj))

    theta_err = float((th - seg_theta + np.pi) % (2 * np.pi) - np.pi)

    previews_w = np.stack([
        _point_at_arclength(waypoints, s, s_proj + preview_ds[0]),
        _point_at_arclength(waypoints, s, s_proj + preview_ds[1]),
        _point_at_arclength(waypoints, s, s_proj + preview_ds[2]),
    ])

    # Transform previews to robot frame
    dx = previews_w[:, 0] - x
    dy = previews_w[:, 1] - y
    c = np.cos(-th)
    s_ = np.sin(-th)
    xr = c * dx - s_ * dy
    yr = s_ * dx + c * dy
    previews_robot = np.stack([xr, yr], axis=1)

    return PathContext(d_lat=d_lat, theta_err=theta_err, previews_robot=previews_robot)


def closest_and_lookahead(
    pose: Tuple[float, float, float],
    waypoints: np.ndarray,
    lookahead_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return closest projection and lookahead waypoint along a polyline."""

    x, y, _ = pose
    p = np.array([float(x), float(y)], dtype=float)
    s = _polyline_arclength(waypoints)
    s_proj, proj, _, _ = _project_to_polyline(waypoints, p)
    look_s = float(s_proj + float(lookahead_m))
    lookahead = _point_at_arclength(waypoints, s, look_s)
    return proj, lookahead
