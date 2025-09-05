from __future__ import annotations

import math
import numpy as np
from typing import Tuple


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PathTracker:
    """Continuous-progress path tracker for polyline paths.

    Maintains a monotonic arc-length pointer s_ptr and provides path errors
    (lateral and heading) and preview points ahead of s_ptr.
    """

    def __init__(self, waypoints: np.ndarray):
        pts = np.asarray(waypoints, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("waypoints must be (N,2)")
        keep = [0]
        for i in range(1, pts.shape[0]):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
                keep.append(i)
        self.P = pts[keep]
        if self.P.shape[0] < 2:
            raise ValueError("need at least 2 distinct waypoints")
        d = self.P[1:] - self.P[:-1]
        L = np.linalg.norm(d, axis=1)
        mask_valid = L > 1e-12
        self.d = d[mask_valid]
        self.L = L[mask_valid]
        self.Ps = self.P[:-1][mask_valid]
        self.S = np.concatenate([[0.0], np.cumsum(self.L)])
        self.S_end = float(self.S[-1])
        self.s_ptr = 0.0

    def _locate_by_s(self, s: float) -> int:
        s = float(max(0.0, min(s, self.S_end)))
        k = int(np.searchsorted(self.S, s, side="right") - 1)
        return max(0, min(k, self.L.shape[0] - 1))

    def sample_at_s(self, s: float) -> Tuple[float, float, np.ndarray]:
        if s <= 0.0:
            k = 0
            t = 0.0
        elif s >= self.S_end:
            k = self.L.shape[0] - 1
            t = 1.0
        else:
            k = self._locate_by_s(s)
            t = (s - self.S[k]) / (self.L[k] + 1e-12)
        q = self.Ps[k] + t * self.d[k]
        t_hat = self.d[k] / (self.L[k] + 1e-12)
        return float(q[0]), float(q[1]), t_hat

    def project_to_path(self, p: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, int]:
        p = np.asarray(p, dtype=float)
        m = self.L.shape[0]
        if m == 0:
            return 0.0, self.P[0], np.array([1.0, 0.0]), 0
        best = (float("inf"), 0.0, self.Ps[0], np.array([1.0, 0.0]), 0)
        for k in range(m):
            pk = self.Ps[k]
            dk = self.d[k]
            Lk = self.L[k]
            t = float(np.dot(p - pk, dk) / (Lk * Lk + 1e-12))
            t = float(np.clip(t, 0.0, 1.0))
            q = pk + t * dk
            dist2 = float(np.dot(p - q, p - q))
            if dist2 < best[0]:
                s = self.S[k] + t * Lk
                that = dk / (Lk + 1e-12)
                best = (dist2, s, q, that, k)
        return float(best[1]), np.asarray(best[2]), np.asarray(best[3]), int(best[4])

    def update_progress(self, p: np.ndarray):
        s_prev = self.s_ptr
        s_proj, q, that, k = self.project_to_path(p)
        self.s_ptr = max(self.s_ptr, s_proj)
        ds = max(0.0, self.s_ptr - s_prev)
        return s_prev, self.s_ptr, ds, q, that, k

    def errors(self, pose: Tuple[float, float, float]):
        x, y, th = pose
        _, _, _, q, t_hat, _ = self.update_progress(np.array([x, y], dtype=float))
        n_hat = np.array([-t_hat[1], t_hat[0]])
        e_lat = float(np.dot(np.array([x, y]) - q, n_hat))
        ang_t = math.atan2(t_hat[1], t_hat[0])
        e_head = _wrap_pi(ang_t - th)
        return e_lat, e_head

    def preview_points(self, K: int, ds: float) -> np.ndarray:
        out = []
        for i in range(1, K + 1):
            s = self.s_ptr + i * ds
            x, y, _ = self.sample_at_s(s)
            out.append([x, y])
        return np.asarray(out, dtype=float)

    def goal_reached(self, p: np.ndarray, goal_xy: tuple[float, float], tol: float) -> bool:
        near_end = self.s_ptr >= (self.S_end - 1e-4)
        dx = float(p[0] - goal_xy[0])
        dy = float(p[1] - goal_xy[1])
        return bool(near_end and (dx * dx + dy * dy) <= (tol * tol))

