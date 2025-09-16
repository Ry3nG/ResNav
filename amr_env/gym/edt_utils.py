from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def compute_edt_meters(free_mask: np.ndarray, resolution_m: float) -> Tuple[np.ndarray, float]:
    """Compute Euclidean distance transform (in meters) for a binary free-space mask.

    Args:
        free_mask: array where non-zero entries mark free cells; zeros mark obstacles.
        resolution_m: map resolution in meters per cell.

    Returns:
        A tuple (edt_meters, build_time_ms).
    """
    t0 = time.time()
    edt = None

    # Primary path: SciPy's fast exact EDT.
    try:
        from scipy.ndimage import distance_transform_edt as _edt

        edt = _edt((free_mask > 0).astype(np.uint8)) * float(resolution_m)
    except Exception:
        # Fall back to an exact NumPy implementation (Felzenszwalb & Huttenlocher 2004).
        edt = _edt_numpy_fh((free_mask > 0).astype(np.uint8)) * float(resolution_m)

    elapsed_ms = (time.time() - t0) * 1000.0
    return edt.astype(np.float32), float(elapsed_ms)


def _edt_numpy_fh(free_u8: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance transform using the FH two-pass algorithm."""
    inf = 10**9
    h, w = free_u8.shape
    # Distance to obstacle: treat obstacles as zeros, free cells as inf.
    dist = np.where(free_u8 == 0, 0.0, float(inf))

    for x in range(w):
        dist[:, x] = _edt_1d(dist[:, x])
    for y in range(h):
        dist[y, :] = _edt_1d(dist[y, :])
    return np.sqrt(dist, out=dist)


def _edt_1d(column: np.ndarray) -> np.ndarray:
    """1D squared distance transform (Felzenszwalb & Huttenlocher)."""
    n = column.shape[0]
    v = np.zeros(n, dtype=np.int32)  # Locations of parabolas.
    z = np.zeros(n + 1, dtype=np.float64)  # Separation points.
    g = column.astype(np.float64)

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        s = _intersection(g, v[k], q)
        while s <= z[k]:
            k -= 1
            s = _intersection(g, v[k], q)
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    out = np.empty(n, dtype=np.float64)
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        out[q] = dq * dq + g[v[k]]
    return out


def _intersection(g: np.ndarray, p: int, q: int) -> float:
    return ((g[q] + q * q) - (g[p] + p * p)) / (2.0 * (q - p))
