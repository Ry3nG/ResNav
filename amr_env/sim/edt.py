from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_edt_meters(free_mask: np.ndarray, resolution_m: float) -> Tuple[np.ndarray, float]:
    """Compute Euclidean distance transform (in meters) for a binary free-space mask."""
    t0 = time.time()
    free = (free_mask > 0).astype(np.uint8)
    edt = distance_transform_edt(free) * float(resolution_m)
    elapsed_ms = (time.time() - t0) * 1000.0
    return edt.astype(np.float32), float(elapsed_ms)
