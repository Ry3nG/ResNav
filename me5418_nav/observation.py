from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    lidar: np.ndarray       # (L,)
    kinematics: np.ndarray  # (2,) [v_norm, w_norm]
    path_errors: np.ndarray # (2,) [e_lat_norm, e_head_norm]
    preview: np.ndarray     # (K,2) in robot frame

    def flatten(self) -> np.ndarray:
        parts = [
            self.lidar.astype(np.float32),
            self.kinematics.astype(np.float32),
            self.path_errors.astype(np.float32),
            self.preview.astype(np.float32).reshape(-1),
        ]
        return np.concatenate(parts, axis=0)

