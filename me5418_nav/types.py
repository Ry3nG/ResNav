from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Grids:
    """
    Container for the two-grid convention used throughout the project.

    - sensing: raw occupancy grid used for LiDAR and rendering
    - cspace: configuration-space grid (inflated) used for feasibility/collisions
    """

    sensing: Any
    cspace: Any
    edt: Optional[Any] = None  # distance-to-obstacle map in meters (numpy array) if available
    res: float = 0.1  # meters per cell

    def distance_to_obstacle(self, x: float, y: float) -> Optional[float]:
        """
        Euclidean distance (meters) from (x,y) to nearest obstacle in the sensing grid.
        Requires EDT. Returns None if EDT not available or indexing fails.
        """
        if self.edt is None:
            return None
        try:
            # Convert world (meters) to EDT array indices
            # Assuming sensing grid origin at (0,0) and same res/origin for edt
            gy = int(y / max(1e-9, self.res))
            gx = int(x / max(1e-9, self.res))
            H, W = self.edt.shape[:2]
            if gy < 0 or gx < 0 or gy >= H or gx >= W:
                return None
            d = float(self.edt[gy, gx])
            return d
        except Exception:
            return None

    def clearance(self, x: float, y: float, robot_radius: float) -> Optional[float]:
        """
        Clearance (meters): distance to nearest obstacle minus robot radius.
        Returns None if EDT not available.
        """
        d = self.distance_to_obstacle(x, y)
        if d is None:
            return None
        return float(d - float(robot_radius))
