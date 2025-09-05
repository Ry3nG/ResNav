from __future__ import annotations

# Core robot limits
ROBOT_V_MIN_MPS: float = 0.0
ROBOT_V_MAX_MPS: float = 1.5
ROBOT_W_MIN_RPS: float = -2.0
ROBOT_W_MAX_RPS: float = 2.0

# Geometry
GRID_RESOLUTION_M: float = 0.05
ROBOT_DIAMETER_M: float = 0.5
ROBOT_SAFETY_MARGIN_M: float = 0.0
ROBOT_COLLISION_RADIUS_M: float = (ROBOT_DIAMETER_M / 2.0) + ROBOT_SAFETY_MARGIN_M

# LiDAR
LIDAR_NUM_BEAMS: int = 24
LIDAR_FOV_DEG: float = 240.0
LIDAR_MAX_RANGE_M: float = 4.0

# Episode
GOAL_TOLERANCE_M: float = 0.3
DT: float = 0.1
EPISODE_MAX_STEPS: int = 400

# Path preview
PATH_PREVIEW_K: int = 5
PATH_PREVIEW_DS: float = 0.6
PATH_PREVIEW_RANGE_M: float = 3.0

# Clearance shaping
CLEARANCE_SAFE_M: float = 0.5
