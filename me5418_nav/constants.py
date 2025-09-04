from __future__ import annotations

"""
Centralized constants for environment, robot, LiDAR, and map defaults.

Only keep constants that are used across the codebase to avoid drift.
"""

# Environment / Map
GRID_RESOLUTION_M = 0.05  # meters per cell (grid + EDT resolution)

# Robot
ROBOT_DIAMETER_M = 0.5
ROBOT_RADIUS_M = ROBOT_DIAMETER_M / 2.0
ROBOT_V_MAX_MPS = 1.5
ROBOT_V_MIN_MPS = 0.0  # no backward by default; adjust if needed
ROBOT_W_MAX_RPS = 2.0
ROBOT_W_MIN_RPS = -2.0
CONTROL_FREQ_HZ = 10.0
DT_S = 1.0 / CONTROL_FREQ_HZ
GOAL_RADIUS_M = 0.3

# LiDAR
LIDAR_BEAMS = 24  # or 36
LIDAR_FOV_DEG = 240.0
LIDAR_RANGE_M = 4.0
LIDAR_STEP_M = 0.02  # ray marching step

# Episode / Evaluation
EPISODE_TIME_S = 120.0
