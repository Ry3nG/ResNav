from __future__ import annotations

"""
Centralized constants for environment, robot, LiDAR, and map defaults.
Values reflect the specifications documented in the wiki.
"""

# Environment / Map
MAP_SIZE_M = (40.0, 30.0)  # meters (width, height)
GRID_RESOLUTION_M = 0.1  # meters per cell (A* planning resolution)

# Aisle widths (for generators or validations)
AISLE_WIDTH_MIN_M = 1.4
AISLE_WIDTH_MAX_M = 1.8

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
LIDAR_FOV_DEG = 270.0
LIDAR_RANGE_M = 4.0
LIDAR_STEP_M = 0.02  # ray marching step

# Episode / Evaluation
EPISODE_TIME_S = 120.0

# Planning / Inflation
INFLATION_MARGIN_M = 0.05  # robot radius + margin used in builder validation

# Map generators
S_PATH_DEFAULT_RES = GRID_RESOLUTION_M
S_PATH_DEFAULT_AMP_M = 2.0
S_PATH_DEFAULT_PERIODS = 1.5
S_PATH_DEFAULT_OBS_FRAC = 0.05
S_PATH_SAMPLE_STEP_M = 0.1
START_GOAL_CLEAR_W_M = 0.6
START_GOAL_CLEAR_H_M = 0.6

# Controller defaults (Trap-aware PP+APF and PP+APF)
CTRL_LOOKAHEAD_M = 1.0
CTRL_V_NOMINAL_MPS = 0.8
CTRL_K_HEADING = 1.8  # trap-aware; baseline PP+APF uses 1.5
CTRL_REPULSE_DIST_M = 1.0
CTRL_REPULSE_GAIN = 1.0
CTRL_ATTRACT_GAIN = 1.0

# Trap-aware extras
CTRL_STUCK_WINDOW_S = 2.0
CTRL_MIN_IDX_PROGRESS = 10
CTRL_FOLLOW_CLEARANCE_M = 0.3
CTRL_FOLLOW_SPEED_MPS = 0.5
CTRL_TANGENTIAL_GAIN = 1.0
CTRL_NO_POINT_PIVOT_W_RPS = 0.6
CTRL_NO_POINT_PIVOT_V_MPS = 0.1
CTRL_CLEAR_HEADING_GAIN = 2.0
CTRL_TURN_SLOW_SCALE = 0.6
CTRL_CONF_SLOW = 0.6

# Rewards
REWARD_STEP = -0.01
REWARD_COLLISION = -1.0
REWARD_SUCCESS = 1.0
