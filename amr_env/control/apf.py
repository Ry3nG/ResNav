"""Artificial Potential Field (APF) navigation controller.

Pure APF baseline for comparison with learned methods.
Combines attractive force to goal and repulsive forces from obstacles.

API: compute_apf_command(pose, goal, lidar_scan, lidar_angles, params) -> (v, omega)
"""

from __future__ import annotations

import numpy as np
from math import cos, sin, atan2, sqrt


def compute_apf_command(
    pose: tuple[float, float, float],
    goal: tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_angles: np.ndarray,
    v_max: float = 1.5,
    v_min: float = -0.5,  # Allow reverse!
    w_max: float = 2.0,
    k_att: float = 2.0,  # Tuned: goal attraction
    k_rep: float = 2.0,  # Tuned: obstacle repulsion
    d_influence: float = 0.5,  # Tuned: obstacle influence radius
    k_v: float = 1.0,  # Tuned: velocity gain
    k_omega: float = 4.0,  # Tuned: angular velocity gain
    robot_radius: float = 0.45,  # Added robot radius parameter
) -> tuple[float, float]:
    """
    Compute APF control command.

    Args:
        pose: Current robot pose (x, y, theta) in world frame
        goal: Goal position (x, y) in world frame
        lidar_scan: Array of LiDAR range measurements (N_beams,)
        lidar_angles: Array of LiDAR beam angles relative to robot heading (N_beams,)
        v_max: Maximum linear velocity (m/s)
        w_max: Maximum angular velocity (rad/s)
        k_att: Attractive potential gain
        k_rep: Repulsive potential gain
        d_influence: Influence distance for repulsive field (m)
        k_v: Velocity gain for converting force magnitude to speed
        k_omega: Angular velocity gain for heading control
        robot_radius: Robot radius for collision avoidance (m)

    Returns:
        (v, omega): Linear and angular velocity commands
    """
    x, y, theta = pose
    gx, gy = goal

    # 1. Attractive force (toward goal)
    dx_goal = gx - x
    dy_goal = gy - y
    dist_to_goal = sqrt(dx_goal**2 + dy_goal**2)

    if dist_to_goal < 0.01:
        # Already at goal
        return 0.0, 0.0

    goal_direction = np.array([dx_goal, dy_goal], dtype=float) / dist_to_goal

    # Attractive force: F_att = k_att * (goal - pos)
    # Increased saturation distance for stronger long-range pull
    att_magnitude = min(k_att * dist_to_goal, k_att * 5.0)  # Increased from 3.0 to 5.0
    F_att_x = att_magnitude * (dx_goal / dist_to_goal)
    F_att_y = att_magnitude * (dy_goal / dist_to_goal)

    # 2. Repulsive forces (from obstacles detected by LiDAR)
    F_rep_x = 0.0
    F_rep_y = 0.0

    # Safety buffer - robot needs extra clearance
    safety_buffer = 0.15  # Reduced from 0.25: less conservative
    min_safe_distance = robot_radius + safety_buffer

    for i, dist in enumerate(lidar_scan):
        # Adjust distance to account for robot radius
        # The LiDAR measures from center, but robot edge extends by robot_radius
        effective_dist = dist - robot_radius

        if (
            effective_dist < d_influence and effective_dist > 0.05
        ):  # Avoid very close/noise
            # Obstacle position in world frame
            beam_angle_world = theta + lidar_angles[i]
            obs_x = x + dist * cos(beam_angle_world)
            obs_y = y + dist * sin(beam_angle_world)

            # Direction away from obstacle
            dx_obs = x - obs_x
            dy_obs = y - obs_y

            # Repulsive force magnitude - use effective distance
            # Moderate emergency repulsion when too close
            if effective_dist < min_safe_distance:
                # Emergency: strong repulsion but not dominating
                rep_magnitude = (
                    k_rep * 8.0 / max(effective_dist**2, 0.01)
                )  # Reduced from 15.0
            else:
                # Normal repulsion
                rep_magnitude = (
                    k_rep
                    * (1.0 / effective_dist - 1.0 / d_influence)
                    * (1.0 / (effective_dist**2))
                )

            # Normalize direction
            obs_dist = sqrt(dx_obs**2 + dy_obs**2)
            if obs_dist > 0.01:
                F_rep_x += rep_magnitude * (dx_obs / obs_dist)
                F_rep_y += rep_magnitude * (dy_obs / obs_dist)

    # 3. Total force
    F_total_x = F_att_x + F_rep_x
    F_total_y = F_att_y + F_rep_y

    # Encourage forward progress during avoidance: keep some goal-directed component
    F_total_vec = np.array([F_total_x, F_total_y], dtype=float)
    min_range = float(np.min(lidar_scan)) if lidar_scan.size else float("inf")
    min_effective_dist = min_range - robot_radius
    safe_norm = max(min_safe_distance, 1e-3)
    if np.isfinite(min_effective_dist):
        clearance_ratio = np.clip(min_effective_dist / safe_norm, 0.0, 1.0)
    else:
        clearance_ratio = 1.0

    front_angle_rad = np.deg2rad(25.0)
    front_clearance = float("inf")
    if lidar_angles is not None:
        angles = np.asarray(lidar_angles)
        if angles.shape == lidar_scan.shape:
            mask = np.abs(angles) <= front_angle_rad
            if np.any(mask):
                front_clearance = float(np.min(lidar_scan[mask]) - robot_radius)

    forward_component = float(np.dot(F_total_vec, goal_direction))
    min_forward_component = 0.15 * att_magnitude * clearance_ratio
    if np.isfinite(front_clearance) and front_clearance < min_safe_distance:
        front_scale = np.clip(front_clearance / min_safe_distance, 0.0, 1.0)
        min_forward_component = min_forward_component * front_scale

    if forward_component < min_forward_component:
        lateral_component = F_total_vec - forward_component * goal_direction
        F_total_vec = lateral_component + min_forward_component * goal_direction
        F_total_x = float(F_total_vec[0])
        F_total_y = float(F_total_vec[1])

    # 4. Convert to unicycle commands
    # Desired heading from total force
    desired_heading = atan2(F_total_y, F_total_x)

    # Heading error (wrapped to [-pi, pi])
    heading_error = desired_heading - theta
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Linear velocity: proportional to force magnitude
    # CRITICAL: When heading error is large (robot facing wrong way), velocity can be negative (reverse)
    force_along_heading = F_total_x * cos(theta) + F_total_y * sin(theta)

    # Velocity proportional to force projection
    v = k_v * force_along_heading

    # Apply safety limits - less conservative velocity scaling
    if np.isfinite(min_effective_dist) and min_effective_dist < min_safe_distance:
        # Less aggressive velocity reduction when close to obstacles
        safety_factor = max(
            0.3, min_effective_dist / min_safe_distance
        )  # Increased floor from 0.05 to 0.3
        v = v * safety_factor

    # If an obstacle is directly ahead and too close, command a reverse motion
    if np.isfinite(front_clearance) and front_clearance < min_safe_distance:
        backoff_ratio = np.clip(
            (min_safe_distance - front_clearance) / max(min_safe_distance, 1e-3),
            0.0,
            1.0,
        )
        max_reverse = abs(v_min)
        desired_backoff = -(
            0.05 + (max_reverse - 0.05) * backoff_ratio
        )  # Smooth ramp toward reverse limit
        v = min(v, desired_backoff)

    # Clamp to velocity limits
    v = np.clip(v, v_min, v_max)

    # Angular velocity: proportional to heading error
    omega = k_omega * heading_error
    omega = np.clip(omega, -w_max, w_max)

    return float(v), float(omega)


def compute_apf_command_with_dict(
    pose: tuple[float, float, float],
    goal: tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_angles: np.ndarray,
    params: dict,
) -> tuple[float, float]:
    """
    Wrapper that accepts parameters as a dictionary.

    Useful for configuration management and hyperparameter tuning.
    """
    return compute_apf_command(
        pose=pose,
        goal=goal,
        lidar_scan=lidar_scan,
        lidar_angles=lidar_angles,
        v_max=params.get("v_max", 1.5),
        w_max=params.get("w_max", 2.0),
        k_att=params.get("k_att", 1.0),
        k_rep=params.get("k_rep", 2.0),
        d_influence=params.get("d_influence", 1.5),
        k_v=params.get("k_v", 0.8),
        k_omega=params.get("k_omega", 3.0),
    )
