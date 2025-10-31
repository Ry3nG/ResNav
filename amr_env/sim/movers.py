"""Scripted disc movers for dynamic obstacle overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np


@dataclass
class DiscMover:
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float
    spawn_t: float = 0.0
    active: bool = True
    ax: float = 0.0
    ay: float = 0.0
    lifetime_s: float = 8.0
    mover_type: str = "lateral"  # "lateral" or "longitudinal"

    def step(
        self,
        dt: float,
        t: float,
        v_lo: float,
        v_hi: float,
        y_bot: float,
        y_top: float,
        reflect_walls: bool,
    ) -> None:
        """Update position and velocity with acceleration, clamping, and wall reflection.

        Args:
            dt: Time step
            t: Current simulation time
            v_lo: Minimum velocity magnitude
            v_hi: Maximum velocity magnitude
            y_bot: Bottom corridor boundary
            y_top: Top corridor boundary
            reflect_walls: Enable Y-direction wall reflection
        """
        # Early exit if not yet spawned
        if t < self.spawn_t:
            return

        # TTL expiry: deactivate and exit
        if t >= self.spawn_t + self.lifetime_s:
            self.active = False
            return

        # Apply acceleration
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Clamp speed to [v_lo, v_hi]
        spd = math.hypot(self.vx, self.vy)
        if spd > v_hi:
            scale = v_hi / (spd + 1e-9)
            self.vx *= scale
            self.vy *= scale
        elif spd < v_lo and spd > 1e-9:
            scale = v_lo / spd
            self.vx *= scale
            self.vy *= scale

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Y-direction wall reflection (optional)
        if reflect_walls:
            R = self.radius_m
            if self.y - R < y_bot:
                self.y = y_bot + R
                self.vy = abs(self.vy)
            elif self.y + R > y_top:
                self.y = y_top - R
                self.vy = -abs(self.vy)


def rasterize_disc(grid: np.ndarray, x: float, y: float, r_m: float, res: float) -> None:
    """Mark all cells inside a disc as occupied."""
    H, W = grid.shape
    radius = max(0.0, float(r_m))
    if radius <= 0.0:
        return
    jc = int(math.floor(x / res))
    ic = int(math.floor(y / res))
    rad_cells = int(math.ceil(radius / res))
    if rad_cells <= 0:
        return
    i0 = max(0, ic - rad_cells)
    i1 = min(H - 1, ic + rad_cells)
    j0 = max(0, jc - rad_cells)
    j1 = min(W - 1, jc + rad_cells)
    if i0 > i1 or j0 > j1:
        return
    yy, xx = np.ogrid[i0 : i1 + 1, j0 : j1 + 1]
    mask = (xx - jc) ** 2 + (yy - ic) ** 2 <= rad_cells**2
    grid[i0 : i1 + 1, j0 : j1 + 1][mask] = True


def spawn_lateral_mover(
    env_cfg: dict,
    scenario_info: dict,
    grid_inflated: np.ndarray,
    resolution_m: float,
    agent_pose: Tuple[float, float, float],
    t_now: float,
    rng: np.random.Generator,
) -> Optional[DiscMover]:
    """Spawn a lateral (along-hallway) mover with safety checks.

    Args:
        env_cfg: Environment configuration
        scenario_info: Scenario metadata with corridor bounds
        grid_inflated: Inflated static occupancy grid
        resolution_m: Grid resolution
        agent_pose: Current agent (x, y, theta)
        t_now: Current simulation time
        rng: Random number generator

    Returns:
        DiscMover instance or None if all attempts failed
    """
    dm_cfg = env_cfg.get("dynamic_movers", {})
    lat_cfg = dm_cfg.get("lateral", {})

    # Read parameters
    radius = float(dm_cfg.get("radius_m", 0.45))
    v_lo, v_hi = dm_cfg.get("velocity_range_mps", [0.4, 1.5])
    a_lo, a_hi = dm_cfg.get("acceleration_range_mps2", [-0.15, 0.15])
    ttl_lo, ttl_hi = dm_cfg.get("lifetime_range_s", [6.0, 10.0])
    safe_dist = float(lat_cfg.get("safe_spawn_distance_m", 3.0))
    lanes = lat_cfg.get("lanes", ["center", "up", "down"])
    lane_offset = float(dm_cfg.get("lane_offset_m", 0.8))

    map_width = float(env_cfg["map"]["size_m"][0])
    y_top = float(scenario_info.get("y_top"))
    y_bot = float(scenario_info.get("y_bot"))
    cy = 0.5 * (y_top + y_bot)

    # Build lane choices
    lane_choices: List[float] = []
    if "center" in lanes:
        lane_choices.append(cy)
    if "up" in lanes:
        lane_choices.append(cy + lane_offset)
    if "down" in lanes:
        lane_choices.append(cy - lane_offset)
    if not lane_choices:
        lane_choices = [cy]

    agent_x, agent_y, agent_theta = agent_pose
    H, W = grid_inflated.shape

    # Determine agent's forward direction (assume moving toward goal, i.e., +x)
    # For OMCF: agent moves left-to-right, so spawn ahead only
    min_forward_dist = max(safe_dist, 2.0)  # At least 2m ahead

    # K-retry loop
    for attempt in range(8):
        # Random x position (ONLY in front of agent)
        x_min = agent_x + min_forward_dist
        x_max = map_width - 1.0
        if x_min >= x_max:
            continue  # Agent too close to goal, skip lateral spawn
        x = float(rng.uniform(x_min, x_max))

        # Random lane + clamp to corridor bounds
        base_y = float(rng.choice(lane_choices))
        y = float(np.clip(base_y, y_bot + radius, y_top - radius))

        # Safety check 1: Euclidean distance from agent (still useful for y-axis)
        if math.hypot(x - agent_x, y - agent_y) < safe_dist:
            continue

        # Safety check 2: Static grid free space (center + neighborhood)
        jc = int(x / resolution_m)
        ic = int(y / resolution_m)
        rad_cells = max(1, int(math.ceil(radius / resolution_m)))
        if not (0 <= ic < H and 0 <= jc < W):
            continue
        i0 = max(0, ic - rad_cells)
        i1 = min(H - 1, ic + rad_cells)
        j0 = max(0, jc - rad_cells)
        j1 = min(W - 1, jc + rad_cells)
        if np.any(grid_inflated[i0 : i1 + 1, j0 : j1 + 1]):
            continue

        # Passed checks: generate mover moving TOWARD agent (counterflow)
        speed = float(rng.uniform(v_lo, v_hi))
        # Agent moves +x (right), so mover moves -x (left) to create counterflow
        vx = -speed  # Always counterflow
        vy = 0.0  # No vertical drift - stays in lane
        ax = float(rng.uniform(a_lo, a_hi))  # Horizontal acceleration for speed variability
        ay = 0.0  # No vertical acceleration - pure lateral motion
        ttl = float(rng.uniform(ttl_lo, ttl_hi))

        return DiscMover(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius_m=radius,
            spawn_t=t_now,
            ax=ax,
            ay=ay,
            lifetime_s=ttl,
            mover_type="lateral",
        )

    # K attempts failed
    return None


def spawn_longitudinal_mover(
    env_cfg: dict,
    scenario_info: dict,
    grid_inflated: np.ndarray,
    resolution_m: float,
    agent_pose: Tuple[float, float, float],
    t_now: float,
    rng: np.random.Generator,
) -> Optional[DiscMover]:
    """Spawn a longitudinal (from-hole) mover with safety checks.

    Args:
        env_cfg: Environment configuration
        scenario_info: Scenario metadata with holes_x list
        grid_inflated: Inflated static occupancy grid
        resolution_m: Grid resolution
        agent_pose: Current agent (x, y, theta)
        t_now: Current simulation time
        rng: Random number generator

    Returns:
        DiscMover instance or None if no holes or all attempts failed
    """
    dm_cfg = env_cfg.get("dynamic_movers", {})
    lon_cfg = dm_cfg.get("longitudinal", {})

    holes_x = scenario_info.get("holes_x") or []
    if not holes_x:  # Fallback: no holes available
        return None

    # Read parameters
    radius = float(dm_cfg.get("radius_m", 0.45))
    v_lo, v_hi = dm_cfg.get("velocity_range_mps", [0.4, 1.5])
    a_lo, a_hi = dm_cfg.get("acceleration_range_mps2", [-0.15, 0.15])
    ttl_lo, ttl_hi = dm_cfg.get("lifetime_range_s", [6.0, 10.0])
    safe_dist = float(lon_cfg.get("safe_spawn_distance_m", 3.0))

    y_top = float(scenario_info.get("y_top"))
    y_bot = float(scenario_info.get("y_bot"))
    agent_x, agent_y, _ = agent_pose

    # K-retry loop
    for attempt in range(8):
        x = float(rng.choice(holes_x))
        side_top = bool(rng.random() < 0.5)
        y_edge = y_top if side_top else y_bot
        y = y_edge + (0.6 if side_top else -0.6)

        # Safety check: Euclidean distance from agent
        if math.hypot(x - agent_x, y - agent_y) < safe_dist:
            continue

        # Velocity and acceleration (pure vertical motion through hole)
        speed = float(rng.uniform(v_lo, v_hi))
        vy = -speed if side_top else speed
        vx = 0.0  # Pure vertical, no horizontal drift
        ax = 0.0  # No horizontal acceleration
        ay = float(rng.uniform(a_lo, a_hi))  # Full range for speed variability (vertical only)
        ttl = float(rng.uniform(ttl_lo, ttl_hi))

        return DiscMover(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius_m=radius,
            spawn_t=t_now,
            ax=ax,
            ay=ay,
            lifetime_s=ttl,
            mover_type="longitudinal",
        )

    # K attempts failed
    return None


def sample_movers_for_omcf(
    env_cfg: dict,
    scenario_info: dict,
    rng: Optional[np.random.Generator],
) -> List[DiscMover]:
    """DEPRECATED: Continuous Poisson spawning is now used.

    Returns empty list. This function is kept for backward compatibility.
    """
    import warnings

    warnings.warn(
        "sample_movers_for_omcf() is deprecated. "
        "Dynamic movers now use continuous Poisson spawning.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []
