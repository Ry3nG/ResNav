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

    def step(self, dt: float, t: float) -> None:
        if not self.active or t < self.spawn_t:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt


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


def sample_movers_for_omcf(
    env_cfg: dict,
    scenario_info: dict,
    rng: Optional[np.random.Generator],
) -> List[DiscMover]:
    """Sample movers according to env_cfg.dynamic_movers."""
    dm_cfg = env_cfg.get("dynamic_movers", {})
    if not dm_cfg.get("enabled", False):
        return []
    r = rng or np.random.default_rng()

    speed_lo, speed_hi = map(float, dm_cfg.get("speed_mps", [0.6, 1.2]))
    radius = float(dm_cfg.get("radius_m", 0.45))
    lane_offset = float(dm_cfg.get("lane_offset_m", 0.8))
    spawn_jitter = float(dm_cfg.get("spawn_jitter_m", 0.3))
    map_width = float(env_cfg["map"]["size_m"][0])
    movers: List[DiscMover] = []

    def _spawn_range(section: dict) -> Tuple[float, float]:
        rng_vals = section.get("spawn_time_range_s")
        if isinstance(rng_vals, (list, tuple)) and len(rng_vals) == 2:
            lo, hi = float(rng_vals[0]), float(rng_vals[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        default = dm_cfg.get("spawn_time_range_s", [0.0, 0.0])
        if isinstance(default, (list, tuple)) and len(default) == 2:
            lo, hi = float(default[0]), float(default[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        return (0.0, 0.0)

    # Movers from the right boundary (counterflow).
    fr_cfg = dm_cfg.get("from_right", {})
    n_fr = int(
        r.integers(
            int(fr_cfg.get("count_min", 0)),
            int(fr_cfg.get("count_max", 0)) + 1,
        )
    )
    lanes = fr_cfg.get("lanes", ["center", "up", "down"])
    y_top = float(scenario_info.get("y_top"))
    y_bot = float(scenario_info.get("y_bot"))
    cy = 0.5 * (y_top + y_bot)
    lane_choices: List[float] = []
    if "center" in lanes:
        lane_choices.append(cy)
    if "up" in lanes:
        lane_choices.append(cy + lane_offset)
    if "down" in lanes:
        lane_choices.append(cy - lane_offset)
    for _ in range(n_fr):
        base_y = float(r.choice(lane_choices)) if lane_choices else cy
        y0 = base_y + float(r.uniform(-spawn_jitter, spawn_jitter))
        speed = float(r.uniform(speed_lo, speed_hi))
        delay_lo, delay_hi = _spawn_range(fr_cfg)
        movers.append(
            DiscMover(
                x=map_width - 1e-3,
                y=y0,
                vx=-speed,
                vy=0.0,
                radius_m=radius,
                spawn_t=float(r.uniform(delay_lo, delay_hi)),
            )
        )

    # Movers emerging from the wall holes (merge).
    fh_cfg = dm_cfg.get("from_holes", {})
    holes = scenario_info.get("holes_x") or []
    if fh_cfg.get("enabled", False) and holes:
        count = int(
            r.integers(
                int(fh_cfg.get("count_min", 0)),
                int(fh_cfg.get("count_max", 0)) + 1,
            )
        )
        hole_side_choices = fh_cfg.get("hole_choice", ["top", "bottom", "both"])
        for _ in range(count):
            x_h = float(r.choice(holes))
            side = r.choice(hole_side_choices)
            if side == "both":
                side = r.choice(["top", "bottom"])
            side_top = side == "top"
            y_edge = y_top if side_top else y_bot
            # spawn just outside wall then move vertically through the gap.
            y0 = y_edge + (0.6 if side_top else -0.6)
            speed = float(r.uniform(speed_lo, speed_hi))
            vy = -speed if side_top else speed
            delay_lo, delay_hi = _spawn_range(fh_cfg)
            movers.append(
                DiscMover(
                    x=x_h,
                    y=y0,
                    vx=0.0,
                    vy=vy,
                    radius_m=radius,
                    spawn_t=float(r.uniform(delay_lo, delay_hi)),
                )
            )

    return movers
