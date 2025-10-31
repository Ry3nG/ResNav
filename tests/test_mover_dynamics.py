"""Unit tests for DiscMover dynamics (velocity clamp, TTL expiry, etc.)."""

import math
import numpy as np

from amr_env.sim.movers import DiscMover


def test_mover_velocity_clamp():
    """Velocity should be clamped to configured range after acceleration."""
    mover = DiscMover(
        x=5.0,
        y=5.0,
        vx=1.0,
        vy=0.0,
        radius_m=0.45,
        spawn_t=0.0,
        ax=1.0,  # Strong acceleration
        ay=0.0,
        lifetime_s=10.0,
    )

    v_lo = 0.4
    v_hi = 1.5
    dt = 0.1
    y_bot = 3.0
    y_top = 7.0

    # Step multiple times to accumulate acceleration
    for _ in range(20):
        mover.step(dt, 0.5, v_lo, v_hi, y_bot, y_top, reflect_walls=False)

    # Verify speed is clamped to v_hi
    spd = math.hypot(mover.vx, mover.vy)
    assert spd <= v_hi + 1e-6, f"Speed {spd} exceeds max {v_hi}"
    assert spd >= v_hi - 0.1, f"Speed {spd} should be near max {v_hi} after acceleration"


def test_mover_velocity_clamp_low():
    """Velocity should be clamped to minimum when decelerating."""
    mover = DiscMover(
        x=5.0,
        y=5.0,
        vx=0.5,
        vy=0.0,
        radius_m=0.45,
        spawn_t=0.0,
        ax=-0.3,  # Strong deceleration
        ay=0.0,
        lifetime_s=10.0,
    )

    v_lo = 0.4
    v_hi = 1.5
    dt = 0.1
    y_bot = 3.0
    y_top = 7.0

    # Step multiple times
    for _ in range(10):
        mover.step(dt, 0.5, v_lo, v_hi, y_bot, y_top, reflect_walls=False)

    # Verify speed is clamped to v_lo
    spd = math.hypot(mover.vx, mover.vy)
    assert spd >= v_lo - 1e-6, f"Speed {spd} below min {v_lo}"


def test_mover_ttl_expiry():
    """Mover should deactivate after lifetime_s expires."""
    mover = DiscMover(
        x=5.0,
        y=5.0,
        vx=1.0,
        vy=0.0,
        radius_m=0.45,
        spawn_t=0.0,
        lifetime_s=1.0,
    )

    v_lo = 0.4
    v_hi = 1.5
    y_bot = 3.0
    y_top = 7.0

    # Step before TTL expiry
    mover.step(dt=0.5, t=0.5, v_lo=v_lo, v_hi=v_hi, y_bot=y_bot, y_top=y_top, reflect_walls=False)
    assert mover.active, "Mover should still be active before TTL"

    # Step after TTL expiry
    mover.step(dt=0.6, t=1.1, v_lo=v_lo, v_hi=v_hi, y_bot=y_bot, y_top=y_top, reflect_walls=False)
    assert not mover.active, "Mover should be inactive after TTL expires"


def test_mover_spawn_delay():
    """Mover should not move before spawn_t."""
    mover = DiscMover(
        x=5.0,
        y=5.0,
        vx=1.0,
        vy=0.0,
        radius_m=0.45,
        spawn_t=2.0,  # Spawn at t=2.0
        lifetime_s=10.0,
    )

    v_lo = 0.4
    v_hi = 1.5
    y_bot = 3.0
    y_top = 7.0

    # Step before spawn_t
    initial_x = mover.x
    mover.step(dt=0.1, t=1.0, v_lo=v_lo, v_hi=v_hi, y_bot=y_bot, y_top=y_top, reflect_walls=False)
    assert mover.x == initial_x, "Mover should not move before spawn_t"
    assert mover.active, "Mover should remain active"

    # Step after spawn_t
    mover.step(dt=0.1, t=2.5, v_lo=v_lo, v_hi=v_hi, y_bot=y_bot, y_top=y_top, reflect_walls=False)
    assert mover.x > initial_x, "Mover should move after spawn_t"


def test_mover_wall_reflection():
    """Mover should reflect off top/bottom walls when enabled."""
    mover = DiscMover(
        x=5.0,
        y=6.8,  # Near top wall
        vx=0.0,
        vy=1.0,  # Moving up
        radius_m=0.2,
        spawn_t=0.0,
        lifetime_s=10.0,
    )

    v_lo = 0.4
    v_hi = 1.5
    y_bot = 3.0
    y_top = 7.0
    dt = 0.5

    # Step with reflection enabled
    mover.step(dt, t=0.5, v_lo=v_lo, v_hi=v_hi, y_bot=y_bot, y_top=y_top, reflect_walls=True)

    # Should be pushed back and velocity reversed
    assert mover.y <= y_top - mover.radius_m + 1e-6, "Mover should be inside wall boundary"
    assert mover.vy < 0, "Velocity should be reflected downward"
