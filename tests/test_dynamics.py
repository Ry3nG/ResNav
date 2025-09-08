import math

from amr_env.sim.dynamics import UnicycleModel, UnicycleState


def test_unicycle_straight_motion():
    model = UnicycleModel(v_max=1.5, w_max=2.0)
    model.reset(UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0))
    dt = 0.1
    steps = 10
    for _ in range(steps):
        model.step((1.0, 0.0), dt)
    x, y, th = model.as_pose()
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
    assert abs(th - 0.0) < 1e-12


def test_unicycle_turn_angle_wrap():
    model = UnicycleModel(v_max=1.5, w_max=2.0)
    # Start near +pi and turn slightly positive to test wrap
    model.reset(UnicycleState(0.0, 0.0, 3.2, 0.0, 0.0))
    dt = 0.1
    model.step((0.0, 2.0), dt)  # omega=2.0 rad/s for 0.1s -> +0.2 rad
    _, _, th = model.as_pose()
    assert -math.pi <= th <= math.pi

