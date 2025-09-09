import numpy as np

from amr_env.sim.dynamics import UnicycleModel, UnicycleState


def test_unicycle_reverse_clips_to_vmin():
    model = UnicycleModel(v_max=1.5, w_max=2.0, v_min=-0.3)
    model.reset(UnicycleState(0.0, 0.0, 0.0, 0.0, 0.0))
    s1 = model.step(action=(-1.0, 0.0), dt=0.1)
    # Applied velocity should be clipped to v_min
    assert np.isclose(s1.v, -0.3, atol=1e-6)
    # Integrate backward in x for a small step
    assert s1.x < 0.0
