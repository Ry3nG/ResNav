import math

import numpy as np

from amr_env.sim.scenarios_omcf import OMCFConfig, create_omcf_scenario


def test_holes_create_wall_gaps():
    cfg = OMCFConfig(
        holes_enabled=True,
        holes_count_pairs=1,
        holes_x_lo_m=15.0,
        holes_x_hi_m=15.0,
        holes_open_len_m=2.0,
    )
    grid, _, _, _, info = create_omcf_scenario(cfg, np.random.default_rng(0))
    y_top = float(info["y_top"])
    res = float(cfg.resolution_m)
    wall_cells = int(math.ceil(cfg.wall_thickness_m / res))
    i0 = int(math.floor(y_top / res))
    x_idx = int(round(15.0 / res))
    strip = grid[i0 : i0 + wall_cells + 1, x_idx]
    assert not np.any(strip), "Expected hole gap to keep wall cells free"
