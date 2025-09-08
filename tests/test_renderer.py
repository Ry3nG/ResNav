"""Headless renderer smoke test to validate basic frame generation."""

import pytest
import numpy as np


def test_headless_renderer_smoke():
    try:
        from visualization.pygame_renderer import Renderer, VizConfig
    except Exception:
        pytest.skip("pygame not available")
    # Minimal map and empty grids
    map_size = (4.0, 3.0)
    res = 1.0
    rend = Renderer(
        map_size, res, viz_cfg=VizConfig(size_px=(200, 150), fps=5), display=False
    )
    H = int(map_size[1] / res)
    W = int(map_size[0] / res)
    raw = np.zeros((H, W), dtype=bool)
    frame = rend.render_frame(
        raw_grid=raw,
        inflated_grid=None,
        pose=(1.0, 1.0, 0.0),
        radius_m=0.2,
        lidar=None,
        path=None,
        proj=None,
        lookahead=None,
        actions=None,
        hud={"reward": 0.0},
    )
    assert frame.get_width() == 200 and frame.get_height() == 150
