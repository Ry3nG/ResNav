from __future__ import annotations

import os
import math
from typing import Tuple
import numpy as np


class PygameViewer:
    def __init__(self, map_w_m: float, map_h_m: float, width_px: int = 800):
        self.map_w = float(map_w_m)
        self.map_h = float(map_h_m)
        self.width = int(width_px)
        self.height = int(round(self.width * (self.map_h / max(1e-6, self.map_w))))
        self._pygame = None
        self._screen = None
        self._clock = None
        self._bg_cache = None
        self._last_grid_hash = None
        self._init_pygame()

    def _init_pygame(self) -> None:
        # Set up headless rendering support
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        # For headless environments, also set video driver if not already set
        if "DISPLAY" not in os.environ or os.environ.get("DISPLAY") == "":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self._pygame = pygame
        if not pygame.get_init():
            pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AMR Bottleneck Navigation")
        self._clock = pygame.time.Clock()

    def _w2s(self, x: float, y: float) -> Tuple[int, int]:
        sx = int((x / self.map_w) * self.width)
        sy = int(self.height - (y / self.map_h) * self.height)
        return sx, sy

    def draw(
        self,
        grid: np.ndarray,
        path: np.ndarray,
        previews: np.ndarray,
        pose: Tuple[float, float, float],
        lidar_ranges: np.ndarray | None,
        lidar_rel_angles: np.ndarray | None,
        goal_xy: Tuple[float, float],
        robot_radius_m: float,
        fps_limit: int = 30,
        capture: bool = False,
    ) -> np.ndarray | None:
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Background from grid (cached to avoid recomputation)
        grid_hash = hash(grid.tobytes())
        if self._bg_cache is None or self._last_grid_hash != grid_hash:
            free = np.array([255, 255, 255], dtype=np.uint8)
            occ = np.array([60, 60, 60], dtype=np.uint8)
            rgb_small = np.where(np.flipud(grid).T[..., None], occ, free)
            surf_small = pygame.surfarray.make_surface(rgb_small)
            self._bg_cache = pygame.transform.smoothscale(surf_small, (self.width, self.height))
            self._last_grid_hash = grid_hash
        self._screen.blit(self._bg_cache, (0, 0))

        if path is not None and path.shape[0] >= 2:
            pts = [self._w2s(float(px), float(py)) for px, py in path]
            pygame.draw.lines(self._screen, (31, 119, 180), False, pts, 2)

        if previews is not None and previews.size > 0:
            for px, py in previews:
                sx, sy = self._w2s(float(px), float(py))
                pygame.draw.circle(self._screen, (31, 119, 180), (sx, sy), 3)

        x, y, th = pose
        sx, sy = self._w2s(x, y)
        px_per_m = self.width / self.map_w
        rr = int(max(2, round(robot_radius_m * px_per_m)))
        pygame.draw.circle(self._screen, (44, 160, 44), (sx, sy), rr)
        hx = x + max(0.4, robot_radius_m * 1.8) * math.cos(th)
        hy = y + max(0.4, robot_radius_m * 1.8) * math.sin(th)
        hx_s, hy_s = self._w2s(hx, hy)
        pygame.draw.line(self._screen, (0, 0, 0), (sx, sy), (hx_s, hy_s), 2)

        if lidar_ranges is not None and lidar_rel_angles is not None:
            for a_rel, d in zip(lidar_rel_angles, lidar_ranges):
                a = th + float(a_rel)
                ex = x + float(d) * math.cos(a)
                ey = y + float(d) * math.sin(a)
                ex_s, ey_s = self._w2s(ex, ey)
                pygame.draw.line(self._screen, (220, 20, 60), (sx, sy), (ex_s, ey_s), 1)

        gx, gy = goal_xy
        gx_s, gy_s = self._w2s(gx, gy)
        pygame.draw.circle(self._screen, (255, 215, 0), (gx_s, gy_s), 6)
        pygame.draw.circle(self._screen, (0, 0, 0), (gx_s, gy_s), 6, 1)

        pygame.display.flip()
        self._clock.tick(fps_limit)

        if capture:
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))
        return None

    def close(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:
                pass
            try:
                self._pygame.quit()
            except Exception:
                pass
        self._pygame = None
        self._screen = None
        self._clock = None

