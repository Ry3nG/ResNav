"""Pygame renderer for AMR residual navigation.

Renders:
- Raw and inflated occupancy grids
- Robot pose and radius
- LiDAR rays and hit points
- Global path, closest projection, lookahead target
- Actions: u_track, Δu, u_final
- HUD with reward components

Supports windowed (interactive) and headless modes. Returns frames for recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - renderer optional in headless CI
    pygame = None


@dataclass
class VizConfig:
    size_px: Tuple[int, int] = (800, 800)
    show_inflated: bool = True
    show_lidar: bool = True
    show_actions: bool = True
    fps: int = 20


class Renderer:
    def __init__(
        self,
        map_size_m: Tuple[float, float],
        resolution_m: float,
        viz_cfg: Optional[VizConfig] = None,
        display: bool = True,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame not available; install pygame to use the renderer")
        self.viz = viz_cfg or VizConfig()
        self.map_w, self.map_h = float(map_size_m[0]), float(map_size_m[1])
        self.res = float(resolution_m)
        self.width, self.height = self.viz.size_px
        self.scale = min(self.width / self.map_w, self.height / self.map_h)
        self.display = bool(display)

        pygame.init()
        if self.display:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))
        pygame.display.set_caption("AMR Residual Nav")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 14)

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        # Flip Y for screen coords (origin top-left)
        sx = int(x * self.scale)
        sy = int(self.height - y * self.scale)
        # Clamp to window bounds
        sx = max(0, min(self.width - 1, sx))
        sy = max(0, min(self.height - 1, sy))
        return sx, sy

    def draw_grid(self, grid: np.ndarray, color=(60, 60, 60)) -> None:
        H, W = grid.shape
        cell = self.res * self.scale
        for i in range(H):
            for j in range(W):
                if grid[i, j]:
                    x0 = j * cell
                    y0 = self.height - (i + 1) * cell
                    rect = pygame.Rect(int(x0), int(y0), int(cell + 1), int(cell + 1))
                    pygame.draw.rect(self.screen, color, rect)

    def draw_robot(self, pose: Tuple[float, float, float], radius_m: float) -> None:
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)
        r_px = int(radius_m * self.scale)
        pygame.draw.circle(self.screen, (50, 180, 255), (sx, sy), r_px, width=2)
        # Heading arrow
        hx = x + radius_m * 1.5 * np.cos(th)
        hy = y + radius_m * 1.5 * np.sin(th)
        hpx, hpy = self.world_to_screen(hx, hy)
        pygame.draw.line(self.screen, (0, 255, 0), (sx, sy), (hpx, hpy), width=2)

    def draw_lidar(self, pose: Tuple[float, float, float], ranges: np.ndarray, beams: int, fov_rad: float, max_range: float) -> None:
        if not self.viz.show_lidar:
            return
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)
        # Inclusive endpoints
        offsets = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, beams)
        for k in range(beams):
            ang = th + float(offsets[k])
            d = float(ranges[k])
            ex = x + d * np.cos(ang)
            ey = y + d * np.sin(ang)
            ex_px, ey_px = self.world_to_screen(ex, ey)
            pygame.draw.line(self.screen, (255, 255, 0), (sx, sy), (ex_px, ey_px), width=1)
            pygame.draw.circle(self.screen, (255, 0, 0), (ex_px, ey_px), 2)

    def draw_path(self, waypoints: np.ndarray, proj: Optional[Tuple[float, float]] = None, lookahead: Optional[Tuple[float, float]] = None) -> None:
        pts = [(self.world_to_screen(float(x), float(y))) for x, y in waypoints]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, (0, 200, 200), False, pts, width=2)
        if proj is not None:
            px, py = self.world_to_screen(proj[0], proj[1])
            pygame.draw.circle(self.screen, (200, 100, 255), (px, py), 4)
        if lookahead is not None:
            lx, ly = self.world_to_screen(lookahead[0], lookahead[1])
            pygame.draw.circle(self.screen, (100, 255, 100), (lx, ly), 4)

    def draw_actions(self, pose: Tuple[float, float, float], u_track: Tuple[float, float], du: Tuple[float, float], u_final: Tuple[float, float], scale: float = 0.5) -> None:
        if not self.viz.show_actions:
            return
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)
        def vec_to_end(v: float, w: float, color):
            # Represent (v,w) as arrow in heading direction scaled by v; w shown via color only.
            ex = x + scale * v * np.cos(th)
            ey = y + scale * v * np.sin(th)
            ex_px, ey_px = self.world_to_screen(ex, ey)
            pygame.draw.line(self.screen, color, (sx, sy), (ex_px, ey_px), width=3)
        vec_to_end(*u_track, (0, 120, 255))
        vec_to_end(*du, (255, 120, 0))
        vec_to_end(*u_final, (255, 0, 255))

    def draw_hud(self, text_lines: Dict[str, float], y0: int = 10) -> None:
        x = 10
        y = y0
        for k, v in text_lines.items():
            surf = self.font.render(f"{k}: {v:.3f}", True, (255, 255, 255))
            self.screen.blit(surf, (x, y))
            y += 18

    def render_frame(
        self,
        raw_grid: np.ndarray,
        inflated_grid: Optional[np.ndarray],
        pose: Tuple[float, float, float],
        radius_m: float,
        lidar: Optional[Tuple[np.ndarray, int, float, float]] = None,
        path: Optional[np.ndarray] = None,
        proj: Optional[Tuple[float, float]] = None,
        lookahead: Optional[Tuple[float, float]] = None,
        actions: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
        hud: Optional[Dict[str, float]] = None,
    ) -> "pygame.Surface":
        # Clear
        self.screen.fill((20, 20, 20))
        # Grids
        if raw_grid is not None:
            self.draw_grid(raw_grid, color=(60, 60, 60))
        if inflated_grid is not None and self.viz.show_inflated:
            self.draw_grid(inflated_grid, color=(80, 20, 20))
        # Path
        if path is not None:
            self.draw_path(path, proj=proj, lookahead=lookahead)
        # Robot
        self.draw_robot(pose, radius_m)
        # LiDAR
        if lidar is not None:
            ranges, beams, fov_rad, max_range = lidar
            self.draw_lidar(pose, ranges, beams, fov_rad, max_range)
        # Actions
        if actions is not None:
            self.draw_actions(pose, *actions)
        # HUD
        if hud:
            self.draw_hud(hud)
        if self.display:
            pygame.display.flip()
            self.clock.tick(self.viz.fps)
        return self.screen

    def poll_events(self) -> bool:
        """Return False if a quit event is received."""
        if not self.display:
            return True
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        except Exception:
            return True
        return True

    def close(self) -> None:
        try:
            if self.display and pygame is not None:
                pygame.display.quit()
            if pygame is not None:
                pygame.quit()
        except Exception:
            pass
