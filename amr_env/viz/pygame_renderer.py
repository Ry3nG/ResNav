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

from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
import pygame


@dataclass
class Colors:
    background: tuple[int, int, int] = (20, 20, 20)
    raw_grid: tuple[int, int, int] = (60, 60, 60)
    inflated_grid: tuple[int, int, int] = (80, 20, 20)
    robot: tuple[int, int, int] = (50, 180, 255)
    robot_heading: tuple[int, int, int] = (0, 255, 0)
    lidar_ray: tuple[int, int, int] = (255, 255, 0)
    lidar_hit: tuple[int, int, int] = (255, 0, 0)
    path: tuple[int, int, int] = (0, 200, 200)
    projection: tuple[int, int, int] = (200, 100, 255)
    lookahead: tuple[int, int, int] = (100, 255, 100)
    action_track: tuple[int, int, int] = (0, 120, 255)
    action_delta: tuple[int, int, int] = (255, 120, 0)
    action_final: tuple[int, int, int] = (255, 0, 255)
    text: tuple[int, int, int] = (255, 255, 255)
    # Dynamic mover colors
    mover_lateral: tuple[int, int, int] = (255, 69, 0)  # Red-Orange (counterflow danger)
    mover_longitudinal: tuple[int, int, int] = (138, 43, 226)  # Blue-Violet (merge from side)


@dataclass
class VizConfig:
    size_px: tuple[int, int] = (800, 800)
    show_inflated: bool = True
    show_lidar: bool = True
    show_actions: bool = True
    fps: int = 20
    colors: Colors = field(default_factory=Colors)


class Renderer:
    def __init__(
        self,
        map_size_m: tuple[float, float],
        resolution_m: float,
        viz_cfg: Optional[VizConfig] = None,
        display: bool = True,
    ) -> None:
        self.viz = viz_cfg or VizConfig()
        self.colors = self.viz.colors
        self.map_w, self.map_h = float(map_size_m[0]), float(map_size_m[1])
        self.res = float(resolution_m)
        self.width, self.height = self.viz.size_px
        self.scale = min(self.width / self.map_w, self.height / self.map_h)
        self.display = bool(display)

        if not self.display:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        if self.display:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))
        pygame.display.set_caption("AMR Residual Nav")
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 14)

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        # Flip Y for screen coords (origin top-left)
        sx = int(x * self.scale)
        sy = int(self.height - y * self.scale)
        # Clamp to window bounds
        sx = max(0, min(self.width - 1, sx))
        sy = max(0, min(self.height - 1, sy))
        return sx, sy

    def draw_grid(self, grid: np.ndarray, color: tuple[int, int, int]) -> None:
        H, W = grid.shape
        cell_size = self.res * self.scale
        occupied = np.where(grid)
        for i, j in zip(occupied[0], occupied[1]):
            x0 = j * cell_size
            y0 = self.height - (i + 1) * cell_size
            rect = pygame.Rect(int(x0), int(y0), int(cell_size + 1), int(cell_size + 1))
            pygame.draw.rect(self.screen, color, rect)

    def draw_robot(self, pose: tuple[float, float, float], radius_m: float) -> None:
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)
        r_px = int(radius_m * self.scale)
        pygame.draw.circle(self.screen, self.colors.robot, (sx, sy), r_px, width=2)
        # Heading arrow
        hx = x + radius_m * 1.5 * np.cos(th)
        hy = y + radius_m * 1.5 * np.sin(th)
        hpx, hpy = self.world_to_screen(hx, hy)
        pygame.draw.line(
            self.screen, self.colors.robot_heading, (sx, sy), (hpx, hpy), width=2
        )

    def draw_lidar(
        self,
        pose: tuple[float, float, float],
        lidar_data: tuple[np.ndarray, int, float, float],
    ) -> None:
        if not self.viz.show_lidar:
            return
        ranges, beams, fov_rad, _ = lidar_data
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)

        # Vectorized computation
        offsets = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, beams)
        angles = th + offsets
        end_x = x + ranges * np.cos(angles)
        end_y = y + ranges * np.sin(angles)

        for k in range(beams):
            ex_px, ey_px = self.world_to_screen(end_x[k], end_y[k])
            pygame.draw.line(
                self.screen, self.colors.lidar_ray, (sx, sy), (ex_px, ey_px), width=1
            )
            pygame.draw.circle(self.screen, self.colors.lidar_hit, (ex_px, ey_px), 2)

    def draw_path(
        self,
        waypoints: np.ndarray,
        proj: Optional[tuple[float, float]] = None,
        lookahead: Optional[tuple[float, float]] = None,
    ) -> None:
        if len(waypoints) >= 2:
            pts = [self.world_to_screen(float(x), float(y)) for x, y in waypoints]
            pygame.draw.lines(self.screen, self.colors.path, False, pts, width=2)
        if proj is not None:
            px, py = self.world_to_screen(proj[0], proj[1])
            pygame.draw.circle(self.screen, self.colors.projection, (px, py), 4)
        if lookahead is not None:
            lx, ly = self.world_to_screen(lookahead[0], lookahead[1])
            pygame.draw.circle(self.screen, self.colors.lookahead, (lx, ly), 4)

    def draw_actions(
        self,
        pose: tuple[float, float, float],
        actions_data: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        scale: float = 0.5,
    ) -> None:
        if not self.viz.show_actions:
            return
        u_track, du, u_final = actions_data
        x, y, th = pose
        sx, sy = self.world_to_screen(x, y)

        actions = [u_track, du, u_final]
        colors = [
            self.colors.action_track,
            self.colors.action_delta,
            self.colors.action_final,
        ]

        for (v, _), color in zip(actions, colors):
            ex = x + scale * v * np.cos(th)
            ey = y + scale * v * np.sin(th)
            ex_px, ey_px = self.world_to_screen(ex, ey)
            pygame.draw.line(self.screen, color, (sx, sy), (ex_px, ey_px), width=3)

    def draw_movers(self, movers: list) -> None:
        """Draw dynamic movers with type-specific colors."""
        for mover in movers:
            sx, sy = self.world_to_screen(mover.x, mover.y)
            r_px = int(mover.radius_m * self.scale)

            # Select color based on mover type
            color = (
                self.colors.mover_lateral
                if mover.mover_type == "lateral"
                else self.colors.mover_longitudinal
            )

            # Draw filled circle for mover body
            pygame.draw.circle(self.screen, color, (sx, sy), r_px)

            # Draw velocity vector (optional, for debugging)
            if hasattr(mover, 'vx') and hasattr(mover, 'vy'):
                vel_scale = 0.8  # Scale factor for velocity visualization
                vx_end = mover.x + mover.vx * vel_scale
                vy_end = mover.y + mover.vy * vel_scale
                vx_px, vy_px = self.world_to_screen(vx_end, vy_end)
                # Draw thin white line for velocity direction
                pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (vx_px, vy_px), width=1)

    def draw_hud(self, text_lines: dict[str, float], y0: int = 10) -> None:
        x, y = 10, y0
        for k, v in text_lines.items():
            surf = self.font.render(f"{k}: {v:.3f}", True, self.colors.text)
            self.screen.blit(surf, (x, y))
            y += 18

    def render_frame(
        self,
        raw_grid: np.ndarray,
        inflated_grid: Optional[np.ndarray],
        pose: tuple[float, float, float],
        radius_m: float,
        lidar: Optional[tuple[np.ndarray, int, float, float]] = None,
        path: Optional[np.ndarray] = None,
        proj: Optional[tuple[float, float]] = None,
        lookahead: Optional[tuple[float, float]] = None,
        actions: Optional[
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        ] = None,
        hud: Optional[dict[str, float]] = None,
        movers: Optional[list] = None,
    ) -> "pygame.Surface":
        self.screen.fill(self.colors.background)

        # Draw components in order
        if raw_grid is not None:
            self.draw_grid(raw_grid, self.colors.raw_grid)
        if inflated_grid is not None and self.viz.show_inflated:
            self.draw_grid(inflated_grid, self.colors.inflated_grid)
        if path is not None:
            self.draw_path(path, proj, lookahead)

        # Draw dynamic movers (before robot so robot appears on top)
        if movers is not None:
            self.draw_movers(movers)

        self.draw_robot(pose, radius_m)

        if lidar is not None:
            self.draw_lidar(pose, lidar)
        if actions is not None:
            self.draw_actions(pose, actions)
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self) -> None:
        if self.display and pygame is not None:
            pygame.display.quit()
        if pygame is not None:
            pygame.quit()
