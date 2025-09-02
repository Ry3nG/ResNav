from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class _View:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    scale: float  # pixels per meter
    w: int
    h: int
    ox: int  # x offset (letterbox)
    oy: int  # y offset (letterbox)


class PygameRenderer:
    """
    Lightweight 2D renderer using pygame.

    - Draws occupancy grid, path, goal, robot, and LiDAR rays.
    - Provides a HUD from an optional status dict (e.g., controller debug).
    - Keeps a single window per env instance.
    """

    def __init__(self, window_size: Tuple[int, int] = (800, 800)) -> None:
        self._pg = None  # lazy import & init
        self._screen = None
        self._font = None
        self._clock = None
        self._surf_grid = None
        self._view: Optional[_View] = None
        self._win_size = window_size

    # --------- Public API ----------
    def draw(self, env) -> None:
        pg = self._ensure_init()
        if self._screen is None:
            self._screen = pg.display.set_mode(self._win_size)
            pg.display.set_caption("ME5418 Nav (pygame)")
            try:
                self._font = pg.font.SysFont("monospace", 14)
            except Exception:
                self._font = None
            self._clock = pg.time.Clock()

        # Handle close events to allow quitting the window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return

        self._screen.fill((245, 245, 245))

        # Compute view and (re)build grid surface if needed
        workspace, grid_img, res = self._extract_grid(env)
        if workspace is None and grid_img is not None:
            H, W = grid_img.shape[:2]
            workspace = (0.0, W * res, 0.0, H * res)
        if workspace is None:
            # fallback fixed view
            workspace = (0.0, 12.0, 0.0, 12.0)

        if self._view is None:
            self._view = self._make_view(workspace, self._win_size)

        if self._surf_grid is None and grid_img is not None:
            self._surf_grid = self._make_grid_surface(pg, grid_img, workspace)

        # draw occupancy grid (preserve crisp cells: nearest-neighbor + letterbox)
        if self._surf_grid is not None:
            self._blit_grid(pg, self._surf_grid)

        # draw path
        wp = getattr(env, "path_waypoints", None)
        if isinstance(wp, np.ndarray) and wp.shape[1] == 2 and wp.shape[0] >= 2:
            pts = [self._w2s(tuple(w)) for w in wp]
            pg.draw.lines(self._screen, (0, 180, 180), False, pts, 2)
            k = max(1, wp.shape[0] // 30)
            for p in wp[::k]:
                pg.draw.circle(self._screen, (0, 180, 180), self._w2s(tuple(p)), 2)

        # goal
        goal = getattr(env, "goal_xy", None)
        if goal is not None:
            gx, gy = self._w2s((float(goal[0]), float(goal[1])))
            pg.draw.line(
                self._screen, (0, 160, 0), (gx - 6, gy - 6), (gx + 6, gy + 6), 2
            )
            pg.draw.line(
                self._screen, (0, 160, 0), (gx - 6, gy + 6), (gx + 6, gy - 6), 2
            )

        # robot + lidar
        pose = env.robot.as_pose()
        x, y, th = pose
        px, py = self._w2s((x, y))

        # Use actual robot radius from config for realistic rendering
        robot_radius = getattr(env.cfg, "robot_radius", 0.25)
        robot_radius_px = int(robot_radius * self._view.scale)
        pg.draw.circle(self._screen, (40, 90, 240), (px, py), max(2, robot_radius_px))

        # Heading indicator proportional to robot size
        heading_length = robot_radius * 0.8  # slightly smaller than radius
        hx, hy = self._w2s(
            (x + heading_length * np.cos(th), y + heading_length * np.sin(th))
        )
        pg.draw.line(self._screen, (40, 90, 240), (px, py), (hx, hy), 3)
        try:
            ranges, endpoints = env.lidar.cast(pose, env.grid)
            for i in range(env.lidar.n_beams):
                ex, ey = self._w2s((endpoints[i, 0], endpoints[i, 1]))
                pg.draw.line(self._screen, (220, 60, 60), (px, py), (ex, ey), 1)
        except Exception:
            pass

        # HUD
        status = getattr(env, "_debug_status", None)
        if status and self._font is not None:
            lines = []
            keys = [
                "mode",
                "follow_side",
                "min_range",
                "d_to_wall",
                "clear_err",
                "heading_err",
                "v_cmd",
                "w_cmd",
                "step",
                "wp_idx",
            ]
            for k in keys:
                if k in status and status[k] is not None:
                    v = status[k]
                    if isinstance(v, float):
                        v = f"{v:.2f}"
                    lines.append(f"{k}: {v}")
            y0 = 8
            for line in lines[:18]:
                surf = self._font.render(line, True, (20, 20, 20))
                self._screen.blit(surf, (8, y0))
                y0 += surf.get_height() + 2

        pg.display.flip()
        if self._clock is not None:
            dt = float(getattr(env.cfg, "dt", 0.05))
            fps = max(1, int(round(1.0 / max(1e-3, dt))))
            self._clock.tick(fps)

    def close(self) -> None:
        if self._pg is None:
            return
        try:
            self._pg.display.quit()
            self._pg.quit()
        except Exception:
            pass
        self._pg = None
        self._screen = None
        self._font = None
        self._clock = None
        self._surf_grid = None
        self._view = None

    # --------- Internals ----------
    def _ensure_init(self):
        if self._pg is not None:
            return self._pg
        import pygame as pg  # lazy import to avoid hard dep at import time

        pg.init()
        self._pg = pg
        return pg

    def _extract_grid(self, env):
        grid = getattr(env, "grid", None)
        res = getattr(env.cfg, "res", 0.1)
        if grid is None:
            return None, None, res
        workspace = None
        grid_img = None
        try:
            workspace = tuple(grid.workspace)
            grid_img = grid.grid.astype(np.uint8)
            res = getattr(grid, "_cellsize", res)
        except Exception:
            grid_img = getattr(grid, "grid", None)
        return workspace, grid_img, res

    def _make_view(self, ws, win_size) -> _View:
        xmin, xmax, ymin, ymax = ws
        W, H = win_size
        sx = W / max(1e-6, (xmax - xmin))
        sy = H / max(1e-6, (ymax - ymin))
        scale = min(sx, sy)
        cw = int(round((xmax - xmin) * scale))
        ch = int(round((ymax - ymin) * scale))
        ox = (W - cw) // 2
        oy = (H - ch) // 2
        return _View(xmin, xmax, ymin, ymax, scale, W, H, ox, oy)

    def _w2s(self, p: Tuple[float, float]) -> Tuple[int, int]:
        # world (meters) -> screen (pixels), y axis up in world, down on screen
        assert self._view is not None
        x, y = float(p[0]), float(p[1])
        vx = int(self._view.ox + (x - self._view.xmin) * self._view.scale)
        vy = int(self._view.oy + (self._view.ymax - y) * self._view.scale)
        return vx, vy

    def _make_grid_surface(self, pg, grid_img: np.ndarray, ws) -> "pg.Surface":
        # bool grid -> grayscale surface; True=occupied (dark), False=free (light)
        H, W = grid_img.shape[:2]
        img = np.empty((H, W, 3), dtype=np.uint8)
        occ = grid_img.astype(bool)
        img[:, :, :] = 230  # free
        img[occ] = (60, 60, 60)
        # flip vertically to match world y-up
        img = np.flipud(img)
        # pygame expects (W,H,3)
        surf = pg.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        return surf

    def _blit_grid(self, pg, grid_surf):
        # scale with nearest-neighbor to keep sharp edges; letterbox to preserve aspect
        assert self._view is not None
        xmin, xmax, ymin, ymax = (
            self._view.xmin,
            self._view.xmax,
            self._view.ymin,
            self._view.ymax,
        )
        dest_w = max(1, int(round((xmax - xmin) * self._view.scale)))
        dest_h = max(1, int(round((ymax - ymin) * self._view.scale)))
        scaled = pg.transform.scale(grid_surf, (dest_w, dest_h))
        self._screen.blit(scaled, (self._view.ox, self._view.oy))
