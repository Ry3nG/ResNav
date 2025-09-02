from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def draw_env(env, ax, status: dict | None = None):
    ax.clear()
    extent = None
    grid_img = None
    try:
        extent = list(env.grid.workspace)
        grid_img = env.grid.grid
    except Exception:
        grid_img = getattr(env.grid, "grid", None)
    if grid_img is not None:
        if extent is not None:
            ax.imshow(grid_img, origin="lower", cmap="Greys", extent=extent)
        else:
            ax.imshow(grid_img, origin="lower", cmap="Greys")

    # draw path waypoints if available
    wp = getattr(env, "path_waypoints", None)
    if wp is not None and isinstance(wp, np.ndarray) and wp.shape[1] == 2:
        ax.plot(wp[:, 0], wp[:, 1], "c-", linewidth=1.5, alpha=0.9, label="path")
        k = max(1, wp.shape[0] // 30)
        ax.plot(wp[::k, 0], wp[::k, 1], "co", markersize=2, alpha=0.7)

    goal = getattr(env, "goal_xy", None)
    if goal is not None:
        ax.plot(goal[0], goal[1], "gx", markersize=8, markeredgewidth=2, label="goal")

    pose = env.robot.as_pose()
    x, y, th = pose
    ranges, endpoints = env.lidar.cast(pose, env.grid)
    ax.plot(x, y, "bo")
    ax.arrow(x, y, 0.3 * np.cos(th), 0.3 * np.sin(th), head_width=0.1, color="b")
    for i in range(env.lidar.n_beams):
        ax.plot([x, endpoints[i, 0]], [y, endpoints[i, 1]], "r-", linewidth=0.8, alpha=0.6)

    ax.set_aspect("equal")
    ax.set_title("Env: LiDAR (red), robot (blue), path (cyan), goal (green x)")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Simple overlay of key parameters in top-left
    if status:
        keys = [
            "mode",
            "follow_side",
            "min_range",
            "rmin_front",
            "d_to_wall",
            "clear_err",
            "heading_err",
            "v_cmd",
            "w_cmd",
            "step",
            "wp_idx",
            "lookahead",
            "v_nominal",
            "k_heading",
            "repulse_dist",
            "repulse_gain",
            "follow_clearance",
            "follow_speed",
            "tangential_gain",
        ]
        lines = []
        for k in keys:
            if k in status and status[k] is not None:
                v = status[k]
                if isinstance(v, float):
                    v = f"{v:.2f}"
                lines.append(f"{k}: {v}")
        if lines:
            text = "\n".join(lines)
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="k",
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
            )
