from __future__ import annotations

import os
import argparse
import numpy as np

from me5418_nav.envs import UnicycleNavEnv, EnvConfig


def save_video(frames: list[np.ndarray], out_path: str, fps: int):
    try:
        import imageio
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Saved video: {out_path}")
    except Exception as e:
        # fallback: save frames as PNGs in a folder
        base, _ = os.path.splitext(out_path)
        folder = base + "_frames"
        os.makedirs(folder, exist_ok=True)
        for i, frame in enumerate(frames):
            try:
                from imageio.v2 import imwrite
                imwrite(os.path.join(folder, f"frame_{i:05d}.png"), frame)
            except Exception:
                # last resort: numpy save
                np.save(os.path.join(folder, f"frame_{i:05d}.npy"), frame)
        print(f"Saved frames to folder: {folder} (imageio not available)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--out', type=str, default='episode.gif')
    args = parser.parse_args()

    env = UnicycleNavEnv(EnvConfig(), render_mode='rgb_array')
    obs, info = env.reset()
    frames: list[np.ndarray] = []
    for t in range(args.steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    # capture final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    fps = max(1, int(1.0 / env.cfg.dt))
    save_video(frames, args.out, fps)


if __name__ == "__main__":
    main()

