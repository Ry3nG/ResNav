from __future__ import annotations

import argparse
import os
import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO

from me5418_nav.envs import UnicycleNavEnv
from me5418_nav.constants import DT


def record_episode(env: UnicycleNavEnv, model: PPO, out_path: str, fps: int = 12, stride: int = 2, scale: float = 1.0, scenario_kwargs: Dict[str, Any] | None = None) -> None:
    obs, info = env.reset(options={"scenario": "blockage", "kwargs": (scenario_kwargs or {})})
    frames: list[np.ndarray] = []
    t = 0
    done = False
    while not done and t < 6000:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if t % max(1, int(stride)) == 0:
            frame = env.render("rgb_array")
            if frame is not None:
                if scale and scale != 1.0:
                    try:
                        import cv2  # type: ignore
                        h, w = frame.shape[:2]
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass
                frames.append(frame)
        done = bool(term or trunc)
        t += 1

    # Write GIF via imageio or PIL
    if not frames:
        print("[WARN] No frames captured; skipping GIF save.")
        return
    try:
        import imageio.v2 as imageio  # type: ignore
        imageio.mimsave(out_path, frames, fps=max(1, int(fps)))
        print(f"[INFO] Saved GIF via imageio: {out_path}")
        return
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        duration_ms = int(1000.0 / max(1, int(fps)))
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], optimize=False, duration=duration_ms, loop=0)
        print(f"[INFO] Saved GIF via PIL: {out_path}")
    except Exception:
        print("[WARN] Unable to save GIF (imageio and PIL not available).")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO model and optionally record a GIF")
    parser.add_argument("--model", type=str, required=True, help="Path to model zip (best_model.zip or final_model.zip)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation random seed")
    parser.add_argument("--num-pallets", type=int, default=1, help="Blockage scenario pallets")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path (records a best/typical episode)")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--gif-scale", type=float, default=1.0)
    parser.add_argument("--show", action="store_true", help="Display evaluation window (default: headless)")
    args = parser.parse_args()

    # Build env with appropriate rendering mode
    render_mode = "human" if args.show else None
    env = UnicycleNavEnv({"render_mode": render_mode})

    # Load model
    model = PPO.load(args.model, env=None, device="cpu")

    # Evaluation loop
    results = []
    success = 0
    collisions = 0
    timeouts = 0
    total_steps = 0

    for ep in range(int(args.episodes)):
        obs, info = env.reset(options={"scenario": "blockage", "kwargs": {"num_pallets": int(args.num_pallets)}})
        steps = 0
        goal = False
        collision = False
        timeout = False
        s_end = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            goal = bool(info.get("goal", False))
            collision = bool(info.get("collision", False))
            timeout = bool(info.get("timeout", False))
            s_ptr = float(info.get("s_ptr", 0.0))
            s_end = s_ptr
            if term or trunc:
                break
        success += int(goal)
        collisions += int(collision)
        timeouts += int(timeout and not goal and not collision)
        total_steps += steps
        results.append({
            "episode": ep,
            "steps": steps,
            "time_s": steps * DT,
            "goal": goal,
            "collision": collision,
            "timeout": timeout,
            "s_ptr": s_end,
        })

    # Summary
    succ_rate = success / len(results) if results else 0.0
    coll_rate = collisions / len(results) if results else 0.0
    timeout_rate = timeouts / len(results) if results else 0.0
    mean_time = np.mean([r["time_s"] for r in results]) if results else 0.0
    print("[EVAL] episodes=", len(results))
    print("[EVAL] success_rate=", succ_rate)
    print("[EVAL] collision_rate=", coll_rate)
    print("[EVAL] timeout_rate=", timeout_rate)
    print("[EVAL] mean_time_s=", mean_time)

    # Save results
    outdir = os.path.join("runs", "eval", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "episodes": len(results),
            "success_rate": succ_rate,
            "collision_rate": coll_rate,
            "timeout_rate": timeout_rate,
            "mean_time_s": float(mean_time),
            "details": results,
        }, f, indent=2)

    # Optional GIF: re-run a best/typical episode
    if args.gif:
        # pick best: among successes, minimal time; else shortest time overall
        chosen = None
        succs = [r for r in results if r["goal"]]
        if succs:
            chosen = sorted(succs, key=lambda r: r["time_s"])[:1][0]
        else:
            chosen = sorted(results, key=lambda r: r["time_s"])[:1][0]
        print(f"[EVAL] Recording GIF for episode template: {chosen}")
        record_episode(env, model, args.gif, fps=args.gif_fps, stride=args.gif_stride, scale=args.gif_scale, scenario_kwargs={"num_pallets": int(args.num_pallets)})

    env.close()


if __name__ == "__main__":
    main()
