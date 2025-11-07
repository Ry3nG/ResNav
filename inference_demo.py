#!/usr/bin/env python3
"""Tiny wrapper around `training/rollout.py` for quick inference demos."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "training/rollout.py", "--model", args.path]

    if args.vecnorm:
        cmd.extend(["--vecnorm", args.vecnorm])
    if args.render:
        cmd.append("--render")
    if args.record:
        cmd.extend(["--record", args.record])
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd.append("--deterministic")
    if args.env_cfg:
        cmd.extend(["--env_cfg", args.env_cfg])
    if args.robot_cfg:
        cmd.extend(["--robot_cfg", args.robot_cfg])
    if args.reward_cfg:
        cmd.extend(["--reward_cfg", args.reward_cfg])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference using a saved ResNav SAC policy."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a model artifact or run directory (e.g. runs/demo_1031/best)",
    )
    parser.add_argument(
        "--vecnorm",
        default="",
        help="Optional VecNormalize stats path; autodetected when omitted",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Maximum rollout steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display realtime rendering during rollout",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Path to save MP4 video; defaults to project_root/demo.mp4",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions",
    )
    parser.add_argument(
        "--env-cfg",
        dest="env_cfg",
        default="",
        help="Override environment config path",
    )
    parser.add_argument(
        "--robot-cfg",
        dest="robot_cfg",
        default="",
        help="Override robot config path",
    )
    parser.add_argument(
        "--reward-cfg",
        dest="reward_cfg",
        default="",
        help="Override reward config path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it",
    )

    args = parser.parse_args()

    path_obj = Path(args.path)
    if not path_obj.exists():
        parser.error(f"Path does not exist: {args.path}")

    repo_root = Path(__file__).resolve().parent

    record_arg = args.record
    if record_arg is None:
        args.record = str(repo_root / "demo.mp4")
    elif isinstance(record_arg, str) and record_arg.strip().lower() in {"", "none"}:
        args.record = ""
    else:
        args.record = str(Path(record_arg))

    command = build_command(args)

    print("[inference_demo] Launching:")
    print(" ".join(command))

    if args.dry_run:
        return

    env = os.environ.copy()

    try:
        subprocess.run(command, cwd=repo_root, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[inference_demo] Rollout failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
