#!/usr/bin/env python3
"""Tiny wrapper that launches the standard SAC training command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN = {
    "env": "omcf",
    "robot": "allow_reverse",
    "reward": "lower_w_path",
    "algo": "sac",
    "network": "lidar_cnn",
    "wandb": "default",
    "run.vec_envs": "20",
    "run.total_timesteps": "200000",
    "run.seed": "20030413",
}


def build_command(overrides: dict[str, str]) -> list[str]:
    """Construct the Hydra override command for `training/train_sac.py`."""

    cmd = [sys.executable, "training/train_sac.py"]
    for key, value in overrides.items():
        cmd.append(f"{key}={value}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the standard ResNav SAC training run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional Hydra-style override (repeatable).",
    )
    args = parser.parse_args()

    overrides = dict(DEFAULT_RUN)
    for item in args.override:
        if "=" not in item:
            parser.error(f"Invalid override '{item}'; expected KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    command = build_command(overrides)

    print("[train_demo] Launching:")
    print(" ".join(command))

    if args.dry_run:
        return

    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()

    try:
        subprocess.run(command, cwd=repo_root, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[train_demo] Training failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
