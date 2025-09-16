"""Minimal interactive launcher for AMR workflows.

Provides a lightweight prompt to:
- Train with Hydra overrides (select config groups and run params)
- Render/record PPO models (auto-detect config + vecnorm)

No third-party prompt libraries required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from amr_env.utils import (
    detect_run_root,
    load_config_any,
    load_config_dict,
    load_resolved_run_config,
    read_run_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_group_options(group: str) -> List[str]:
    d = REPO_ROOT / "configs" / group
    if not d.exists() or not d.is_dir():
        return []
    return [p.stem for p in sorted(d.glob("*.yaml"))]


def prompt_choice(title: str, options: List[str], default_idx: int = 0) -> str:
    if not options:
        return ""
    print(f"\n{title}")
    for i, opt in enumerate(options):
        mark = "*" if i == default_idx else " "
        print(f"  [{i}] {mark} {opt}")
    while True:
        ans = input(f"Select [0-{len(options)-1}] (default {default_idx}): ").strip()
        if ans == "":
            return options[default_idx]
        if ans.isdigit():
            idx = int(ans)
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid choice. Try again.")


def prompt_text(title: str, default: str = "") -> str:
    ans = input(f"{title} (default '{default}'): ").strip()
    return ans if ans else default
def load_config_groups(run_dir: str) -> Dict[str, str]:
    """Parse config group overrides used for a run.

    Returns an empty dict if no overrides are present or format is unexpected.
    """
    return read_run_overrides(run_dir)


def load_run_config(run_dir: str) -> Dict[str, Any]:
    """Load resolved config for a run.

    Prefers `resolved.yaml` (if exported), otherwise falls back to Hydra config.
    Returns an empty dict if neither exists.
    """
    return load_resolved_run_config(run_dir)


def display_model_config(model_path: str) -> None:
    """Display configuration information for a model."""
    run_dir = detect_run_root(model_path)

    print(f"[INFO] Detected run directory: {run_dir}")

    # Show complete configuration overrides
    rd = Path(run_dir)
    print("=" * 60)
    print("🔧 TRAINING CONFIGURATION")
    print("=" * 60)

    overrides = load_config_any(str(rd / ".hydra" / "overrides.yaml"))
    if isinstance(overrides, list):
        print("📋 Configuration overrides:")
        for override in overrides:
            print(f"   • {override}")
        print()
    elif overrides is not None:
        print(f"[WARN] Unexpected overrides.yaml format: {type(overrides)}")

    # Show config groups used
    config_groups = load_config_groups(run_dir)
    if config_groups:
        print("🏷️  Config groups:")
        for group, value in config_groups.items():
            print(f"   • {group}: {value}")
        print()

    # Show training details
    cfg = load_run_config(run_dir)
    print("📊 Training details:")
    run_info = cfg.get("run", {}) if isinstance(cfg, dict) else {}
    print(f"   • Total timesteps: {run_info.get('total_timesteps', 'unknown')}")
    print(f"   • Vec envs: {run_info.get('vec_envs', 'unknown')}")
    print(f"   • Seed: {run_info.get('seed', 'unknown')}")
    print(f"   • DT: {run_info.get('dt', 'unknown')}")

    print("=" * 60)
    print()  # Blank line for readability


def build_train_command() -> Tuple[str, str]:
    # Config group selections
    env = prompt_choice("Environment", find_group_options("env"), 0)
    robot = prompt_choice("Robot", find_group_options("robot"), 0)
    reward = prompt_choice("Reward", find_group_options("reward"), 0)
    algo = prompt_choice("Algorithm", find_group_options("algo"), 0)
    network = prompt_choice("Network", find_group_options("network"), 0)
    wandb = prompt_choice("Weights & Biases", find_group_options("wandb"), 0)

    # Run params
    n_envs = prompt_text("Vec envs", "16")
    timesteps = prompt_text("Total timesteps", "200000")
    seed = prompt_text("Seed", "0")

    overrides = [
        f"env={env}",
        f"robot={robot}",
        f"reward={reward}",
        f"algo={algo}",
        f"network={network}",
        f"wandb={wandb}",
        f"run.vec_envs={n_envs}",
        f"run.total_timesteps={timesteps}",
        f"run.seed={seed}",
    ]
    # Route to algorithm-specific trainer
    script = "training/train_ppo.py" if algo == "ppo" else "training/train_sac.py"
    cmd = f"python {script} {' '.join(overrides)}"
    return ("train", cmd)


def build_render_command() -> Tuple[str, str]:
    print("\nExamples:")
    print("  runs/20250910_104559/best/")
    print("  runs/20250910_104559/final/")
    print("  runs/20250910_104559/checkpoints/ckpt_step_100000/")

    model_path = prompt_text("Model directory (best/final/ckpt_step_N)", "")
    if not model_path:
        raise SystemExit("Model directory is required.")
    model_p = Path(model_path)
    if not model_p.is_dir():
        raise SystemExit(
            "Please provide a directory (best/, final/, or checkpoints/ckpt_step_N/)"
        )

    # Display model configuration (full run context)
    display_model_config(model_path)

    # Show VecNormalize that will be used (directory-only rule)
    p = Path(model_path).resolve()
    vec_candidate = None
    # Prefer explicit best/final naming if present
    if (p / "vecnorm_best.pkl").exists():
        vec_candidate = p / "vecnorm_best.pkl"
    elif (p / "vecnorm_final.pkl").exists():
        vec_candidate = p / "vecnorm_final.pkl"
    elif (p / "vecnorm.pkl").exists():
        vec_candidate = p / "vecnorm.pkl"
    if vec_candidate is not None:
        print(
            f"[INFO] Rollout will load VecNormalize from this directory: {vec_candidate}"
        )
    else:
        print(
            "[WARN] No VecNormalize stats found in the selected directory; playback uses raw observations"
        )

    steps = prompt_text("Steps", "600")
    seed = prompt_text("Seed", "42")
    record_name = prompt_text("Output MP4 filename (saves to run/outputs/)", "demo")

    # Ensure .mp4 extension
    if not record_name.endswith(".mp4"):
        record_name += ".mp4"

    # Create output path in run directory
    run_dir = detect_run_root(model_path)
    output_dir = Path(run_dir) / "outputs"
    output_path = output_dir / record_name

    cmd = (
        f"python training/rollout.py --model '{model_path}' --record '{output_path}' "
        f"--steps {steps} --deterministic --seed {seed}"
    )
    return ("render", cmd)


def main() -> None:
    print("AMR Launcher — pick a mode. Ctrl+C to exit.")
    try:
        while True:
            mode = prompt_choice("\nMode", ["train", "render", "exit"], 0)
            if mode == "exit":
                print("Bye!")
                return
            if mode == "train":
                _, cmd = build_train_command()
            elif mode == "render":
                _, cmd = build_render_command()
            print(f"\n[COMMAND] {cmd}")
            run = input("Run now? [Y/n]: ").strip().lower()
            if run in ("", "y", "yes"):
                os.system(cmd)
            else:
                print("Skipped execution. Showing command only.")
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")


if __name__ == "__main__":
    main()
