"""Helpers for locating Hydra run directories and reading metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import load_config_any


def detect_run_root(model_path: str) -> str:
    """Detect the Hydra run directory associated with a model artifact."""
    p = Path(model_path).resolve()
    if p.is_dir():
        if p.name in {"best", "final"}:
            return str(p.parent)
        if p.parent.name == "checkpoints":
            return str(p.parent.parent)
        return str(p)

    if p.name == "best_model.zip" and p.parent.name == "best":
        return str(p.parent.parent)
    if p.name == "final_model.zip":
        return str(p.parent)
    if p.name == "model.zip" and p.parent.parent.name == "checkpoints":
        return str(p.parent.parent.parent)
    return str(p.parent)


def load_resolved_run_config(run_dir: str) -> Dict[str, Any]:
    """Load the resolved Hydra config for a run if present."""
    rd = Path(run_dir)
    resolved = rd / "resolved.yaml"
    hydra_cfg = rd / ".hydra" / "config.yaml"

    if resolved.is_file():
        cfg = load_config_any(str(resolved))
        return cfg if isinstance(cfg, dict) else {}
    if hydra_cfg.is_file():
        cfg = load_config_any(str(hydra_cfg))
        return cfg if isinstance(cfg, dict) else {}
    return {}


def read_run_overrides(run_dir: str) -> Dict[str, str]:
    """Parse Hydra overrides used to launch a run."""
    overrides_path = Path(run_dir) / ".hydra" / "overrides.yaml"
    if not overrides_path.is_file():
        return {}

    overrides = load_config_any(str(overrides_path))
    if not isinstance(overrides, list):
        return {}

    parsed: Dict[str, str] = {}
    for item in overrides:
        if isinstance(item, str) and "=" in item and not item.startswith("run."):
            key, value = item.split("=", 1)
            parsed[key] = value
    return parsed
