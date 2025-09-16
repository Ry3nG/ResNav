"""Config loading helpers built around OmegaConf."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf


def load_config_any(path: str) -> Any:
    """Load a YAML/OMEGACONF file and return the resolved Python object."""
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_config_dict(path: str) -> Dict[str, Any]:
    """Load a config file and guarantee a `dict` result."""
    cfg = load_config_any(path)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(cfg)}")
    return cfg
