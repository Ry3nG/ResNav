"""Configuration parsing and external service initialization.

Extracted from training/common.py to separate concerns.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


__all__ = ["resolve_cfg", "maybe_init_wandb"]


def _to_dict(cfg_section: Any) -> dict[str, Any]:
    """Convert OmegaConf section to plain dict."""
    if isinstance(cfg_section, DictConfig):
        data = OmegaConf.to_container(cfg_section, resolve=True)
    else:
        data = cfg_section
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Expected config section to resolve to a dict")
    return dict(data)


def resolve_cfg(
    cfg: DictConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Parse Hydra config into individual dictionaries.

    Args:
        cfg: Hydra DictConfig with keys: env, robot, reward, algo, network, wandb, run

    Returns:
        Tuple of (env_cfg, robot_cfg, reward_cfg, algo_cfg, network_cfg, wandb_cfg, run_cfg)
    """
    env_cfg = _to_dict(cfg.env)
    robot_cfg = _to_dict(cfg.robot)
    reward_cfg = _to_dict(cfg.reward)
    algo_cfg = _to_dict(cfg.algo)
    network_cfg = _to_dict(cfg.network)
    wandb_cfg = _to_dict(cfg.get("wandb", {}))
    run_cfg = _to_dict(cfg.run)
    return env_cfg, robot_cfg, reward_cfg, algo_cfg, network_cfg, wandb_cfg, run_cfg


def maybe_init_wandb(wandb_cfg: dict[str, Any], extra_config: dict[str, Any]):
    """Initialize Weights & Biases if enabled in config.

    Args:
        wandb_cfg: Dict with keys: mode, project, entity
        extra_config: Additional config to log to wandb

    Returns:
        wandb.Run instance if enabled, None otherwise
    """
    mode = str(wandb_cfg.get("mode", "disabled"))
    if mode == "disabled":
        return None

    import wandb

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        mode=mode,
        sync_tensorboard=True,
        dir=".",
        config=extra_config,
    )
    return run
