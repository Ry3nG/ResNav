"""Model initialization and policy network configuration.

Extracted from training/common.py to separate model-specific logic.
"""

from __future__ import annotations

from typing import Any
import os

import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from training.feature_extractors import LiDAR1DConvExtractor


__all__ = ["build_policy_kwargs", "_init_model", "configure_logger"]


def build_policy_kwargs(
    network_cfg: dict[str, Any], env_cfg: dict[str, Any], algo_name: str
) -> dict[str, Any]:
    """Build policy_kwargs for Stable-Baselines3 algorithm.

    Args:
        network_cfg: Network architecture config with actor/critic/feature_extractor sections
        env_cfg: Environment config (used for lidar/frame_stack parameters)
        algo_name: Algorithm name (currently only 'sac' supported)

    Returns:
        Dict with net_arch, activation_fn, and optional features_extractor_* keys
    """
    actor_cfg = network_cfg.get("actor", {})
    critic_cfg = network_cfg.get("critic", {})
    activation = str(actor_cfg.get("activation", "relu")).lower()
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    # SAC uses pi (actor) and qf (Q-function) architecture
    net_arch: dict[str, Any] = {
        "pi": actor_cfg.get("hidden_sizes", []),
        "qf": critic_cfg.get("hidden_sizes", []),
    }

    policy_kwargs: dict[str, Any] = {
        "net_arch": net_arch,
        "activation_fn": act_map.get(activation, nn.ReLU),
    }

    fe_cfg = network_cfg.get("feature_extractor", {})
    if isinstance(fe_cfg, dict) and fe_cfg.get("lidar_branch", "mlp") == "cnn1d":
        frame_stack = int(env_cfg["wrappers"]["frame_stack"]["k"])
        beams = int(env_cfg["lidar"]["beams"])
        policy_kwargs["features_extractor_class"] = LiDAR1DConvExtractor
        fe_kwargs: dict[str, Any] = {
            "lidar_k": int(fe_cfg.get("lidar_k", frame_stack)),
            "lidar_beams": int(fe_cfg.get("lidar_beams", beams)),
            "lidar_channels": list(fe_cfg.get("lidar_channels", [16, 32, 16])),
            "kernel_sizes": list(fe_cfg.get("kernel_sizes", [3, 5, 3])),
            "out_dim": int(fe_cfg.get("out_dim", 128)),
            "kin_dim": int(fe_cfg.get("kin_dim", 16)),
            "path_dim": int(fe_cfg.get("path_dim", 16)),
        }
        policy_kwargs["features_extractor_kwargs"] = fe_kwargs
    return policy_kwargs


def _init_model(
    algo_name: str,
    algo_cfg: dict[str, Any],
    policy_kwargs: dict[str, Any],
    train_env,
    seed: int,
):
    """Initialize SAC model with given configuration.

    Args:
        algo_name: Algorithm name (currently only 'sac' supported)
        algo_cfg: Algorithm hyperparameters (lr, buffer_size, batch_size, etc.)
        policy_kwargs: Policy network configuration from build_policy_kwargs()
        train_env: Training VecEnv instance
        seed: Random seed

    Returns:
        Initialized SAC model
    """
    if algo_name.lower() != "sac":
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    load_path = str(algo_cfg.get("load_path") or "")
    if load_path:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"SAC load_path not found: {load_path}")
        print(f"\n{'='*60}")
        print(f"🔄 Loading pretrained SAC model from checkpoint:")
        print(f"   {load_path}")
        print(f"{'='*60}\n")
        return SAC.load(load_path, env=train_env, tensorboard_log="tb_logs")

    return SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=float(algo_cfg["lr"]),
        buffer_size=int(algo_cfg["buffer_size"]),
        batch_size=int(algo_cfg["batch_size"]),
        tau=float(algo_cfg.get("tau", 0.005)),
        gamma=float(algo_cfg["gamma"]),
        train_freq=int(algo_cfg.get("train_freq", 1)),
        gradient_steps=int(algo_cfg.get("gradient_steps", 1)),
        learning_starts=int(algo_cfg.get("learning_starts", 10000)),
        ent_coef=algo_cfg.get("ent_coef", "auto"),
        target_update_interval=int(algo_cfg.get("target_update_interval", 1)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log="tb_logs",
    )


def configure_logger(model) -> None:
    """Configure Stable-Baselines3 logger for stdout, CSV, and TensorBoard.

    Args:
        model: SB3 model instance to configure
    """
    logger = configure("logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
