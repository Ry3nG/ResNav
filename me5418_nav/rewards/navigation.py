from __future__ import annotations

from dataclasses import dataclass
from ..constants import REWARD_STEP, REWARD_COLLISION
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class RewardConfig:
    step_penalty: float = REWARD_STEP
    collision_penalty: float = REWARD_COLLISION
    progress_weight: float = 1.0
    tracking_weight: float = -0.1
    clearance_weight: float = -0.05
    smoothness_weight: float = -0.01


def navigation_reward(
    obs: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    events: Dict[str, Any],
    cfg: RewardConfig = RewardConfig(),
) -> Tuple[float, Dict[str, float]]:
    r_step = cfg.step_penalty
    r_collision = cfg.collision_penalty if events.get("collision", False) else 0.0
    r_progress = cfg.progress_weight * float(events.get("progress", 0.0))
    cross_track = float(obs[-1 - 6 - 2])
    heading_err = float(obs[-1 - 6 - 1])
    r_tracking = cfg.tracking_weight * (abs(cross_track) + abs(heading_err))
    clearance = float(events.get("clearance", 0.0))
    r_clearance = cfg.clearance_weight * clearance
    if prev_action is None:
        r_smooth = 0.0
    else:
        r_smooth = cfg.smoothness_weight * float(np.linalg.norm(action - prev_action))

    reward = r_step + r_collision + r_progress + r_tracking + r_clearance + r_smooth
    comps = {
        "step": r_step,
        "collision": r_collision,
        "progress": r_progress,
        "tracking": r_tracking,
        "clearance": r_clearance,
        "smoothness": r_smooth,
    }
    return float(reward), comps
