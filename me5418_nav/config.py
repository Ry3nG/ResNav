from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import math

from .constants import (
    DT,
    ROBOT_V_MAX_MPS,
    ROBOT_W_MAX_RPS,
    EPISODE_MAX_STEPS,
    LIDAR_NUM_BEAMS,
    LIDAR_FOV_DEG,
    LIDAR_MAX_RANGE_M,
    GRID_RESOLUTION_M,
    PATH_PREVIEW_K,
    PATH_PREVIEW_DS,
    PATH_PREVIEW_RANGE_M,
)


@dataclass
class RewardConfig:
    w_prog: float = 1.0
    w_lat: float = 0.2
    w_head: float = 0.1
    w_clear: float = 0.4
    w_dv: float = 0.05
    w_dw: float = 0.02
    R_goal: float = 50.0
    R_collide: float = 50.0
    R_timeout: float = 10.0
    clearance_safe_m: float = 0.5

    def __post_init__(self) -> None:
        assert self.w_prog > 0.0, "w_prog must be positive"
        for name in ("w_lat", "w_head", "w_clear", "w_dv", "w_dw"):
            assert getattr(self, name) >= 0.0, f"{name} must be >= 0"
        for name in ("R_goal", "R_collide", "R_timeout"):
            assert getattr(self, name) >= 0.0, f"{name} must be >= 0"
        assert self.clearance_safe_m > 0.0, "clearance_safe_m must be > 0"


@dataclass
class DynamicsConfig:
    dt: float = DT
    v_max: float = ROBOT_V_MAX_MPS
    w_max: float = ROBOT_W_MAX_RPS
    max_steps: int = EPISODE_MAX_STEPS

    def __post_init__(self) -> None:
        assert self.dt > 0.0, "dt must be > 0"
        assert self.v_max > 0.0, "v_max must be > 0"
        assert self.w_max > 0.0, "w_max must be > 0"
        assert self.max_steps > 0, "max_steps must be > 0"


@dataclass
class LidarConfig:
    beams: int = LIDAR_NUM_BEAMS
    fov_deg: float = LIDAR_FOV_DEG
    max_range_m: float = LIDAR_MAX_RANGE_M
    step_m: float = max(1e-3, GRID_RESOLUTION_M * 0.5)

    def __post_init__(self) -> None:
        assert self.beams > 0, "beams must be > 0"
        assert 0.0 < self.fov_deg <= 360.0, "fov_deg in (0,360]"
        assert self.max_range_m > 0.0, "max_range_m must be > 0"
        assert self.step_m > 0.0, "step_m must be > 0"


@dataclass
class PathPreviewConfig:
    K: int = PATH_PREVIEW_K
    ds: float = PATH_PREVIEW_DS
    range_m: float = PATH_PREVIEW_RANGE_M

    def __post_init__(self) -> None:
        assert self.K >= 0, "K must be >= 0"
        assert self.ds > 0.0, "ds must be > 0"
        assert self.range_m > 0.0, "range_m must be > 0"


@dataclass
class CurriculumStage:
    """Single stage in curriculum learning with episode range and difficulty parameters."""
    episode_range: Tuple[int, int]  # (start, end) episode numbers
    num_pallets_range: Tuple[int, int]  # (min, max) pallet count
    corridor_width_range: Tuple[float, float]  # (min, max) corridor width


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning system."""
    enabled: bool = False
    stages: List[CurriculumStage] = field(default_factory=list)

    def __post_init__(self):
        if self.enabled and not self.stages:
            # Default 3-stage curriculum
            self.stages = [
                CurriculumStage([0, 20000], [1, 1], [2.6, 3.0]),
                CurriculumStage([20000, 60000], [1, 2], [2.4, 2.8]),
                CurriculumStage([60000, 100000], [2, 3], [2.2, 2.6]),
            ]


class CurriculumManager:
    """Manages curriculum learning progression based on training episodes."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_stage_idx = 0
        
    def get_stage_config(self, episode_num: int) -> CurriculumStage:
        """Get current curriculum stage configuration for given episode number."""
        if not self.config.enabled or not self.config.stages:
            # Return default stage if curriculum disabled
            return CurriculumStage([0, float('inf')], [1, 1], [2.2, 3.0])
        
        # Find appropriate stage based on episode number
        for stage_config in self.config.stages:
            start, end = stage_config.episode_range
            if start <= episode_num < end:
                return stage_config
                
        # If past all stages, return the final stage
        return self.config.stages[-1]
    
    def should_update_stage(self, episode_num: int) -> bool:
        """Check if curriculum should transition to next stage."""
        if not self.config.enabled or not self.config.stages:
            return False
            
        current_stage = self.get_stage_config(episode_num)
        prev_stage = self.get_stage_config(episode_num - 1) if episode_num > 0 else None
        
        return prev_stage != current_stage
    
    def get_stage_name(self, episode_num: int) -> str:
        """Get human-readable stage name for logging."""
        stage = self.get_stage_config(episode_num)
        pallets = f"{stage.num_pallets_range[0]}-{stage.num_pallets_range[1]}"
        corridor = f"{stage.corridor_width_range[0]:.1f}-{stage.corridor_width_range[1]:.1f}m"
        return f"Pallets:{pallets}, Corridor:{corridor}"


@dataclass
class EnvConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    preview: PathPreviewConfig = field(default_factory=PathPreviewConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    render_mode: Optional[str] = None
    scenario: str = "blockage"
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> "EnvConfig":
        d = cfg or {}
        # Allow nested dicts or top-level overrides
        reward = d.get("reward") or d.get("reward_weights") or {}
        dynamics = {
            "dt": d.get("dt", d.get("dynamics", {}).get("dt", DT)),
            "v_max": d.get("v_max", d.get("dynamics", {}).get("v_max", ROBOT_V_MAX_MPS)),
            "w_max": d.get("w_max", d.get("dynamics", {}).get("w_max", ROBOT_W_MAX_RPS)),
            "max_steps": d.get("max_steps", d.get("dynamics", {}).get("max_steps", EPISODE_MAX_STEPS)),
        }
        lidar_cfg = d.get("lidar", {})
        preview_cfg = d.get("preview", {})
        curriculum_cfg = d.get("curriculum", {})
        
        # Parse curriculum stages if provided
        stages = []
        if curriculum_cfg.get("stages"):
            for stage_dict in curriculum_cfg["stages"]:
                stages.append(CurriculumStage(
                    episode_range=tuple(stage_dict["episode_range"]),
                    num_pallets_range=tuple(stage_dict["num_pallets_range"]),
                    corridor_width_range=tuple(stage_dict["corridor_width_range"])
                ))
        
        env = cls(
            reward=RewardConfig(**reward),
            dynamics=DynamicsConfig(**dynamics),
            lidar=LidarConfig(**lidar_cfg),
            preview=PathPreviewConfig(**preview_cfg),
            curriculum=CurriculumConfig(
                enabled=curriculum_cfg.get("enabled", False),
                stages=stages
            ),
            render_mode=d.get("render_mode"),
            scenario=d.get("scenario", "blockage"),
            scenario_kwargs=dict(d.get("scenario_kwargs", {})),
        )
        return env

