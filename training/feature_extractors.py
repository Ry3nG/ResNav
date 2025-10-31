from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from training.temporal_blocks import TemporalPyramidDS



class LiDAR1DConvExtractor(BaseFeaturesExtractor):
    """LiDAR-focused feature extractor for Dict observations.

    - Expects observation space with keys: 'lidar' (1D Box), 'kin' (4,), 'path' (8,)
    - LiDAR is provided as a flat vector of length (K * beams), where K is the
      number of stacked frames. We reshape to (B, C=K, L=beams) and apply Conv1d
      over the angular dimension (L), treating frames as channels.
    - K and beams are supplied via features_extractor_kwargs.

    Args (features_extractor_kwargs expected keys):
      - lidar_k: int, number of stacked frames
      - lidar_beams: int, number of lidar beams
      - lidar_channels: list[int], conv channel sizes (default [16, 32, 16])
      - kernel_sizes: list[int], conv kernel sizes (default [3, 5, 3])
      - out_dim: int, output feature dimension after fusion (default 128)
      - kin_dim: int, feature size for kinematics branch (default 16)
      - path_dim: int, feature size for path branch (default 16)
      - temporal_enabled: bool, apply optional temporal depthwise conv along K (default False)
      - temporal_kernel_size: int, kernel size for temporal conv (default 3)
      - temporal_dilation: int, dilation for temporal conv (default 1)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        lidar_k: int,
        lidar_beams: int,
        lidar_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        out_dim: int = 128,
        kin_dim: int = 16,
        path_dim: int = 16,
        temporal_enabled: bool = False,
        temporal_kernel_size: int = 3,
        temporal_dilation: int = 1,
        temporal_ks_list: list[int] | None = None,
        temporal_causal: bool = False,
        temporal_se_reduction: int = 8,
        use_batch_norm: bool = False,
        dropout_p: float = 0.0,
        pool_out_len: int = 8,
    ) -> None:
        # Compute total features dim before calling super
        self._out_dim = int(out_dim)
        super().__init__(observation_space, features_dim=self._out_dim)

        assert isinstance(
            observation_space, spaces.Dict
        ), "Requires Dict observation space"
        assert "lidar" in observation_space.spaces, "Missing 'lidar' key"
        assert "kin" in observation_space.spaces and "path" in observation_space.spaces

        self.lidar_k = int(lidar_k)
        self.lidar_beams = int(lidar_beams)
        self.temporal_enabled = bool(temporal_enabled)
        self._temporal_kernel_size = int(temporal_kernel_size)
        self._temporal_dilation = int(temporal_dilation)
        self._temporal_causal = bool(temporal_causal)
        self._temporal_se_reduction = int(temporal_se_reduction)
        self._temporal_ks_list = (
            list(temporal_ks_list) if temporal_ks_list is not None else [3, 5, 7]
        )
        self._use_batch_norm = bool(use_batch_norm)
        self._dropout_p = float(dropout_p)
        self._pool_out_len = int(pool_out_len)

        # LiDAR conv spec
        if lidar_channels is None:
            lidar_channels = [16, 32, 16]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 3]
        assert len(lidar_channels) == len(kernel_sizes) >= 1

        layers: list[nn.Module] = []
        in_ch = self.lidar_k  # frames as channels
        for ch, ks in zip(lidar_channels, kernel_sizes):
            pad = ks // 2
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=ks, padding=pad))
            if self._use_batch_norm:
                layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            if self._dropout_p > 0.0:
                layers.append(nn.Dropout(self._dropout_p))
            in_ch = ch
        # Adaptive pooling to small fixed length for stability
        layers += [nn.AdaptiveAvgPool1d(self._pool_out_len)]
        self.lidar_conv = nn.Sequential(*layers)

        # Optional temporal block along K (time) using multi-scale depthwise-separable convs
        # Only register when enabled to retain strict backward compatibility for checkpoints
        if self.temporal_enabled:
            ks_list = self._temporal_ks_list if isinstance(self._temporal_ks_list, list) else [3, 5, 7]
            self.temporal_block = TemporalPyramidDS(
                beams=self.lidar_beams,
                ks_list=ks_list,
                dilation=self._temporal_dilation,
                causal=bool(self._temporal_causal),
                se_reduction=int(self._temporal_se_reduction),
            )

        # Branch heads for kin/path (tiny MLPs)
        self.kin_head = nn.Sequential(nn.Linear(4, kin_dim), nn.ReLU())
        self.path_head = nn.Sequential(nn.Linear(8, path_dim), nn.ReLU())

        # Fusion MLP to target out_dim
        # lidar output size: last_channels * pool_out_len
        lidar_feat_dim = in_ch * self._pool_out_len
        fused_in = lidar_feat_dim + kin_dim + path_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, max(self._out_dim, 64)),
            nn.ReLU(),
            nn.Linear(max(self._out_dim, 64), self._out_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # LiDAR: (B, K*beams) -> (B, K, beams)
        lidar = observations["lidar"]
        B = lidar.shape[0]
        # Ensure divisibility
        assert (
            lidar.shape[1] == self.lidar_k * self.lidar_beams
        ), f"Unexpected lidar dim {lidar.shape[1]} ≠ {self.lidar_k}*{self.lidar_beams}"
        lidar = lidar.view(B, self.lidar_k, self.lidar_beams)
        # Optional temporal block along K (time) per angle beam
        if self.temporal_enabled:
            # (B, K, beams) -> (B, beams, K)
            lidar_t = lidar.transpose(1, 2)
            lidar_t = self.temporal_block(lidar_t)
            # (B, beams, K) -> (B, K, beams)
            lidar = lidar_t.transpose(1, 2)
        # Conv over beams with frames as channels
        x_lidar = self.lidar_conv(lidar)  # (B, C, 8)
        x_lidar = torch.flatten(x_lidar, start_dim=1)  # (B, C*8)

        # Kin & path heads
        x_kin = self.kin_head(observations["kin"])  # (B, kin_dim)
        x_path = self.path_head(observations["path"])  # (B, path_dim)

        # Fuse
        x = torch.cat([x_lidar, x_kin, x_path], dim=1)
        x = self.fusion(x)
        return x
