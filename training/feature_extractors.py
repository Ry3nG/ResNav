from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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

        # LiDAR conv spec
        if lidar_channels is None:
            lidar_channels = [16, 32, 16]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 3]
        assert len(lidar_channels) == len(kernel_sizes) >= 1

        layers: list[nn.Module] = []
        in_ch = self.lidar_k  # frames as channels
        L = self.lidar_beams
        for ch, ks in zip(lidar_channels, kernel_sizes):
            pad = ks // 2
            layers += [nn.Conv1d(in_ch, ch, kernel_size=ks, padding=pad), nn.ReLU()]
            in_ch = ch
        # Adaptive pooling to small fixed length for stability
        layers += [nn.AdaptiveAvgPool1d(8)]
        self.lidar_conv = nn.Sequential(*layers)

        # Optional temporal conv along K (time) dimension using depthwise conv per angle
        # Only register when enabled to retain strict backward compatibility for checkpoints
        if self.temporal_enabled:
            # Input for temporal conv will be shaped as (B, beams, K)
            # Keep length by padding = dilation * (k-1) // 2
            k = max(1, self._temporal_kernel_size)
            d = max(1, self._temporal_dilation)
            pad_t = (d * (k - 1)) // 2
            self.temporal_conv = nn.Conv1d(
                in_channels=self.lidar_beams,
                out_channels=self.lidar_beams,
                kernel_size=k,
                padding=pad_t,
                dilation=d,
                groups=self.lidar_beams,
            )
            self.temporal_act = nn.ReLU()

        # Branch heads for kin/path (tiny MLPs)
        self.kin_head = nn.Sequential(nn.Linear(4, kin_dim), nn.ReLU())
        self.path_head = nn.Sequential(nn.Linear(8, path_dim), nn.ReLU())

        # Fusion MLP to target out_dim
        # lidar output size: last_channels * 8
        lidar_feat_dim = in_ch * 8
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
        # Optional temporal conv along K (time) per angle beam
        if self.temporal_enabled:
            # (B, K, beams) -> (B, beams, K)
            lidar_t = lidar.transpose(1, 2)
            lidar_t = self.temporal_conv(lidar_t)
            lidar_t = self.temporal_act(lidar_t)
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
