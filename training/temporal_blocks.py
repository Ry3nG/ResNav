# ==== temporal_blocks.py ====
import torch
import torch.nn as nn

class TemporalPyramidDS(nn.Module):
    """
    Multi-scale depthwise-separable temporal convolution block:
      - Parallel depthwise 1D convolutions along time, kernel_size ∈ ks_list
      - Concatenate then fuse with 1×1 pointwise to capture cross-beam dynamics
      - Optional causal convolution (uses only past frames to avoid leakage)
      - Lightweight SE gating (channel re-weighting to emphasize risky sectors)
    Input:  x ∈ R^{B × beams × K}
    Output: y ∈ R^{B × beams × K}
    """
    def __init__(
        self,
        beams: int,
        ks_list=(3, 5, 7),
        dilation: int = 1,
        causal: bool = False,
        se_reduction: int = 8,
        act: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = bool(causal)
        self.dilation = int(dilation)
        self.act = act if act is not None else nn.ReLU(inplace=True)

        # Multi-scale depthwise branches
        branches = []
        for k in ks_list:
            k = int(k)
            # Causal: left padding (d*(k-1)); Non-causal: same padding
            pad = (self.dilation * (k - 1)) if self.causal else (self.dilation * (k - 1)) // 2
            branches.append(
                nn.Conv1d(
                    in_channels=beams, out_channels=beams,
                    kernel_size=k, padding=pad, dilation=self.dilation,
                    groups=beams, bias=True
                )
            )
        self.branches = nn.ModuleList(branches)

        # Fusion: concat → 1×1 pointwise to restore to 'beams'
        self.pointwise = nn.Conv1d(
            in_channels=beams * len(self.branches),
            out_channels=beams,
            kernel_size=1, bias=True
        )

        # Lightweight SE (per-beam gating)
        hidden = max(1, beams // int(se_reduction))
        self.se_fc1 = nn.Conv1d(beams, hidden, kernel_size=1)
        self.se_fc2 = nn.Conv1d(hidden, beams, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, beams, K)
        outs = []
        for conv in self.branches:
            y = conv(x)  # (B, beams, K') with padding chosen above
            if self.causal:
                # Trim to match input temporal length (keep only past frames)
                y = y[..., : x.size(-1)]
            outs.append(y)
        y = torch.cat(outs, dim=1)           # (B, beams * n_scales, K)
        y = self.act(self.pointwise(y))      # (B, beams, K)

        # SE gating (temporal average to obtain per-beam gating coefficients)
        s = y.mean(dim=-1, keepdim=True)     # (B, beams, 1)
        s = self.act(self.se_fc1(s))         # (B, hidden, 1)
        s = torch.sigmoid(self.se_fc2(s))    # (B, beams, 1)
        return y * s                         # (B, beams, K)
