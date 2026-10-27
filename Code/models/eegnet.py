"""EEGNet-style network for subject-level group classification (about 2 k parameters for 32 inputs).

1x1 spatial convolution (inputs -> 16), depthwise temporal convolution (kernel 31),
GroupNorm, ELU, average pool 4, dropout 0.25; separable convolution (kernel 15),
GroupNorm, ELU, average pool 8, dropout 0.25; LayerNorm, dropout 0.3, linear output.
It differs from the original EEGNet-8,2 in block order, kernel lengths, GroupNorm
instead of BatchNorm, and weight decay instead of max-norm constraints.

The same network serves one scout (32 PCs), several scouts stacked as input rows (the
"flat" multi-scout model: the first 1x1 convolution mixes all scouts), and sensors.

``fast=True`` computes the same layers with the same weights, but runs the 1x1
convolutions as matrix multiplies and the kernel-15 depthwise convolution via unfold,
which PyTorch's CPU convolutions handle slowly (4-16x faster training depending on the
number of inputs). Outputs agree to ~1e-6, so trained models are statistically
equivalent but not bit-identical to ``fast=False``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _depthwise_unfold(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    k = weight.shape[-1]
    return torch.einsum("bctk,ck->bct", F.pad(x, (k // 2, k // 2)).unfold(-1, k, 1), weight[:, 0])


class EEGNet(nn.Module):
    def __init__(self, n_in: int, n_times: int, fast: bool = False) -> None:
        super().__init__()
        self.fast = fast
        self.features = nn.Sequential(
            nn.Conv1d(n_in, 16, 1, bias=False),
            nn.Conv1d(16, 16, 31, padding=15, groups=16, bias=False), nn.GroupNorm(4, 16), nn.ELU(),
            nn.AvgPool1d(4), nn.Dropout(0.25),
            nn.Conv1d(16, 16, 15, padding=7, groups=16, bias=False), nn.Conv1d(16, 16, 1, bias=False),
            nn.GroupNorm(4, 16), nn.ELU(), nn.AvgPool1d(8), nn.Dropout(0.25), nn.Flatten())
        width = 16 * (n_times // 32)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.3), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fast:
            return self.head(self.features(x))
        f = self.features
        x = torch.matmul(f[0].weight[:, :, 0], x)  # 1x1 spatial convolution
        for layer in f[1:6]:
            x = layer(x)
        x = _depthwise_unfold(x, f[6].weight)  # separable: depthwise (kernel 15) ...
        x = torch.matmul(f[7].weight[:, :, 0], x)  # ... then pointwise
        for layer in f[8:]:
            x = layer(x)
        return self.head(x)
