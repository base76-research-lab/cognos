"""
exp_010 — Conv-backbone + mHC för CIFAR-10.

    ConvBackbone       — delat Conv-feature-extractor
    TwoLayerConvMHC    — open_loop / closed_loop: backbone → mHC_1 → mHC_2 → head
    GateOnlyConvNet    — gate_only: backbone → mHC_1 → head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mhc import mHCLayer

HIDDEN_DIM = 256
N_STREAMS  = 4


class ConvBackbone(nn.Module):
    """
    Minimal Conv-backbone för CIFAR-10 (3×32×32 → 256-dim vektor).

    Conv(3→32) → BN → ReLU → MaxPool   # (32,16,16)
    Conv(32→64) → BN → ReLU → MaxPool  # (64,8,8)
    Flatten → Dense(256) → ReLU
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(64 * 8 * 8, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B,32,16,16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B,64,8,8)
        x = x.flatten(1)                                  # (B,4096)
        return F.relu(self.fc(x))                         # (B,256)


class TwoLayerConvMHC(nn.Module):
    """backbone → mHC_1 → mHC_2 → head  (open_loop och closed_loop)"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 n_streams: int = N_STREAMS, n_classes: int = 10):
        super().__init__()
        self.backbone = ConvBackbone(hidden_dim)
        self.mhc_1    = mHCLayer(hidden_dim, n_streams)
        self.mhc_2    = mHCLayer(hidden_dim, n_streams)
        self.head     = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        h              = self.backbone(x)
        r1, H_res_1    = self.mhc_1(h)
        r2, H_res_2    = self.mhc_2(r1)
        return self.head(r2), H_res_1, H_res_2


class GateOnlyConvNet(nn.Module):
    """backbone → mHC_1 → head  (gate_only, exp_009-analog)"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 n_streams: int = N_STREAMS, n_classes: int = 10):
        super().__init__()
        self.backbone = ConvBackbone(hidden_dim)
        self.mhc_1    = mHCLayer(hidden_dim, n_streams)
        self.head     = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        h           = self.backbone(x)
        r1, H_res_1 = self.mhc_1(h)
        return self.head(r1), H_res_1
