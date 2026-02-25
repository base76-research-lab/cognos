"""
exp_009 — Modeller för tre betingelser.

    TwoLayerMHC   — gemensam arkitektur för open_loop och closed_loop
                    Flatten → Dense(128) → mHC_1 → mHC_2 → head

    GateOnlyNet   — exp_008-analog för jämförelse
                    Flatten → Dense(128) → mHC_1 → head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mhc import mHCLayer, routing_entropy

HIDDEN_DIM = 128
N_STREAMS = 4


class TwoLayerMHC(nn.Module):
    """
    Två-lagers mHC-modell.

    forward() returnerar alltid (logits, H_res_1, H_res_2).
    Gaten tillämpas UTANFÖR modellen vid eval (se run.py).
    """

    def __init__(self,
                 hidden_dim: int = HIDDEN_DIM,
                 n_streams: int = N_STREAMS,
                 n_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.mhc_1 = mHCLayer(hidden_dim, n_streams)
        self.mhc_2 = mHCLayer(hidden_dim, n_streams)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(self.flatten(x)))
        r1, H_res_1 = self.mhc_1(h)
        r2, H_res_2 = self.mhc_2(r1)
        logits = self.head(r2)
        return logits, H_res_1, H_res_2


class GateOnlyNet(nn.Module):
    """
    Enkellagers mHC + gate — direkt analog till exp_008.

    forward() returnerar (logits, H_res_1).
    """

    def __init__(self,
                 hidden_dim: int = HIDDEN_DIM,
                 n_streams: int = N_STREAMS,
                 n_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.mhc_1 = mHCLayer(hidden_dim, n_streams)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(self.flatten(x)))
        r1, H_res_1 = self.mhc_1(h)
        logits = self.head(r1)
        return logits, H_res_1
