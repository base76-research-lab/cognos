"""
mHC-lager — extraherat från exp_008 för återanvändning.

Exponerar:
    sinkhorn_knopp(H, n_iter)   — Birkhoff-projektion
    mHCLayer(in_dim, n_streams) — routing-lager med epistemic signal
    IdentityRoutingLayer(...)   — kontroll (H_res = I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SINKHORN_N = 20


def sinkhorn_knopp(H: torch.Tensor, n_iter: int = SINKHORN_N) -> torch.Tensor:
    """Projicerar (batch, n, n) till Birkhoff-polytopen."""
    H = H.abs() + 1e-8
    for _ in range(n_iter):
        H = H / H.sum(dim=2, keepdim=True)
        H = H / H.sum(dim=1, keepdim=True)
    return H


def routing_entropy(H_res: torch.Tensor) -> torch.Tensor:
    """
    H(H_res) = −Σ_i Σ_j h_ij · log(h_ij + ε)  per sample.
    Input: (batch, n, n).  Output: (batch,)
    """
    eps = 1e-8
    return -(H_res * (H_res + eps).log()).sum(dim=(1, 2))


class mHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Layer.

    Returnerar: (routed_flat, H_res) — (B, in_dim), (B, n, n)
    """

    def __init__(self, in_dim: int, n_streams: int,
                 sinkhorn_iters: int = SINKHORN_N):
        super().__init__()
        assert in_dim % n_streams == 0
        self.n_streams = n_streams
        self.stream_dim = in_dim // n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.gate = nn.Linear(in_dim, n_streams * n_streams)
        self.stream_proj = nn.Linear(self.stream_dim, self.stream_dim)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        streams = x.view(batch, self.n_streams, self.stream_dim)
        H_raw = self.gate(x).view(batch, self.n_streams, self.n_streams)
        H_res = sinkhorn_knopp(H_raw, self.sinkhorn_iters)
        routed = torch.bmm(H_res, streams)
        flat = routed.reshape(batch * self.n_streams, self.stream_dim)
        out = F.relu(self.stream_proj(flat))
        return out.reshape(batch, -1), H_res


class IdentityRoutingLayer(nn.Module):
    """H_res = I — kontrollbetingelse."""

    def __init__(self, in_dim: int, n_streams: int):
        super().__init__()
        assert in_dim % n_streams == 0
        self.n_streams = n_streams
        self.stream_dim = in_dim // n_streams
        self.stream_proj = nn.Linear(self.stream_dim, self.stream_dim)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        streams = x.view(batch, self.n_streams, self.stream_dim)
        H_res = (torch.eye(self.n_streams, device=x.device)
                 .unsqueeze(0).expand(batch, -1, -1))
        flat = streams.reshape(batch * self.n_streams, self.stream_dim)
        out = F.relu(self.stream_proj(flat))
        return out.reshape(batch, -1), H_res
