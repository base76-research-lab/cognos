#!/usr/bin/env python3
"""
exp_008 — micro-mHC på MNIST: Empirisk test av P1

Prediktion P1:
    En mHC-arkitektur med Birkhoff-routing (Sinkhorn-Knopp) ger
    r(routing_entropy, softmax_entropy) > 0.

Nollhypotes:
    Kontrollbetingelse H_res=I ger r ≤ 0 (replikerar exp_007-inversionsfyndet).

Bakgrund:
    exp_007 visade att standard dense-lager ger INVERTERAD signal:
    r(state_div, Ue) ≈ −0.35. mHC-hypotesen säger att Birkhoff-routing
    vänder tecknet. Denna prediktion testas här från scratch.

Design:
    - micro-mHC MLP: Flatten → Dense(128) ReLU → mHC(4 strömmar×32) → Dense(10)
    - mHC: per-sample H_res via gating-nät + Sinkhorn-Knopp → dubbelt stokastisk
    - Kontroll: H_res = I (identitetsrouting — ingen blandning)
    - 5 seeds (0–4) per betingelse
    - 2000 MNIST-testsamples
    - Primärsignal: routing-entropi H(H_res) vs softmax-entropi H(p)
    - Sekundärsignal: tillstånds-divergens L2 vs H(p)

Körning:
    python3 run_exp_008.py

Utdata:
    results.json           — rådata per seed/betingelse
    fig_exp_008_scatter.png — scatter-plot 2×2 (routing_ent + state_div, mHC + ctrl)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

# ── matplotlib: valfritt (genererar figur om tillgänglig) ─────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Konfiguration ──────────────────────────────────────────────────────────────

N_STREAMS     = 4
STREAM_DIM    = 32          # HIDDEN_DIM / N_STREAMS
HIDDEN_DIM    = 128         # N_STREAMS × STREAM_DIM
SINKHORN_N    = 20
BATCH_SIZE    = 256
EPOCHS        = 5
EVAL_N        = 2000
SEEDS         = [0, 1, 2, 3, 4]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASY_DIGITS   = {0, 1}
HARD_DIGITS   = {4, 7, 9}
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))


# ── Sinkhorn-Knopp ─────────────────────────────────────────────────────────────

def sinkhorn_knopp(H: torch.Tensor, n_iter: int = SINKHORN_N) -> torch.Tensor:
    """
    Projicerar en batch av matriser till Birkhoff-polytopen.
    H: (batch, n, n) — returnerar dubbelt stokastisk per sample.
    Alternerar rad- och kolumnnormalisering.
    """
    H = H.abs() + 1e-8
    for _ in range(n_iter):
        H = H / H.sum(dim=2, keepdim=True)   # rad-normalisering
        H = H / H.sum(dim=1, keepdim=True)   # kolumn-normalisering
    return H


# ── Lager ──────────────────────────────────────────────────────────────────────

class mHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Layer (förenklad micro-version).

    1. Dela indata i n_streams parallella strömmar.
    2. Gating-nät: indata → per-sample routing-logits (n×n).
    3. Sinkhorn-Knopp: logits → dubbelt stokastisk H_res ∈ Birkhoff-polytopen.
    4. Route:   routed_i = Σ_j H_res[i,j] · stream_j
    5. Per-ström ReLU-projektion.

    Returnerar: (routed_flat, H_res) — (batch, in_dim), (batch, n, n)
    """

    def __init__(self, in_dim: int, n_streams: int,
                 sinkhorn_iters: int = SINKHORN_N):
        super().__init__()
        assert in_dim % n_streams == 0, "in_dim måste vara delbart med n_streams"
        self.n_streams      = n_streams
        self.stream_dim     = in_dim // n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Gating: indata → routing-logits
        self.gate = nn.Linear(in_dim, n_streams * n_streams)

        # Per-ström linjär projektion
        self.stream_proj = nn.Linear(self.stream_dim, self.stream_dim)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]

        # Strömmar
        streams = x.view(batch, self.n_streams, self.stream_dim)  # (B, n, d)

        # Routing-matris: Birkhoff-projicerad
        H_raw = self.gate(x).view(batch, self.n_streams, self.n_streams)
        H_res = sinkhorn_knopp(H_raw, self.sinkhorn_iters)         # (B, n, n)

        # Blandning via bmm: routed[b,i,:] = Σ_j H_res[b,i,j] · streams[b,j,:]
        routed = torch.bmm(H_res, streams)                         # (B, n, d)

        # Per-ström projektion
        flat     = routed.reshape(batch * self.n_streams, self.stream_dim)
        out_flat = F.relu(self.stream_proj(flat))
        out      = out_flat.reshape(batch, self.n_streams, self.stream_dim)

        return out.reshape(batch, -1), H_res


class IdentityRoutingLayer(nn.Module):
    """
    Kontrollbetingelse: H_res = I (identitetsrouting, ingen blandning).

    Ingen gating, inga parametrar utöver per-ström projektion.
    Replikerar exp_007-scenariot — förväntas ge r(state_div, Ue) < 0.

    Returnerar: (routed_flat, H_res) — (batch, in_dim), (batch, n, n)
    """

    def __init__(self, in_dim: int, n_streams: int):
        super().__init__()
        assert in_dim % n_streams == 0
        self.n_streams  = n_streams
        self.stream_dim = in_dim // n_streams
        self.stream_proj = nn.Linear(self.stream_dim, self.stream_dim)

    def forward(self, x: torch.Tensor):
        batch   = x.shape[0]
        streams = x.view(batch, self.n_streams, self.stream_dim)

        # Fast identitetsrouting
        H_res = (torch.eye(self.n_streams, device=x.device)
                 .unsqueeze(0)
                 .expand(batch, -1, -1))

        flat     = streams.reshape(batch * self.n_streams, self.stream_dim)
        out_flat = F.relu(self.stream_proj(flat))
        out      = out_flat.reshape(batch, self.n_streams, self.stream_dim)

        return out.reshape(batch, -1), H_res


# ── Modeller ───────────────────────────────────────────────────────────────────

class MNISTNet_mHC(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28 * 28, HIDDEN_DIM)
        self.mhc     = mHCLayer(HIDDEN_DIM, N_STREAMS)
        self.fc_out  = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        h              = F.relu(self.fc1(self.flatten(x)))
        routed, H_res  = self.mhc(h)
        return self.fc_out(routed), H_res


class MNISTNet_ctrl(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28 * 28, HIDDEN_DIM)
        self.ctrl    = IdentityRoutingLayer(HIDDEN_DIM, N_STREAMS)
        self.fc_out  = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        h              = F.relu(self.fc1(self.flatten(x)))
        routed, H_res  = self.ctrl(h)
        return self.fc_out(routed), H_res


# ── Signaler ───────────────────────────────────────────────────────────────────

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(p) = −Σ p·log(p+ε),  (batch,)"""
    p = F.softmax(logits, dim=1)
    return -(p * (p + 1e-9).log()).sum(dim=1)


def routing_entropy(H_res: torch.Tensor) -> torch.Tensor:
    """
    Elementvis entropi i routing-matrisen per sample.
    H_res: (batch, n, n) — dubbelt stokastisk.
    Returnerar: (batch,)  — summa av −h·log(h) för alla element.
    Hög värde = diffus routing (blandning); lågt värde = koncentrerad routing.
    """
    H = H_res.clamp(1e-9, 1.0)
    return -(H * H.log()).sum(dim=(1, 2))


def state_divergence(routed_streams: torch.Tensor) -> torch.Tensor:
    """
    Genomsnittlig L2-avstånd från strömmar till medelvektorn.
    routed_streams: (batch, n_streams, stream_dim)
    Returnerar: (batch,)
    """
    mean = routed_streams.mean(dim=1, keepdim=True)
    return torch.norm(routed_streams - mean, dim=2).mean(dim=1)


# ── Träning / Evaluering ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


# ── Mät alla signaler för 2000 samples ────────────────────────────────────────

def measure_signals(model, test_x: torch.Tensor, test_y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits, H_res = model(test_x)

        # Extrahera routed streams direkt
        h_flat = F.relu(model.fc1(model.flatten(test_x)))
        if hasattr(model, "mhc"):
            out_flat, H = model.mhc(h_flat)
        else:
            out_flat, H = model.ctrl(h_flat)

    batch       = test_x.shape[0]
    streams_out = out_flat.view(batch, N_STREAMS, STREAM_DIM)

    ent  = softmax_entropy(logits).cpu().numpy()
    rte  = routing_entropy(H_res).cpu().numpy()
    sdiv = state_divergence(streams_out).cpu().numpy()

    probs     = F.softmax(logits, dim=1)
    p_max     = probs.max(dim=1).values.cpu().numpy()
    pred      = logits.argmax(dim=1).cpu().numpy()
    labels    = test_y.numpy()
    correct   = (pred == labels)
    easy_mask = np.isin(labels, list(EASY_DIGITS))
    hard_mask = np.isin(labels, list(HARD_DIGITS))

    return {
        "entropy":   ent,
        "rte":       rte,
        "sdiv":      sdiv,
        "p_max":     p_max,
        "correct":   correct,
        "labels":    labels,
        "easy_mask": easy_mask,
        "hard_mask": hard_mask,
    }


# ── Ett körpass (en seed + en betingelse) ─────────────────────────────────────

def run_seed(seed: int, model_class, train_ld, test_x, test_y,
             condition_name: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        train_epoch(model, train_ld, optimizer, criterion)

    test_ld = DataLoader(
        datasets.MNIST("/tmp/mnist", train=False, download=False,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE)
    acc = eval_accuracy(model, test_ld)

    sig = measure_signals(model, test_x, test_y)

    r_rte, p_rte   = pearsonr(sig["entropy"], sig["rte"])
    rs_rte, _      = spearmanr(sig["entropy"], sig["rte"])
    r_sdiv, p_sdiv = pearsonr(sig["entropy"], sig["sdiv"])
    rs_sdiv, _     = spearmanr(sig["entropy"], sig["sdiv"])

    hard_rte_mean  = sig["rte"][sig["hard_mask"]].mean()
    easy_rte_mean  = sig["rte"][sig["easy_mask"]].mean()
    hard_sdiv_mean = sig["sdiv"][sig["hard_mask"]].mean()
    easy_sdiv_mean = sig["sdiv"][sig["easy_mask"]].mean()

    return {
        "seed":            seed,
        "condition":       condition_name,
        "acc":             float(acc),
        "r_rte":           float(r_rte),
        "p_rte":           float(p_rte),
        "rs_rte":          float(rs_rte),
        "r_sdiv":          float(r_sdiv),
        "p_sdiv":          float(p_sdiv),
        "rs_sdiv":         float(rs_sdiv),
        "ratio_rte":       float(hard_rte_mean / (easy_rte_mean + 1e-9)),
        "ratio_sdiv":      float(hard_sdiv_mean / (easy_sdiv_mean + 1e-9)),
        "signals":         {
            "entropy":   sig["entropy"].tolist(),
            "rte":       sig["rte"].tolist(),
            "sdiv":      sig["sdiv"].tolist(),
            "p_max":     sig["p_max"].tolist(),
            "correct":   sig["correct"].tolist(),
            "labels":    sig["labels"].tolist(),
        },
    }


# ── Scatter-plot ───────────────────────────────────────────────────────────────

def make_scatter(all_results: list):
    """2×2 scatter: (routing_ent, state_div) × (mHC, ctrl) — pooled over all seeds + OLS trendline."""
    if not HAS_MPL:
        print("  [OBS] matplotlib saknas — hoppar över figur")
        return

    mhc_res  = [r for r in all_results if r["condition"] == "mhc"]
    ctrl_res = [r for r in all_results if r["condition"] == "ctrl"]

    # Pool alla seeds
    m_ent    = np.concatenate([r["signals"]["entropy"] for r in mhc_res])
    m_rte    = np.concatenate([r["signals"]["rte"]     for r in mhc_res])
    m_sdiv   = np.concatenate([r["signals"]["sdiv"]    for r in mhc_res])
    m_labels = np.concatenate([r["signals"]["labels"]  for r in mhc_res])

    c_ent    = np.concatenate([r["signals"]["entropy"] for r in ctrl_res])
    c_rte    = np.concatenate([r["signals"]["rte"]     for r in ctrl_res])
    c_sdiv   = np.concatenate([r["signals"]["sdiv"]    for r in ctrl_res])
    c_labels = np.concatenate([r["signals"]["labels"]  for r in ctrl_res])

    n_seeds = len(mhc_res)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"exp_008 — micro-mHC på MNIST  (pooled {n_seeds} seeds × {EVAL_N} samples)\n"
        "Red=hard(4,7,9)  Blue=easy(0,1)  Grey=other  — dashed = OLS trendline",
        fontsize=10)

    def scatter(ax, x, y, labels, xlabel, ylabel, r, title):
        x, y, labels = np.array(x), np.array(y), np.array(labels)
        colors = np.where(np.isin(labels, list(HARD_DIGITS)), "#d62728",
                 np.where(np.isin(labels, list(EASY_DIGITS)), "#1f77b4",
                          "#aec7e8"))
        ax.scatter(x, y, c=colors, s=1.5, alpha=0.25, rasterized=True)
        # OLS trendlinje — hoppa över om y är konstant (t.ex. ctrl rte=0)
        if len(np.unique(y)) > 1:
            z      = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, np.poly1d(z)(x_line), "k--", linewidth=1.5, alpha=0.8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\nPearson r = {r:+.3f}", fontsize=9)

    mean_mhc_rte  = np.mean([r["r_rte"]  for r in mhc_res])
    mean_mhc_sdiv = np.mean([r["r_sdiv"] for r in mhc_res])
    mean_ctrl_sdiv = np.mean([r["r_sdiv"] for r in ctrl_res])

    scatter(axes[0, 0],
            m_ent, m_rte, m_labels,
            "Predictive entropy H(p)", "Routing entropy H(H_res)",
            mean_mhc_rte,
            "mHC — Routing entropy  [P1: r > 0]")

    scatter(axes[0, 1],
            m_ent, m_sdiv, m_labels,
            "Predictive entropy H(p)", "State divergence L2",
            mean_mhc_sdiv,
            "mHC — State divergence  [expected: r < 0]")

    scatter(axes[1, 0],
            c_ent, c_rte, c_labels,
            "Predictive entropy H(p)", "Routing entropy H(H_res)",
            0.0,
            "Ctrl (H=I) — Routing entropy  [constant = 0, no trendline]")

    scatter(axes[1, 1],
            c_ent, c_sdiv, c_labels,
            "Predictive entropy H(p)", "State divergence L2",
            mean_ctrl_sdiv,
            "Ctrl (H=I) — State divergence  [exp_007 replication]")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_exp_008_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figur sparad: {out_path}")


# ── Confident-wrong rate vs routing entropy bins ───────────────────────────────

CONF_THRESHOLD = 0.80   # p_max > detta = "konfident"
N_BINS         = 3      # tertiler: låg / medium / hög routing-entropi

BIN_LABELS = ["Låg routing-entropi\n(bottom 33%)",
              "Medium routing-entropi\n(33–67%)",
              "Hög routing-entropi\n(top 33%)"]


def confident_wrong_by_bins(all_results: list) -> dict:
    """
    Pooler alla seeds (mHC-betingelse) och beräknar confident-wrong rate
    per routing-entropibin (tertiler).

    confident-wrong: p_max > CONF_THRESHOLD OCH fel prediktion.

    Hypotes: hög routing-entropi → modellen tveksam → lägre confident-wrong.
    """
    mhc = [r for r in all_results if r["condition"] == "mhc"]

    all_rte = np.concatenate([np.array(r["signals"]["rte"])     for r in mhc])
    all_pmax = np.concatenate([np.array(r["signals"]["p_max"])   for r in mhc])
    all_ok   = np.concatenate([np.array(r["signals"]["correct"]) for r in mhc])

    # Tertilgränser
    q33, q67 = np.percentile(all_rte, [33.3, 66.7])
    edges = [all_rte.min(), q33, q67, all_rte.max() + 1e-9]

    cw_rates, error_rates, n_total, n_cw = [], [], [], []
    for i in range(N_BINS):
        mask = (all_rte >= edges[i]) & (all_rte < edges[i + 1])
        n  = mask.sum()
        cw = ((all_pmax[mask] > CONF_THRESHOLD) & (~all_ok[mask])).sum()
        er = (~all_ok[mask]).sum()
        cw_rates.append(float(cw / n) if n > 0 else 0.0)
        error_rates.append(float(er / n) if n > 0 else 0.0)
        n_total.append(int(n))
        n_cw.append(int(cw))

    return {
        "conf_threshold": CONF_THRESHOLD,
        "q33": float(q33),
        "q67": float(q67),
        "bin_labels": BIN_LABELS,
        "cw_rates":   cw_rates,
        "error_rates": error_rates,
        "n_total":    n_total,
        "n_cw":       n_cw,
    }


def make_cw_plot(cw_data: dict):
    """Bar chart: confident-wrong rate per routing-entropibin."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    x    = np.arange(N_BINS)
    bars = ax.bar(x, cw_data["cw_rates"], color=["#2196F3", "#FF9800", "#F44336"],
                  width=0.55, alpha=0.85)

    # Annotate med n_cw / n_total
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{cw_data['n_cw'][i]}/{cw_data['n_total'][i]}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=9)
    ax.set_ylabel(f"Confident-wrong rate\n(p_max > {cw_data['conf_threshold']})", fontsize=9)
    ax.set_title(
        "exp_008 — Confident-wrong rate vs Routing entropy bins\n"
        f"(pooled over 5 seeds, mHC condition, n={sum(cw_data['n_total'])} sample-slots)",
        fontsize=9)
    ax.set_ylim(0, max(cw_data["cw_rates"]) * 1.35 + 0.001)

    # Riktningspil
    ax.annotate("", xy=(2.25, max(cw_data["cw_rates"]) * 0.5),
                xytext=(2.25, max(cw_data["cw_rates"]) * 0.05),
                arrowprops=dict(arrowstyle="->", color="grey"))
    ax.text(2.35, max(cw_data["cw_rates"]) * 0.27, "hypotes:\nminskar →",
            fontsize=7, color="grey")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_exp_008_cw_bins.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figur sparad: {out_path}")


# ── P2: Gate-aktivt experiment ────────────────────────────────────────────────

P2_THRESHOLDS_PCT = [25, 33, 50, 67, 75]   # τ som percentil av routing-entropi


def p2_gate_experiment(all_results: list) -> dict:
    """
    Minimal P2-test: CognOS-gate med routing-entropi-tröskel.

    Prediktion P2: gate aktivt → färre confident-wrong bland auto-beslut
    jämfört med baseline (ingen gate, alla samples passerar).

    För varje tröskel τ (angiven som percentil):
      auto-pool  = samples med routing_entropy < τ     (gate: pass)
      eskalerade = samples med routing_entropy >= τ    (gate: escalate)

    Mäter per τ:
      coverage          = |auto-pool| / |total|
      precision_auto    = andel korrekta i auto-pool
      cw_rate_auto      = confident-wrong rate i auto-pool
      precision_gain    = precision_auto − baseline_precision
      cw_reduction_abs  = baseline_cw_rate − cw_rate_auto
    """
    mhc = [r for r in all_results if r["condition"] == "mhc"]

    all_rte  = np.concatenate([np.array(r["signals"]["rte"])     for r in mhc])
    all_pmax = np.concatenate([np.array(r["signals"]["p_max"])   for r in mhc])
    all_ok   = np.concatenate([np.array(r["signals"]["correct"]) for r in mhc])

    n_total = len(all_rte)

    # Baseline: ingen gate
    baseline_precision = float(all_ok.mean())
    baseline_cw        = float(((all_pmax > CONF_THRESHOLD) & (~all_ok)).mean())

    rows = []
    for pct in P2_THRESHOLDS_PCT:
        tau       = float(np.percentile(all_rte, pct))
        auto_mask = all_rte < tau
        n_auto    = int(auto_mask.sum())
        n_esc     = n_total - n_auto

        if n_auto == 0:
            continue

        prec_auto = float(all_ok[auto_mask].mean())
        cw_auto   = float(((all_pmax[auto_mask] > CONF_THRESHOLD) &
                           (~all_ok[auto_mask])).mean())

        rows.append({
            "threshold_pct":  pct,
            "tau":            tau,
            "n_auto":         n_auto,
            "n_escalated":    n_esc,
            "coverage":       float(n_auto / n_total),
            "precision_auto": prec_auto,
            "cw_rate_auto":   cw_auto,
            "precision_gain": prec_auto - baseline_precision,
            "cw_reduction":   baseline_cw - cw_auto,
        })

    return {
        "n_total":            n_total,
        "conf_threshold":     CONF_THRESHOLD,
        "baseline_precision": baseline_precision,
        "baseline_cw_rate":   baseline_cw,
        "thresholds":         rows,
    }


def make_p2_plot(p2: dict):
    """Tvåpanel: precision-gain + CW-reduction vs coverage (gate-stränghet)."""
    if not HAS_MPL:
        return

    rows     = p2["thresholds"]
    coverage = [r["coverage"]        for r in rows]
    prec_g   = [r["precision_gain"]  * 100 for r in rows]   # procentenheter
    cw_red   = [r["cw_reduction"]    * 100 for r in rows]
    pcts     = [r["threshold_pct"]   for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"exp_008 — P2 Gate Experiment  (pooled 5 seeds, mHC condition)\n"
        f"Baseline: precision={p2['baseline_precision']*100:.2f}%  "
        f"CW-rate={p2['baseline_cw_rate']*100:.3f}%  "
        f"n={p2['n_total']}",
        fontsize=10)

    # Panel 1: precision gain
    bars1 = ax1.bar(range(len(rows)), prec_g,
                    color=["#4CAF50" if v > 0 else "#F44336" for v in prec_g],
                    width=0.55, alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.8)
    for i, (bar, cov, pct) in enumerate(zip(bars1, coverage, pcts)):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"τ=p{pct}\ncov={cov:.0%}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels([f"p{r['threshold_pct']}" for r in rows], fontsize=9)
    ax1.set_ylabel("Precision gain vs baseline (pp)", fontsize=9)
    ax1.set_title("Precision in auto-pool (gate active)\nvs baseline (no gate)", fontsize=9)

    # Panel 2: CW-rate reduction
    bars2 = ax2.bar(range(len(rows)), cw_red,
                    color=["#2196F3" if v > 0 else "#FF9800" for v in cw_red],
                    width=0.55, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    for i, (bar, cov, pct) in enumerate(zip(bars2, coverage, pcts)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0001,
                 f"τ=p{pct}\ncov={cov:.0%}",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([f"p{r['threshold_pct']}" for r in rows], fontsize=9)
    ax2.set_ylabel("CW-rate reduction vs baseline (pp)", fontsize=9)
    ax2.set_title("Confident-wrong rate reduction in auto-pool\nvs baseline (no gate)", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_exp_008_p2_gate.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figur sparad: {out_path}")


# ── Huvudkörning ───────────────────────────────────────────────────────────────

def run():
    print(f"exp_008 — micro-mHC på MNIST")
    print(f"Enhet: {DEVICE}")
    print(f"Seeds: {SEEDS}  |  Epoker: {EPOCHS}  |  N_eval: {EVAL_N}")
    print()

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds  = datasets.MNIST("/tmp/mnist", train=True,  download=True,
                               transform=transform)
    test_ds   = datasets.MNIST("/tmp/mnist", train=False, download=True,
                               transform=transform)
    train_ld  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0)

    # Extrahera fasta 2000 testsamples (deterministisk ordning)
    test_ld   = DataLoader(test_ds, batch_size=EVAL_N, shuffle=False)
    test_x, test_y = next(iter(test_ld))
    test_x    = test_x.to(DEVICE)

    all_results = []

    # ── mHC-betingelse ─────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"mHC-betingelse (Birkhoff-routing, Sinkhorn-Knopp iter={SINKHORN_N})")
    print("=" * 64)
    for seed in SEEDS:
        print(f"  Seed {seed}: tränar ({EPOCHS} epoker)... ", end="", flush=True)
        r = run_seed(seed, MNISTNet_mHC, train_ld, test_x, test_y, "mhc")
        print(f"acc={r['acc']:.4f} | "
              f"r_rte={r['r_rte']:+.3f} (p={r['p_rte']:.4f}) | "
              f"r_sdiv={r['r_sdiv']:+.3f} | "
              f"ratio_rte={r['ratio_rte']:.2f}x")
        all_results.append(r)

    # ── Kontrollbetingelse ─────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("Kontrollbetingelse (H_res = I, identitetsrouting)")
    print("=" * 64)
    for seed in SEEDS:
        print(f"  Seed {seed}: tränar ({EPOCHS} epoker)... ", end="", flush=True)
        r = run_seed(seed, MNISTNet_ctrl, train_ld, test_x, test_y, "ctrl")
        print(f"acc={r['acc']:.4f} | "
              f"r_rte=N/A (konstant) | "
              f"r_sdiv={r['r_sdiv']:+.3f} | "
              f"ratio_sdiv={r['ratio_sdiv']:.2f}x")
        all_results.append(r)

    # ── Sammanfattning ─────────────────────────────────────────────────────────
    mhc  = [r for r in all_results if r["condition"] == "mhc"]
    ctrl = [r for r in all_results if r["condition"] == "ctrl"]

    mhc_rte_mean    = np.mean([r["r_rte"]  for r in mhc])
    mhc_rte_std     = np.std( [r["r_rte"]  for r in mhc])
    mhc_sdiv_mean   = np.mean([r["r_sdiv"] for r in mhc])
    mhc_sdiv_std    = np.std( [r["r_sdiv"] for r in mhc])
    ctrl_sdiv_mean  = np.mean([r["r_sdiv"] for r in ctrl])
    ctrl_sdiv_std   = np.std( [r["r_sdiv"] for r in ctrl])

    print()
    print("=" * 64)
    print("RESULTAT — medelvärde ± std över 5 seeds")
    print("=" * 64)
    print(f"\nmHC-betingelse:")
    print(f"  Routing-entropi  vs Ue:   r = {mhc_rte_mean:+.3f} ± {mhc_rte_std:.3f}")
    print(f"  Tillstånds-div   vs Ue:   r = {mhc_sdiv_mean:+.3f} ± {mhc_sdiv_std:.3f}")
    print(f"\nKontrollbetingelse (H_res=I):")
    print(f"  Routing-entropi  vs Ue:   N/A (konstant signal = 0)")
    print(f"  Tillstånds-div   vs Ue:   r = {ctrl_sdiv_mean:+.3f} ± {ctrl_sdiv_std:.3f}")

    # ── P1-test ────────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("P1-TEST")
    print("=" * 64)
    print(f"  Prediktion: mHC routing-entropi r > 0")
    print(f"  Observerat: r = {mhc_rte_mean:+.3f}")

    if mhc_rte_mean > 0.1:
        verdict = "✓ P1 STÖDS — Birkhoff-routing producerar korrekt tecken"
    elif mhc_rte_mean > 0:
        verdict = "~ P1 SVAGT STÖDD — positiv men nära noll"
    else:
        verdict = "✗ P1 EJ STÖDD — negativt/noll tecken, hypotesen håller inte"

    print(f"  Utfall: {verdict}")

    ctrl_rep = "✓ Replikerat" if ctrl_sdiv_mean < -0.1 else "~ Partiellt"
    print(f"\n  Kontroll (exp_007-replikering): r_sdiv = {ctrl_sdiv_mean:+.3f}  {ctrl_rep}")

    # ── Spara resultat ─────────────────────────────────────────────────────────
    summary = {
        "exp": "exp_008",
        "date": "2026-02-24",
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "n_eval": EVAL_N,
        "n_streams": N_STREAMS,
        "sinkhorn_iters": SINKHORN_N,
        "mhc_r_rte_mean":   float(mhc_rte_mean),
        "mhc_r_rte_std":    float(mhc_rte_std),
        "mhc_r_sdiv_mean":  float(mhc_sdiv_mean),
        "ctrl_r_sdiv_mean": float(ctrl_sdiv_mean),
        "p1_verdict":       verdict,
        "per_seed": all_results,
    }

    json_path = os.path.join(OUT_DIR, "results.json")
    # Spara utan signals (för liten fil)
    summary_slim = {k: v for k, v in summary.items() if k != "per_seed"}
    summary_slim["per_seed"] = [
        {k: v for k, v in r.items() if k != "signals"}
        for r in all_results
    ]
    with open(json_path, "w") as f:
        json.dump(summary_slim, f, indent=2, default=float)
    print(f"\n  Resultat sparade: {json_path}")

    make_scatter(all_results)

    # ── Confident-wrong rate vs routing-entropibin ─────────────────────────────
    print()
    print("=" * 64)
    print("CONFIDENT-WRONG RATE vs ROUTING-ENTROPIBIN")
    print(f"(p_max > {CONF_THRESHOLD}, pooled 5 seeds, mHC-betingelse)")
    print("=" * 64)

    cw = confident_wrong_by_bins(all_results)
    for i, label in enumerate(BIN_LABELS):
        tag = label.split("\n")[0]
        print(f"  {tag:30s}: CW-rate = {cw['cw_rates'][i]:.4f}"
              f"  ({cw['n_cw'][i]}/{cw['n_total'][i]})"
              f"  err-rate = {cw['error_rates'][i]:.4f}")

    # Trend?
    if cw["cw_rates"][2] < cw["cw_rates"][0]:
        cw_verdict = "✓ Monotont avtagande — hög routing-entropi skyddar mot confident errors"
    elif cw["cw_rates"][2] <= cw["cw_rates"][0] * 1.2:
        cw_verdict = "~ Svagt avtagande / platt"
    else:
        cw_verdict = "✗ Ingen avtagande trend"
    print(f"\n  Trend: {cw_verdict}")

    make_cw_plot(cw)
    summary["confident_wrong_bins"] = cw

    # ── P2: Gate-aktivt experiment ─────────────────────────────────────────────
    print()
    print("=" * 64)
    print("P2-GATE EXPERIMENT")
    print(f"(routing-entropi-tröskel → auto-pool, pooled 5 seeds, mHC)")
    print("=" * 64)

    p2 = p2_gate_experiment(all_results)
    print(f"\n  Baseline (ingen gate):  precision={p2['baseline_precision']*100:.2f}%  "
          f"CW-rate={p2['baseline_cw_rate']*100:.3f}%  n={p2['n_total']}")
    print(f"\n  {'τ (pct)':>8}  {'coverage':>8}  {'prec_auto':>10}  "
          f"{'prec_gain':>10}  {'cw_auto':>9}  {'cw_reduc':>9}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")
    for r in p2["thresholds"]:
        sign = "+" if r["precision_gain"] >= 0 else ""
        cw_sign = "+" if r["cw_reduction"] >= 0 else ""
        print(f"  p{r['threshold_pct']:>7}  {r['coverage']:>7.0%}  "
              f"{r['precision_auto']*100:>9.2f}%  "
              f"{sign}{r['precision_gain']*100:>8.2f}pp  "
              f"{r['cw_rate_auto']*100:>8.3f}%  "
              f"{cw_sign}{r['cw_reduction']*100:>7.3f}pp")

    # P2-utlåtande
    best = max(p2["thresholds"], key=lambda r: r["precision_gain"])
    if best["precision_gain"] > 0 and best["cw_reduction"] > 0:
        p2_verdict = (f"✓ P2 STÖDS — gate vid p{best['threshold_pct']} ger "
                      f"+{best['precision_gain']*100:.2f}pp precision, "
                      f"{best['cw_reduction']*100:.3f}pp CW-reduktion "
                      f"(coverage {best['coverage']:.0%})")
    else:
        p2_verdict = "~ P2 SVAGT / EJ STÖDD"
    print(f"\n  Utfall: {p2_verdict}")

    make_p2_plot(p2)
    summary["p2_gate"] = p2
    summary["p2_verdict"] = p2_verdict

    # Uppdatera JSON
    summary_slim["confident_wrong_bins"] = cw
    summary_slim["p2_gate"]    = {k: v for k, v in p2.items() if k != "thresholds"}
    summary_slim["p2_verdict"] = p2_verdict
    with open(json_path, "w") as f:
        json.dump(summary_slim, f, indent=2, default=float)
    print(f"  results.json uppdaterad: {json_path}")

    return summary


if __name__ == "__main__":
    run()
