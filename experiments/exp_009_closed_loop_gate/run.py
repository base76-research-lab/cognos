#!/usr/bin/env python3
"""
exp_009 — Closed-loop epistemic gate (mHC routing entropy)

Betingelser:
    open_loop    — mHC Layer1 → mHC Layer2 → head (ingen gate)
    closed_loop  — mHC Layer1 → Gate(τ) → mHC Layer2 → head (eskalerar hög entropi)
    gate_only    — mHC Layer1 → Gate(τ) → head (exp_008-analog)

Körning:
    python run.py --config configs/mnist.yaml --mode open_loop
    python run.py --config configs/mnist.yaml --mode closed_loop --tau 0.67
    python run.py --config configs/mnist.yaml --mode gate_only --tau 0.67
    python run.py --config configs/mnist.yaml --mode all  # kör alla tre
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.optim as optim

# Gör exp_009-mappen importerbar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TwoLayerMHC, GateOnlyNet
from data import get_mnist_loaders
from metrics import compute_metrics
from utils import train_one_epoch, collect_eval, gate_mask, load_yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Träning ───────────────────────────────────────────────────────────────────

def train(model, cfg, seed):
    train_loader, _ = get_mnist_loaders(
        batch_size=cfg["batch_size"],
        eval_n=cfg["eval_n"],
        seed=seed,
        data_dir=cfg["data_dir"],
    )
    optimizer = optim.Adam(model.parameters())
    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        print(f"  epoch {epoch+1}/{cfg['epochs']}  loss={loss:.4f}")


# ── Eval per betingelse ───────────────────────────────────────────────────────

def eval_condition(model, cfg, seed, tau_q):
    _, eval_loader = get_mnist_loaders(
        batch_size=cfg["batch_size"],
        eval_n=cfg["eval_n"],
        seed=seed,
        data_dir=cfg["data_dir"],
    )
    probs, labels, ent_l1, ent_l2 = collect_eval(model, eval_loader, DEVICE)

    results = {}

    # open_loop: alla samples, ingen gate
    results["open_loop"] = compute_metrics(
        probs, labels, ent_l1,
        mask_auto=None,
        cw_threshold=cfg["cw_threshold"],
        n_pmax_bins=cfg["n_pmax_bins"],
    )

    if tau_q is not None:
        mask = gate_mask(ent_l1, tau_q)

        # closed_loop: gate på L1-entropi, L2-output för auto-samples
        results["closed_loop"] = compute_metrics(
            probs, labels, ent_l1,
            mask_auto=mask,
            cw_threshold=cfg["cw_threshold"],
            n_pmax_bins=cfg["n_pmax_bins"],
        )

        # L2-entropi-statistik om tillgänglig
        if ent_l2 is not None:
            results["closed_loop"]["mean_ent_l2_auto"] = round(
                float(ent_l2[mask].mean()), 4
            )
            results["closed_loop"]["mean_ent_l2_all"] = round(
                float(ent_l2.mean()), 4
            )

    return results, ent_l1


def eval_gate_only(model, cfg, seed, tau_q):
    """GateOnlyNet — enkellagers mHC med gate (exp_008-replikering)."""
    _, eval_loader = get_mnist_loaders(
        batch_size=cfg["batch_size"],
        eval_n=cfg["eval_n"],
        seed=seed,
        data_dir=cfg["data_dir"],
    )
    from models.mhc import routing_entropy as re_fn
    import torch.nn.functional as F

    model.eval()
    all_probs, all_labels, all_ent = [], [], []
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(DEVICE)
            logits, H_res_1 = model(x)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y.numpy())
            all_ent.append(re_fn(H_res_1).cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ent_l1 = np.concatenate(all_ent)

    mask = gate_mask(ent_l1, tau_q) if tau_q else None
    return compute_metrics(
        probs, labels, ent_l1,
        mask_auto=mask,
        cw_threshold=cfg["cw_threshold"],
        n_pmax_bins=cfg["n_pmax_bins"],
    )


# ── Huvud ──────────────────────────────────────────────────────────────────────

def run_seed(cfg, seed, tau_q, mode):
    print(f"\n{'='*50}")
    print(f"seed={seed}  mode={mode}  tau_q={tau_q}")
    print(f"{'='*50}")

    seed_results = {"seed": seed, "mode": mode, "tau_q": tau_q}

    if mode in ("open_loop", "closed_loop", "all"):
        model = TwoLayerMHC(
            hidden_dim=cfg["hidden_dim"],
            n_streams=cfg["n_streams"],
            n_classes=cfg["n_classes"],
        ).to(DEVICE)
        torch.manual_seed(seed)
        train(model, cfg, seed)
        res, _ = eval_condition(model, cfg, seed, tau_q)
        seed_results["two_layer"] = res

    if mode in ("gate_only", "all"):
        model_go = GateOnlyNet(
            hidden_dim=cfg["hidden_dim"],
            n_streams=cfg["n_streams"],
            n_classes=cfg["n_classes"],
        ).to(DEVICE)
        torch.manual_seed(seed)
        train(model_go, cfg, seed)
        seed_results["gate_only"] = eval_gate_only(model_go, cfg, seed, tau_q)

    return seed_results


def main():
    parser = argparse.ArgumentParser(description="exp_009 — closed-loop epistemic gate")
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--mode", choices=["open_loop", "closed_loop", "gate_only", "all"],
                        default="all")
    parser.add_argument("--tau", type=float, default=None,
                        help="Gate-kvantil (0–1). Override för cfg.tau_quantile.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds att köra (override för cfg.seeds).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tau_q = args.tau if args.tau is not None else cfg.get("tau_quantile", 0.67)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]

    all_results = []
    for seed in seeds:
        result = run_seed(cfg, seed, tau_q, args.mode)
        all_results.append(result)

        # Skriv efter varje seed (incremental output)
        os.makedirs(cfg["results_dir"], exist_ok=True)
        out_path = os.path.join(cfg["results_dir"], f"results_{args.mode}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n→ Sparad: {out_path}")

    print("\n\n=== SAMMANFATTNING ===")
    for r in all_results:
        seed = r["seed"]
        if "two_layer" in r:
            ol = r["two_layer"].get("open_loop", {})
            cl = r["two_layer"].get("closed_loop", {})
            print(f"seed={seed}  open_loop acc={ol.get('accuracy','?')} "
                  f"cw={ol.get('cw_rate','?')}")
            if cl:
                print(f"         closed_loop acc={cl.get('accuracy','?')} "
                      f"cw={cl.get('cw_rate','?')} "
                      f"coverage={cl.get('coverage','?')}")
        if "gate_only" in r:
            go = r["gate_only"]
            print(f"         gate_only   acc={go.get('accuracy','?')} "
                  f"cw={go.get('cw_rate','?')} "
                  f"coverage={go.get('coverage','?')}")


if __name__ == "__main__":
    main()
