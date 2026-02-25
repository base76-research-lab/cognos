#!/usr/bin/env python3
"""
exp_011 — Hierarchical hidden-gate ablation (CIFAR-10)

Modes:
    open_loop             — Conv -> mHC_1 -> mHC_2 -> head
    closed_loop           — open_loop with L1 gate mask for AUTO metrics
    gate_only             — Conv -> mHC_1 -> head with L1 gate mask
    two_stage_no_residual — Conv -> mHC_1 -> mHC_2 -> mHC_3 -> head with L1+L2 masks
    two_stage_with_residual — same as above + residual score gate
    all                   — run all modes in one pass per seed
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TwoLayerConvMHC, GateOnlyConvNet, TwoStageHiddenMHC
from models.mhc import routing_entropy
from data import get_cifar10_loaders
from metrics import compute_metrics
from utils import train_one_epoch, collect_eval, gate_mask, load_yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, cfg, seed):
    train_loader, _ = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"]
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    torch.manual_seed(seed)
    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()
        print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  loss={loss:.4f}")


def eval_two_layer(model, cfg, seed, tau_q):
    _, eval_loader = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"]
    )
    probs, labels, ent_l1, ent_l2 = collect_eval(model, eval_loader, DEVICE)

    results = {
        "open_loop": compute_metrics(
            probs, labels, ent_l1, mask_auto=None,
            cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"]
        )
    }

    if tau_q is not None:
        mask = gate_mask(ent_l1, tau_q)
        results["closed_loop"] = compute_metrics(
            probs, labels, ent_l1, mask_auto=mask,
            cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"]
        )
        if ent_l2 is not None:
            results["closed_loop"]["mean_ent_l2_auto"] = round(float(ent_l2[mask].mean()), 4)
            results["closed_loop"]["mean_ent_l2_all"] = round(float(ent_l2.mean()), 4)

    return results


def eval_gate_only(model, cfg, seed, tau_q):
    import torch.nn.functional as F

    _, eval_loader = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"]
    )

    model.eval()
    all_probs, all_labels, all_ent = [], [], []
    with torch.no_grad():
        for x, y in eval_loader:
            logits, H_res_1 = model(x.to(DEVICE))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y.numpy())
            all_ent.append(routing_entropy(H_res_1).cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ent_l1 = np.concatenate(all_ent)
    mask = gate_mask(ent_l1, tau_q) if tau_q is not None else None

    return compute_metrics(
        probs, labels, ent_l1, mask_auto=mask,
        cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"]
    )


def eval_two_stage(model, cfg, seed, tau_q):
    import torch.nn.functional as F

    _, eval_loader = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"]
    )

    model.eval()
    all_probs, all_labels = [], []
    all_e1, all_e2, all_e3, all_residual = [], [], [], []

    with torch.no_grad():
        for x, y in eval_loader:
            logits, H1, H2, H3, residual_score = model(x.to(DEVICE))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y.numpy())
            all_e1.append(routing_entropy(H1).cpu().numpy())
            all_e2.append(routing_entropy(H2).cpu().numpy())
            all_e3.append(routing_entropy(H3).cpu().numpy())
            all_residual.append(residual_score.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    e1 = np.concatenate(all_e1)
    e2 = np.concatenate(all_e2)
    e3 = np.concatenate(all_e3)
    residual = np.concatenate(all_residual)

    tau_hidden = cfg.get("tau_hidden_quantile", tau_q)
    residual_q = cfg.get("residual_quantile", 0.67)

    mask_l1 = gate_mask(e1, tau_q)
    mask_l2 = gate_mask(e2, tau_hidden)
    mask_no_res = mask_l1 & mask_l2
    residual_tau = np.quantile(residual, residual_q)
    mask_with_res = mask_no_res & (residual <= residual_tau)

    no_res_metrics = compute_metrics(
        probs, labels, e1, mask_auto=mask_no_res,
        cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"]
    )
    with_res_metrics = compute_metrics(
        probs, labels, e1, mask_auto=mask_with_res,
        cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"]
    )

    no_res_metrics["hidden_gate"] = {
        "tau_hidden_quantile": tau_hidden,
        "mean_e2": round(float(e2.mean()), 4),
        "mean_e3": round(float(e3.mean()), 4),
        "residual_tau": round(float(residual_tau), 4),
    }

    with_res_metrics["hidden_gate"] = {
        "tau_hidden_quantile": tau_hidden,
        "residual_quantile": residual_q,
        "residual_tau": round(float(residual_tau), 4),
        "residual_mean": round(float(residual.mean()), 4),
    }

    return {
        "two_stage_no_residual": no_res_metrics,
        "two_stage_with_residual": with_res_metrics,
    }


def decision_criteria(all_results):
    seeds = []
    gate_improves = 0
    closed_improves = 0
    closed_worsens = 0

    for r in all_results:
        seeds.append(r.get("seed"))
        ol = r.get("two_layer", {}).get("open_loop", {})
        cl = r.get("two_layer", {}).get("closed_loop", {})
        go = r.get("gate_only", {})

        cw_open = ol.get("cw_rate")
        cw_closed = cl.get("cw_rate")
        cw_gate = go.get("cw_rate")

        if cw_open is not None and cw_gate is not None and cw_gate < cw_open:
            gate_improves += 1
        if cw_open is not None and cw_closed is not None:
            if cw_closed < cw_open:
                closed_improves += 1
            elif cw_closed > cw_open:
                closed_worsens += 1

    n = max(len(seeds), 1)
    return {
        "n_seeds": len(seeds),
        "gate_only_robust": gate_improves >= 4,
        "gate_only_improve_count": gate_improves,
        "closed_loop_potential": closed_improves >= 3,
        "closed_loop_improve_count": closed_improves,
        "closed_loop_worsen_count": closed_worsens,
        "training_mismatch_signal": gate_improves >= 4 and closed_worsens >= 4,
    }


def run_seed(cfg, seed, tau_q, mode):
    print(f"\n{'='*50}")
    print(f"seed={seed}  mode={mode}  tau_q={tau_q}  device={DEVICE}")
    print(f"{'='*50}")

    result = {"seed": seed, "mode": mode, "tau_q": tau_q}

    if mode in ("open_loop", "closed_loop", "all"):
        model = TwoLayerConvMHC(cfg["hidden_dim"], cfg["n_streams"], cfg["n_classes"]).to(DEVICE)
        train(model, cfg, seed)
        result["two_layer"] = eval_two_layer(model, cfg, seed, tau_q)

    if mode in ("gate_only", "all"):
        model_go = GateOnlyConvNet(cfg["hidden_dim"], cfg["n_streams"], cfg["n_classes"]).to(DEVICE)
        train(model_go, cfg, seed)
        result["gate_only"] = eval_gate_only(model_go, cfg, seed, tau_q)

    if mode in ("two_stage_no_residual", "two_stage_with_residual", "all"):
        model_ts = TwoStageHiddenMHC(cfg["hidden_dim"], cfg["n_streams"], cfg["n_classes"]).to(DEVICE)
        train(model_ts, cfg, seed)
        ts = eval_two_stage(model_ts, cfg, seed, tau_q)
        if mode == "two_stage_no_residual":
            result["two_stage_no_residual"] = ts["two_stage_no_residual"]
        elif mode == "two_stage_with_residual":
            result["two_stage_with_residual"] = ts["two_stage_with_residual"]
        else:
            result.update(ts)

    return result


def main():
    parser = argparse.ArgumentParser(description="exp_011 hierarchical hidden-gate ablation")
    parser.add_argument("--config", default="configs/cifar10.yaml")
    parser.add_argument(
        "--mode",
        choices=[
            "open_loop", "closed_loop", "gate_only",
            "two_stage_no_residual", "two_stage_with_residual", "all"
        ],
        default="all",
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tau_q = args.tau if args.tau is not None else cfg.get("tau_quantile", 0.67)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if args.epochs:
        cfg["epochs"] = args.epochs

    all_results = []
    for seed in seeds:
        r = run_seed(cfg, seed, tau_q, args.mode)
        all_results.append(r)

        os.makedirs(cfg["results_dir"], exist_ok=True)
        out_path = os.path.join(cfg["results_dir"], f"results_{args.mode}.json")
        payload = {
            "config": cfg,
            "mode": args.mode,
            "tau_q": tau_q,
            "results": all_results,
            "decision_criteria": decision_criteria(all_results),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n→ Sparad: {out_path}")

    print("\n\n=== SAMMANFATTNING ===")
    for r in all_results:
        s = r.get("seed")
        ol = r.get("two_layer", {}).get("open_loop", {})
        cl = r.get("two_layer", {}).get("closed_loop", {})
        go = r.get("gate_only", {})
        tsn = r.get("two_stage_no_residual", {})
        tsr = r.get("two_stage_with_residual", {})

        if ol:
            print(f"seed={s}  open_loop acc={ol.get('accuracy','?')} cw={ol.get('cw_rate','?')}")
        if cl:
            print(f"       closed_loop acc={cl.get('accuracy','?')} cw={cl.get('cw_rate','?')} cov={cl.get('coverage','?')}")
        if go:
            print(f"       gate_only   acc={go.get('accuracy','?')} cw={go.get('cw_rate','?')} cov={go.get('coverage','?')}")
        if tsn:
            print(f"       two_stage(no_res) acc={tsn.get('accuracy','?')} cw={tsn.get('cw_rate','?')} cov={tsn.get('coverage','?')}")
        if tsr:
            print(f"       two_stage(+res)   acc={tsr.get('accuracy','?')} cw={tsr.get('cw_rate','?')} cov={tsr.get('coverage','?')}")

    print("\n=== BESLUTSKRITERIER ===")
    dc = decision_criteria(all_results)
    print(json.dumps(dc, indent=2))


if __name__ == "__main__":
    main()
