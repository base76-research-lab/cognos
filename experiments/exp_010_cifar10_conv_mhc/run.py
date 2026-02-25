#!/usr/bin/env python3
"""
exp_010 — Conv-backbone + mHC på CIFAR-10

Betingelser:
    open_loop    — Conv → mHC_1 → mHC_2 → head
    closed_loop  — Conv → mHC_1 → Gate(τ) → mHC_2 → head
    gate_only    — Conv → mHC_1 → Gate(τ) → head

Körning:
    python3 run.py --config configs/cifar10.yaml --mode all
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TwoLayerConvMHC, GateOnlyConvNet
from data import get_cifar10_loaders
from metrics import compute_metrics
from utils import train_one_epoch, collect_eval, gate_mask, load_yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, cfg, seed):
    train_loader, _ = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()
        print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  loss={loss:.4f}")


def eval_two_layer(model, cfg, seed, tau_q):
    _, eval_loader = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"])
    probs, labels, ent_l1, ent_l2 = collect_eval(model, eval_loader, DEVICE)

    results = {}
    results["open_loop"] = compute_metrics(
        probs, labels, ent_l1, mask_auto=None,
        cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"])

    if tau_q is not None:
        mask = gate_mask(ent_l1, tau_q)
        results["closed_loop"] = compute_metrics(
            probs, labels, ent_l1, mask_auto=mask,
            cw_threshold=cfg["cw_threshold"], n_pmax_bins=cfg["n_pmax_bins"])
        if ent_l2 is not None:
            results["closed_loop"]["mean_ent_l2_auto"] = round(float(ent_l2[mask].mean()), 4)
            results["closed_loop"]["mean_ent_l2_all"]  = round(float(ent_l2.mean()), 4)
    return results


def eval_gate_only(model, cfg, seed, tau_q):
    from models.mhc import routing_entropy as re_fn
    import torch.nn.functional as F

    _, eval_loader = get_cifar10_loaders(
        batch_size=cfg["batch_size"], eval_n=cfg["eval_n"],
        seed=seed, data_dir=cfg["data_dir"])

    model.eval()
    all_probs, all_labels, all_ent = [], [], []
    with torch.no_grad():
        for x, y in eval_loader:
            logits, H_res_1 = model(x.to(DEVICE))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(y.numpy())
            all_ent.append(re_fn(H_res_1).cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ent_l1 = np.concatenate(all_ent)
    mask   = gate_mask(ent_l1, tau_q) if tau_q else None
    return compute_metrics(probs, labels, ent_l1, mask_auto=mask,
                           cw_threshold=cfg["cw_threshold"],
                           n_pmax_bins=cfg["n_pmax_bins"])


def run_seed(cfg, seed, tau_q, mode):
    print(f"\n{'='*50}")
    print(f"seed={seed}  mode={mode}  tau_q={tau_q}  device={DEVICE}")
    print(f"{'='*50}")

    result = {"seed": seed, "mode": mode, "tau_q": tau_q}

    if mode in ("open_loop", "closed_loop", "all"):
        model = TwoLayerConvMHC(cfg["hidden_dim"], cfg["n_streams"], cfg["n_classes"]).to(DEVICE)
        torch.manual_seed(seed)
        train(model, cfg, seed)
        result["two_layer"] = eval_two_layer(model, cfg, seed, tau_q)

    if mode in ("gate_only", "all"):
        model_go = GateOnlyConvNet(cfg["hidden_dim"], cfg["n_streams"], cfg["n_classes"]).to(DEVICE)
        torch.manual_seed(seed)
        train(model_go, cfg, seed)
        result["gate_only"] = eval_gate_only(model_go, cfg, seed, tau_q)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/cifar10.yaml")
    parser.add_argument("--mode",    choices=["open_loop","closed_loop","gate_only","all"], default="all")
    parser.add_argument("--tau",     type=float, default=None)
    parser.add_argument("--seeds",   type=int, nargs="+", default=None)
    parser.add_argument("--epochs",  type=int, default=None)
    args = parser.parse_args()

    cfg    = load_yaml(args.config)
    tau_q  = args.tau    if args.tau    is not None else cfg.get("tau_quantile", 0.67)
    seeds  = args.seeds  if args.seeds  is not None else cfg["seeds"]
    if args.epochs: cfg["epochs"] = args.epochs

    all_results = []
    for seed in seeds:
        r = run_seed(cfg, seed, tau_q, args.mode)
        all_results.append(r)
        os.makedirs(cfg["results_dir"], exist_ok=True)
        out = os.path.join(cfg["results_dir"], f"results_{args.mode}.json")
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n→ Sparad: {out}")

    print("\n\n=== SAMMANFATTNING ===")
    for r in all_results:
        s = r["seed"]
        if "two_layer" in r:
            ol = r["two_layer"].get("open_loop", {})
            cl = r["two_layer"].get("closed_loop", {})
            print(f"seed={s}  open_loop acc={ol.get('accuracy','?')} cw={ol.get('cw_rate','?')}")
            if cl:
                print(f"       closed_loop acc={cl.get('accuracy','?')} "
                      f"cw={cl.get('cw_rate','?')} cov={cl.get('coverage','?')}")
        if "gate_only" in r:
            go = r["gate_only"]
            print(f"       gate_only   acc={go.get('accuracy','?')} "
                  f"cw={go.get('cw_rate','?')} cov={go.get('coverage','?')}")


if __name__ == "__main__":
    main()
