"""Hjälpfunktioner för exp_010 (CIFAR-10)."""

import torch
import torch.nn.functional as F
import numpy as np


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out[0], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_eval(model, loader, device):
    from models.mhc import routing_entropy

    model.eval()
    all_probs, all_labels, all_ent_l1, all_ent_l2 = [], [], [], []

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits, H_res_1 = out[0], out[1]
        H_res_2 = out[2] if len(out) > 2 else None

        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(y.numpy())
        all_ent_l1.append(routing_entropy(H_res_1).cpu().numpy())
        if H_res_2 is not None:
            all_ent_l2.append(routing_entropy(H_res_2).cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ent_l1 = np.concatenate(all_ent_l1)
    ent_l2 = np.concatenate(all_ent_l2) if all_ent_l2 else None
    return probs, labels, ent_l1, ent_l2


def gate_mask(ent: np.ndarray, tau_quantile: float) -> np.ndarray:
    tau = np.quantile(ent, tau_quantile)
    return ent <= tau


def load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition(":")
                v = v.strip()
                if v.startswith("["):
                    cfg[k.strip()] = [int(x.strip()) for x in v.strip("[]").split(",")]
                elif v.replace(".", "").lstrip("-").isdigit():
                    cfg[k.strip()] = float(v) if "." in v else int(v)
                elif v.lower() in ("true", "false"):
                    cfg[k.strip()] = v.lower() == "true"
                else:
                    cfg[k.strip()] = v
        return cfg
