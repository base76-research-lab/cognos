"""Hjälpfunktioner för exp_009."""

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
        logits = out[0]  # första element är alltid logits
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_eval(model, loader, device):
    """
    Kör modellen i eval-läge och samlar:
        probs   (N, C)   — softmax-sannolikheter
        labels  (N,)     — sanna etiketter
        ent_l1  (N,)     — routing-entropi från Layer 1 (eller enda lagret)
        ent_l2  (N,)     — routing-entropi från Layer 2 (None om ej tillgänglig)
    """
    from models.mhc import routing_entropy

    model.eval()
    all_probs, all_labels, all_ent_l1, all_ent_l2 = [], [], [], []

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits = out[0]
        H_res_1 = out[1]
        H_res_2 = out[2] if len(out) > 2 else None

        probs = F.softmax(logits, dim=1).cpu().numpy()
        ent1 = routing_entropy(H_res_1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y.numpy())
        all_ent_l1.append(ent1)

        if H_res_2 is not None:
            all_ent_l2.append(routing_entropy(H_res_2).cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ent_l1 = np.concatenate(all_ent_l1)
    ent_l2 = np.concatenate(all_ent_l2) if all_ent_l2 else None

    return probs, labels, ent_l1, ent_l2


def gate_mask(ent: np.ndarray, tau_quantile: float) -> np.ndarray:
    """
    Returnerar bool-array: True = auto (låg entropi), False = eskalera.
    tau = tau_quantile-percentilen av entropin.
    """
    tau = np.quantile(ent, tau_quantile)
    return ent <= tau


def load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback utan PyYAML
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition(":")
                v = v.strip()
                # Enkel typkonvertering
                if v.startswith("["):
                    cfg[k.strip()] = [int(x.strip()) for x in v.strip("[]").split(",")]
                elif v.replace(".", "").lstrip("-").isdigit():
                    cfg[k.strip()] = float(v) if "." in v else int(v)
                elif v.lower() in ("true", "false"):
                    cfg[k.strip()] = v.lower() == "true"
                else:
                    cfg[k.strip()] = v
        return cfg
