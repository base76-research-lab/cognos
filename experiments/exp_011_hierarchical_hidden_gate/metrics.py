"""
exp_009 — Metriker och konfound-kontroller.

Exporterar:
    compute_metrics(logits, labels, ent_l1, pmax_bins, tau)
        → dict med acc, cw_rate, ece, coverage + confound-data

    logistic_incremental(pmax, routing_entropy, wrong)
        → {"coef_pmax", "coef_entropy", "entropy_adds_power": bool}
"""

import numpy as np
from typing import Optional


# ── Kalibrering ───────────────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray,
                                labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE: vägt medel av |acc - conf| per confidence-bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    pmax = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pmax > lo) & (pmax <= hi)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = pmax[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ── pmax-bins (konfound A) ────────────────────────────────────────────────────

def pmax_bin_stats(probs: np.ndarray,
                   labels: np.ndarray,
                   ent_l1: np.ndarray,
                   n_bins: int = 5) -> list:
    """
    Stratifierar samples i pmax-kvantiler.
    Per bin: error_rate + mean_routing_entropy.

    Om routing-entropin fortfarande predicerar fel INOM bins med liknande pmax,
    är signalen inte bara ett proxy för "lätta samples".
    """
    pmax = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    wrong = (preds != labels).astype(float)
    edges = np.quantile(pmax, np.linspace(0, 1, n_bins + 1))
    result = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pmax >= lo) & (pmax <= hi)
        if mask.sum() == 0:
            continue
        result.append({
            "pmax_lo": round(float(lo), 3),
            "pmax_hi": round(float(hi), 3),
            "n": int(mask.sum()),
            "error_rate": round(float(wrong[mask].mean()), 4),
            "mean_routing_entropy": round(float(ent_l1[mask].mean()), 4),
        })
    return result


# ── Korrelationer (konfound B) ────────────────────────────────────────────────

def confound_correlations(ent_l1: np.ndarray,
                           probs: np.ndarray) -> dict:
    """
    corr(routing_entropy_L1, pmax)  — om extrem: gateway är bara confidence-proxy
    corr(routing_entropy_L1, pred_H) — vad vi faktiskt vill se: epistemic alignment
    """
    from scipy.stats import pearsonr, spearmanr
    pmax = probs.max(axis=1)
    pred_H = -(probs * np.log(probs + 1e-8)).sum(axis=1)

    r_pmax, p_pmax = pearsonr(ent_l1, pmax)
    r_predH, p_predH = pearsonr(ent_l1, pred_H)
    rho_pmax, _ = spearmanr(ent_l1, pmax)
    rho_predH, _ = spearmanr(ent_l1, pred_H)

    return {
        "r_ent_pmax": round(float(r_pmax), 4),
        "p_ent_pmax": round(float(p_pmax), 6),
        "r_ent_predH": round(float(r_predH), 4),
        "p_ent_predH": round(float(p_predH), 6),
        "rho_ent_pmax": round(float(rho_pmax), 4),
        "rho_ent_predH": round(float(rho_predH), 4),
    }


# ── Logistisk modell (konfound C) ─────────────────────────────────────────────

def logistic_incremental(pmax: np.ndarray,
                          routing_entropy: np.ndarray,
                          wrong: np.ndarray) -> dict:
    """
    Jämför P(wrong) ~ pmax  vs  P(wrong) ~ pmax + routing_entropy.

    Returnerar koefficienter + om routing_entropy lägger till förklaringskraft
    (likelihood-ratio test, signifikant om p < 0.05).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import chi2
        import warnings

        scaler = StandardScaler()
        X_base = scaler.fit_transform(pmax.reshape(-1, 1))
        X_full = scaler.fit_transform(
            np.column_stack([pmax, routing_entropy])
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_base = LogisticRegression(max_iter=500).fit(X_base, wrong)
            m_full = LogisticRegression(max_iter=500).fit(X_full, wrong)

        ll_base = m_base.score(X_base, wrong) * len(wrong)
        ll_full = m_full.score(X_full, wrong) * len(wrong)
        lr_stat = 2 * (ll_full - ll_base)
        p_val = float(chi2.sf(lr_stat, df=1))

        return {
            "coef_pmax_base": round(float(m_base.coef_[0][0]), 4),
            "coef_pmax_full": round(float(m_full.coef_[0][0]), 4),
            "coef_entropy_full": round(float(m_full.coef_[0][1]), 4),
            "lr_stat": round(float(lr_stat), 4),
            "p_incremental": round(p_val, 6),
            "entropy_adds_power": p_val < 0.05,
        }
    except ImportError:
        return {"error": "sklearn not installed — skipping logistic model"}


# ── Huvudfunktion ─────────────────────────────────────────────────────────────

def compute_metrics(probs: np.ndarray,
                    labels: np.ndarray,
                    ent_l1: np.ndarray,
                    mask_auto: Optional[np.ndarray] = None,
                    cw_threshold: float = 0.80,
                    n_pmax_bins: int = 5) -> dict:
    """
    Komplett metriker för ett betingelse + auto-pool.

    mask_auto: bool-array — None → alla samples är auto.
    """
    if mask_auto is None:
        mask_auto = np.ones(len(probs), dtype=bool)

    n_total = len(probs)
    n_auto = int(mask_auto.sum())
    coverage = n_auto / n_total

    p_auto = probs[mask_auto]
    y_auto = labels[mask_auto]
    e_auto = ent_l1[mask_auto]

    pmax_auto = p_auto.max(axis=1)
    preds_auto = p_auto.argmax(axis=1)
    wrong_auto = (preds_auto != y_auto)

    acc = float((preds_auto == y_auto).mean()) if n_auto > 0 else float("nan")
    cw_mask = (pmax_auto > cw_threshold) & wrong_auto
    cw_rate = float(cw_mask.sum() / n_auto) if n_auto > 0 else float("nan")
    cc_mask = (pmax_auto > cw_threshold) & (~wrong_auto)
    cc_rate = float(cc_mask.sum() / n_auto) if n_auto > 0 else float("nan")
    ece = expected_calibration_error(p_auto, y_auto) if n_auto > 0 else float("nan")

    result = {
        "coverage": round(coverage, 4),
        "n_auto": n_auto,
        "n_escalated": n_total - n_auto,
        "accuracy": round(acc, 4),
        "cw_rate": round(cw_rate, 5),
        "confident_correct_rate": round(cc_rate, 5),
        "ece": round(ece, 4),
    }

    pmax_all = probs.max(axis=1)
    preds_all = probs.argmax(axis=1)
    wrong_all = preds_all != labels

    mask_pass = mask_auto
    mask_drop = ~mask_auto

    def _part(mask):
        n = int(mask.sum())
        if n == 0:
            return {
                "n": 0,
                "accuracy": None,
                "cw_rate": None,
                "confident_correct_rate": None,
            }
        wrong = wrong_all[mask]
        pmax = pmax_all[mask]
        cw = ((pmax > cw_threshold) & wrong).mean()
        cc = ((pmax > cw_threshold) & (~wrong)).mean()
        acc_p = (~wrong).mean()
        return {
            "n": n,
            "accuracy": round(float(acc_p), 4),
            "cw_rate": round(float(cw), 5),
            "confident_correct_rate": round(float(cc), 5),
        }

    result["gate_partition"] = {
        "pass": _part(mask_pass),
        "drop": _part(mask_drop),
    }

    if n_auto > 10:
        result["pmax_bins"] = pmax_bin_stats(p_auto, y_auto, e_auto, n_pmax_bins)
        result["confound_correlations"] = confound_correlations(e_auto, p_auto)
        result["logistic_incremental"] = logistic_incremental(
            pmax_auto, e_auto, wrong_auto.astype(float)
        )

    return result
