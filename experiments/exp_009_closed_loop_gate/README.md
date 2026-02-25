# exp_009 — Closed-loop epistemic gate (mHC routing entropy)

**Goal:** Test whether a closed-loop gate triggered by mHC routing entropy reduces
confident-wrong errors and improves calibration *beyond mere sample selection.*

---

## Experimental conditions

| Mode | Architecture | Gate |
|------|-------------|------|
| `open_loop` | mHC Layer1 → mHC Layer2 → head | None |
| `closed_loop` | mHC Layer1 → Gate(τ) → mHC Layer2 → head | routing_entropy_L1 ≤ τ |
| `gate_only` | mHC Layer1 → Gate(τ) → head | routing_entropy_L1 ≤ τ (exp_008-analog) |

**closed_loop:** samples with routing_entropy > τ are ESCALATED (removed from AUTO pool).
Layer 2 only receives low-entropy (confident) samples.

---

## Key outputs

- **Coverage** — fraction not escalated (AUTO pool size)
- **Accuracy** (AUTO pool)
- **CW-rate** (AUTO pool): pmax > 0.80 AND wrong
- **ECE** (AUTO pool): expected calibration error

---

## Confound controls (selection vs. signal)

To show the gate is *not* just selecting easy samples:

**A — Within-confidence bins:**
`pmax_bins` in results JSON: error_rate + mean_routing_entropy per pmax quantile.
If routing entropy still predicts error *within* bins of similar pmax → signal is independent.

**B — Correlations:**
- `corr(routing_entropy_L1, pmax)` — if extreme (|r| > 0.9): gate is a confidence proxy
- `corr(routing_entropy_L1, pred_entropy_L2)` — epistemic alignment across layers

**C — Logistic incremental model:**
`P(wrong) ~ pmax + routing_entropy`
Routing entropy should add explanatory power beyond pmax (likelihood-ratio test, p < 0.05).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# All conditions (default tau=0.67)
python run.py --config configs/mnist.yaml --mode all

# Single condition
python run.py --config configs/mnist.yaml --mode closed_loop --tau 0.67

# Tau sweep
bash scripts/run_sweep.sh
```

Results are written to `results/` as JSON.

---

## Connection to theory

exp_009 tests **P3** from ECD (Epistemic Circuit Dynamics):

> Closed-loop epistemic control (Layer → Gate → Layer) produces measurable
> improvement in decision quality compared to open-loop (Layer → Layer → Layer).

- **exp_006:** Synthetic validation (r = +0.996, P1 in theory)
- **exp_007:** Inverted signal in standard dense layers (r ≈ −0.35)
- **exp_008:** micro-mHC gives correct sign (r = +0.208, P1+P2 verified empirically)
- **exp_009:** Does the gate improve what Layer 2 sees? → P3

DOI (ECD paper): [10.5281/zenodo.18756421](https://doi.org/10.5281/zenodo.18756421)
