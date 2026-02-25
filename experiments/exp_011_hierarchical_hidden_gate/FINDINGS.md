# exp_011 — Findings: Hierarchical Hidden-Gate Ablation

**Date:** 2026-02-25  
**Researcher:** Björn Wikström  
**Dataset:** CIFAR-10  
**Seeds in current artifact:** 2 (`0, 1`)

---

## Research Question

Does hierarchical gating (L1 + hidden gate), with and without residual filtering, improve confident-wrong control compared with closed-loop and gate-only baselines?

---

## Run Artifact Used

- Source: `results/results_all.json`
- Observed modes in artifact: `open_loop`, `closed_loop`, `gate_only`
- Not present in this artifact: `two_stage_no_residual`, `two_stage_with_residual`, `decision_criteria`

---

## Results

### Per Seed (CW-rate)

| Seed | open_loop | closed_loop | gate_only |
|------|-----------|-------------|-----------|
| 0    | 0.04000   | 0.04328     | 0.03657   |
| 1    | 0.03500   | 0.03134     | 0.02985   |

### Mean Metrics

- Mean CW-rate (open_loop): **0.03750**
- Mean CW-rate (closed_loop): **0.03731**
- Mean CW-rate (gate_only): **0.03321**
- Mean accuracy (open_loop): **0.77025**
- Mean accuracy (closed_loop): **0.74665**
- Mean accuracy (gate_only): **0.76755**
- Mean coverage (closed_loop): **0.67**
- Mean coverage (gate_only): **0.67**

### Seed-Level Directionality

- Gate-only CW better than open-loop: **2/2 seeds**
- Closed-loop CW better than open-loop: **1/2 seeds**
- Closed-loop CW worse than open-loop: **1/2 seeds**

---

## Interpretation

With the currently available artifact, gate-only is the most stable CW reducer. Closed-loop appears mixed under the same coverage level, with one improvement and one degradation versus open-loop. Accuracy is also stronger for gate-only than closed-loop in this 2-seed run.

Because two-stage outputs are missing in this exported file, the hierarchical hidden-gate ablation itself is not yet complete in this record.

---

## Provisional Conclusion

Current evidence supports **gate-only** as the best-performing mode for CW control under the observed setup. A final exp_011 conclusion requires a results artifact that includes `two_stage_no_residual` and `two_stage_with_residual` for direct comparison.
