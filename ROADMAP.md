# CognOS — Roadmap

## Current: v0.4.0

- Reasoning Loop (multi-level epistemic meta-iteration, L0–L5)
- Confidence engine: `compute_confidence()` with epistemic/aleatoric decomposition
- Divergence semantics and assumption extraction
- Strong synthesis
- CognOS Master Protocol v1.2 (prompt-based reasoning scaffold)
- Experiments 001–004: variance profiling across frontier models

---

## v0.5.0 — Fine-tuned Reasoning (Exp 005)

**Goal:** A Mistral model fine-tuned for computational adherence to the CognOS protocol.

Exp 004 established that prompt-based CognOS produces structural adherence but not computational adherence — models simulate the confidence formula without executing it. Fine-tuning is the next intervention.

**Tasks:**
- [ ] Build verified training dataset (target: 100 examples across domains)
- [ ] Each example: math independently verified before inclusion
- [ ] Fine-tune Mistral Small or 7B via Mistral API
- [ ] Evaluate against Exp 004 test question (hospital AI deployment)
- [ ] Compare computational adherence: base model vs fine-tuned

**How to contribute:** See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## v0.6.0 — Protocol v1.2 Integration

**Goal:** Bring prompt-based and code-based CognOS into a unified interface.

- [ ] `cognos_reason()` accepts a protocol-version parameter
- [ ] Protocol v1.2 output structure maps directly to `EpistemicState`
- [ ] QUESTION RESTATEMENT field as framing-drift detection
- [ ] Cᵢ (internal coherence) as a tracked variable in all reasoning paths

---

## v1.0.0 — Stable Reasoning Layer

**Goal:** Production-ready reasoning layer for agentic AI stacks.

- [ ] Fine-tuned model released on HuggingFace
- [ ] Full protocol adherence benchmarks published
- [ ] Integration examples: LangChain, AutoGen, custom agents
- [ ] arXiv paper: "Prompt-based vs Code-based CognOS: Two Types of Epistemic Compliance"
- [ ] Documentation complete

---

## Research Experiments

| Exp | Topic | Status |
| --- | --- | --- |
| 001 | Divergence variance across models | Complete |
| 002 | Epistemic gain measurement | Complete |
| 003 | Ill-posed question detection | Complete |
| 004 | Protocol adherence across frontier models | Complete + Addenda |
| 005 | Fine-tuning for computational adherence | In progress |

---

## Known Protocol Issues

**PROCEED threshold unreachable (v1.2)**
With default coefficients (α=1.0, β=1.0, γ=0.8, δ=0.6) and Ē ∈ [0,1],
the maximum achievable Conf is sigmoid(1.0) ≈ 0.731, below the PROCEED threshold of 0.75.
Fix planned for v0.6.0: either adjust α or recalibrate thresholds.

---

*Last updated: 2026-02-23*
