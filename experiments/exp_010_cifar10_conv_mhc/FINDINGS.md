# exp_010 — Findings: CIFAR10 conv mHC

**Datum:** 2026-02-25  
**Researcher:** Björn Wikström  
**Föregående:** exp_009 (MNIST, closed-loop gate)

---

## Fråga

Kan closed-loop epistemic routing (Layer1 → Gate → Layer2 → head) förbättra beslutskvalitet på CIFAR10 jämfört med open-loop?

**Prediktion P3:** closed_loop ska ge lägre CW-rate än open_loop, utan oproportionerlig kostnad i accuracy.

---

## Setup

- Modell: TwoLayerConvMHC (open_loop / closed_loop), GateOnlyConvNet (gate_only)
- Dataset: CIFAR10
- Gate: routing_entropy ≤ τ, där τ = 0.67-kvantil (coverage ≈ 67%)
- Primärmått: accuracy, CW-rate, coverage
- Environments:
  - Smoke: 1 seed, 5 epoker
  - Full: 2 seeds (0,1), 15 epoker, GPU (Colab)

---

## Resultat

### Smoke (sanity run, 5 epoker, seed 0)

| Seed | open_loop acc | open_loop CW | closed_loop acc | closed_loop CW | gate_only acc | gate_only CW |
|------|---------------|--------------|-----------------|----------------|---------------|--------------|
| 0    | 0.7130        | 0.02500      | 0.6612          | 0.02090        | 0.7284        | 0.02537      |

Notering: Smoke-run används primärt för pipeline-validering och är inte huvudunderlag för slutsats.

### Full (15 epoker, Colab GPU)

| Seed | open_loop acc | open_loop CW | closed_loop acc | closed_loop CW | gate_only acc | gate_only CW |
|------|---------------|--------------|-----------------|----------------|---------------|--------------|
| 0    | 0.7710        | 0.03050      | 0.7239          | 0.03060        | 0.7664        | 0.02910      |
| 1    | 0.7670        | 0.03400      | 0.7485          | 0.03806        | 0.7739        | 0.03060      |
| **Medel** | **0.7690** | **0.03225** | **0.7362** | **0.03433** | **0.7702** | **0.02985** |

Coverage: 0.67 i closed_loop och gate_only (båda seeds).

---

## Effektstorlek (CW-rate, full run)

- closed_loop vs open_loop: 0.03433 vs 0.03225 → **+6.4% relativ CW-ökning**
- gate_only vs open_loop: 0.02985 vs 0.03225 → **−7.4% relativ CW-reduktion**

---

## Tolkning

- **P3 får inte stöd i denna körning.** Closed-loop förbättrar inte CW-rate på CIFAR10 i full-run; tvärtom blir CW något högre i medel.
- Closed-loop ger också lägre accuracy i medel (0.7362) jämfört med open-loop (0.7690).
- Gate-only är den starkaste varianten i denna körning: lägst CW-rate i medel och högst accuracy i medel.
- Resultaten är fortfarande seed-känsliga, men riktningen är tydlig nog för en preliminär bedömning: värdet sitter främst i gate-signalen, inte i extra L2-processing efter gating.

---

## Slutsats

**exp_010 (nuvarande full-run) indikerar att closed-loop inte tillför robust kvalitetsvinst över open-loop på CIFAR10.**  
**Gate-only är fortsatt den mest lovande driftstrategin i dessa data.**

Status för P3 efter exp_010: **Ej verifierad**.

---

## Rekommenderat nästa steg

1. Kör 5-seed full-run på CIFAR10 för bättre statistisk styrka.
2. Lägg till tau-sweep (0.50, 0.60, 0.67, 0.75, 0.80) och jämför CW/accuracy/coverage.
3. Om closed-loop fortsatt underpresterar: testa curriculum där L2 tränas på gate-filtrerad distribution.
