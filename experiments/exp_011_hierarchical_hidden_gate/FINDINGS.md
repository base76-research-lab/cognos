# exp_011 — Findings: Hierarchical hidden-gate ablation

**Datum:** 2026-02-25  
**Researcher:** Björn Wikström  
**Föregående:** exp_010 (CIFAR10 closed-loop vs gate-only)

---

## Fråga

Ger hierarkisk gating (L1 + hidden gate) med/utan residual-analys bättre riskseparation än gate-only och closed-loop?

---

## Beslutskriterier

### Gate_only = robust om
- CW minskar i medel
- CW-minskningen håller i minst 4/5 seeds

### Closed_loop = potentiell vinnare om
- CW minskar
- accuracy-drop inte är oproportionerlig givet coverage (acc/coverage rimlig)

### Closed_loop = training mismatch om
- gate_only stabilt förbättrar CW
- closed_loop stabilt försämrar CW

---

## Extra loggar (obligatoriska)

1. **CW bland gate-pass vs gate-drop**
2. **Confident-correct rate**

---

## Setup

- Dataset: CIFAR10
- Modes:
  - open_loop
  - closed_loop
  - gate_only
  - two_stage_no_residual
  - two_stage_with_residual
- Output-filer:
  - `results/results_all.json` (lokalt)
  - `/content/results/exp_011_*` (Colab)

---

## Resultat

_Fylls i efter körning._

### Per seed

| Seed | open CW | closed CW | gate_only CW | two_stage_no_res CW | two_stage_with_res CW |
|------|---------|-----------|--------------|---------------------|-----------------------|
|      |         |           |              |                     |                       |

### Medel

- open_loop CW:
- closed_loop CW:
- gate_only CW:
- two_stage_no_residual CW:
- two_stage_with_residual CW:

---

## Tolkning

_Fylls i efter körning._

---

## Slutsats

_Fylls i efter körning._
