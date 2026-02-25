# exp_010 — Findings: CIFAR10 conv MHC

**Datum:** 2026-02-24
**Researcher:** Björn Wikström
**Föregående:** exp_009 (closed-loop gate på MNIST)

---

## Fråga

Kan closed-loop epistemic routing (Layer1 → Gate → Layer2 → head) förbättra beslutskvalitet på CIFAR10 jämfört med open-loop?

---

## Setup

- Modell: TwoLayerMHC (conv) — CIFAR10, 2000 testsamples, seed 0
- Betingelser: open_loop, closed_loop
- Gate: routing_entropy ≤ τ (τ = p67 kvantil → 67% coverage)
- Primärmått: accuracy, CW-rate, ECE, coverage, n_auto, n_escalated

---

## Resultat

### Open loop
- coverage: 100%
- n_auto: 2000
- n_escalated: 0
- accuracy: 0.7705
- CW-rate: 0.0400
- ECE: 0.0185

### Closed loop
- coverage: 67%
- n_auto: 1340
- n_escalated: 660
- accuracy: 0.7254
- CW-rate: 0.0433
- ECE: 0.0261

---

## Binvis (pmax)

- Error_rate sjunker med högre pmax (från 0.59 till 0.01)
- Routing entropy ökar med pmax

---

## Korrelationsdata

- r_ent_pmax: 0.2966
- r_ent_predH: -0.3174
- rho_ent_pmax: 0.3437
- rho_ent_predH: -0.3486

---

## Nyckelresultat

- Closed loop ger lägre coverage och något lägre accuracy än open loop.
- CW-rate och ECE är marginellt högre i closed loop.
- Ingen tydlig epistemic förbättring — closed loop tillför inte mätbar kvalitet över open loop.
- Logistic incremental ej körd (sklearn saknas).

---

## Slutsats


## Resultat per seed

| Seed | open_loop acc | open_loop CW | closed_loop acc | closed_loop CW | gate_only acc | gate_only CW |
|------|---------------|--------------|-----------------|----------------|---------------|--------------|
| 0    | 0.7705        | 0.0400       | 0.7254          | 0.0433         | 0.7575        | 0.0366       |
| 1    | 0.7700        | 0.0350       | 0.7679          | 0.0313         | 0.7776        | 0.0299       |
|      |               |              |                 |                |               |              |
| **Medel** | **0.7703 ± 0.0004** | **0.0375 ± 0.0035** | **0.7467 ± 0.0299** | **0.0373 ± 0.0080** | **0.7676 ± 0.0142** | **0.0333 ± 0.0047** |

### ECE (expected calibration error)

| Betingelse | Medel ECE |
|---|---|
| open_loop  | 0.0208 ± 0.0032 |
| closed_loop | 0.0259 ± 0.0004 |
| gate_only  | 0.0245 ± 0.0046 |

---

---


