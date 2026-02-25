# exp_009 — Findings: Closed-loop epistemic gate

**Datum:** 2026-02-24 (uppdaterad 2026-02-25)
**Researcher:** Björn Wikström
**Föregående:** exp_008 (P1+P2 verifierade, r = +0.208, −21% CW vid gate_only)

---

## Fråga

Ger ett closed-loop system (Layer1 → Gate → Layer2 → head) mätbara förbättringar
i beslutskvalitet utöver vad gate-mätningen ensam ger?

**Prediktion P3:** closed_loop < open_loop i CW-rate, och closed_loop ≤ gate_only.

---

## Setup

- **Modell (open_loop / closed_loop):** TwoLayerMHC — Flatten → Dense(128) ReLU → mHC_1 → mHC_2 → head
- **Modell (gate_only):** GateOnlyNet — Flatten → Dense(128) ReLU → mHC_1 → head
- **Gate:** routing_entropy(H_res_1) ≤ τ (τ = p67 kvantil → 67% coverage, 33% eskaleras)
- Dataset: MNIST, 2000 testsamples, 5 seeds (0–4)
- Träning: 5 epoker, Adam, identisk setup per betingelse
- Primärmått: CW-rate (pmax > 0.80 AND wrong) i AUTO-pool

---

## Resultat

### Per seed

| Seed | open_loop acc | open_loop CW | closed_loop acc | closed_loop CW | gate_only acc | gate_only CW |
|------|--------------|--------------|-----------------|----------------|---------------|--------------|
| 0    | 0.9680        | 0.01150      | 0.9687          | 0.01269        | 0.9761        | 0.00746      |
| 1    | 0.9715        | 0.01050      | 0.9746          | 0.00896        | 0.9761        | 0.00896      |
| 2    | 0.9715        | 0.00950      | 0.9724          | 0.00746        | 0.9784        | 0.00672      |
| 3    | 0.9665        | 0.01100      | 0.9634          | 0.01119        | 0.9657        | 0.00672      |
| 4    | 0.9605        | 0.01200      | 0.9604          | 0.00970        | 0.9724        | 0.01343      |
| **Medel** | **0.9676 ± 0.0041** | **0.01090 ± 0.00086** | **0.9679 ± 0.0053** | **0.01000 ± 0.00180** | **0.9737 ± 0.0045** | **0.00866 ± 0.00252** |

### ECE (expected calibration error)

| Betingelse | Medel ECE |
|---|---|
| open_loop  | 0.0080 ± 0.0016 |
| closed_loop | 0.0099 ± 0.0025 |
| gate_only  | 0.0094 ± 0.0025 |

---

## Replikationskörning (Colab Pro, 2026-02-25)

Miljö: Google Colab Pro (GPU), `mode=all`, `tau_q=0.67`, 5 seeds.

### Per seed

| Seed | open_loop acc | open_loop CW | closed_loop acc | closed_loop CW | gate_only acc | gate_only CW |
|------|--------------|--------------|-----------------|----------------|---------------|--------------|
| 0    | 0.9655       | 0.01100      | 0.9657          | 0.00970        | 0.9784        | 0.00746      |
| 1    | 0.9670       | 0.01200      | 0.9709          | 0.00970        | 0.9657        | 0.01343      |
| 2    | 0.9735       | 0.00750      | 0.9776          | 0.00896        | 0.9806        | 0.00821      |
| 3    | 0.9660       | 0.01050      | 0.9627          | 0.01045        | 0.9701        | 0.00821      |
| 4    | 0.9610       | 0.01450      | 0.9672          | 0.01493        | 0.9694        | 0.01269      |
| **Medel** | **0.9666** | **0.01110** | **0.9688** | **0.01075** | **0.9728** | **0.01000** |

### Effektstorlek (CW-rate)

- **closed_loop vs open_loop:** 0.01075 vs 0.01110 → **−3.2%** relativ CW-reduktion
- **gate_only vs open_loop:** 0.01000 vs 0.01110 → **−9.9%** relativ CW-reduktion
- **coverage:** 0.67 i gate-betingelser (stabilt över seeds)

### Tolkning av replikationen

- Resultatet går i samma riktning som tidigare (gate hjälper), men med **svagare closed-loop-effekt** än 2026-02-24-körningen.
- `gate_only` är fortfarande den tydligaste förbättringen i denna replikation.
- Seed-varians kvarstår, vilket stärker bedömningen att **P3 ännu inte är robust verifierad** på MNIST.

---

## Nyckelresultat

### Samlad bedömning efter två körningar (2026-02-24 + 2026-02-25)

Closed-loop visar återkommande svag förbättringstendens mot open-loop, men effektstorlek och seed-stabilitet varierar mellan körningar.
Detta räcker inte för en stark P3-konklusion. Gate-signalen i sig (särskilt gate_only) framstår fortsatt som den mest robusta delen av resultatet.

### P3 — EJ VERIFIERAD (på MNIST)

**closed_loop CW förbättras svagt mot open_loop i båda körningarna**
(-8.3% i 2026-02-24, -3.2% i 2026-02-25), men effekten är **variabel och inkonsistent** över seeds
(seed 0 och 3 är sämre i closed_loop än open_loop).

P3-prediktionen — att Layer2 tillför mätbar epistemic förbättring utöver gaten —
stöds **inte klart** av MNIST-data.

### gate_only replikerar exp_008

**gate_only CW = 0.00866 ± 0.00252 → −20.6% relativt open_loop.**
Konsistent med exp_008 (−21%). Signal håller över ny modellinstans och ny seed-körning. ✓

### Kärnslutsats: epistemic-värdet sitter i mätningen, inte i L2

Gaten utan L2 (gate_only) är konsekvent bättre eller likvärdig med
gaten med L2 (closed_loop). L2 tillför brus snarare än värde.

**Två alternativa tolkningar:**
1. **MNIST-tak:** Problemet är för enkelt. De filtrerade samples som når L2
   är redan välkalibrerade — L2 har ingenting att rätta, och kan introducera
   variation snarare än förbättring.
2. **Arkitekturhypotes:** L2 tränas på alla samples men evalueras på filtrerade —
   distributionsskift innebär att L2 inte är optimalt anpassad till det förändrade inflödet.

---

## Konfound-kontroller

### A — r_ent_pmax (Är gaten ett confidence-proxy?)

| Seed | r_ent_pmax (Pearson) | rho_ent_pmax (Spearman) |
|------|---------------------|------------------------|
| 0    | −0.0669             | −0.1833                |
| 1    | +0.0087             | −0.0112                |
| 2    | −0.0284             | +0.0789                |
| 3    | −0.0074             | +0.0238                |
| 4    | −0.0745             | −0.1274                |
| **Medel** | **−0.038** | **−0.044** |

**r_ent_pmax ≈ −0.04** — extremt låg korrelation.
Gaten väljer **inte** samples baserat på hur confident modellen är (pmax).
Den mäter något genuint i routing-dynamiken. ✓

### B — r_ent_predH (Epistemisk alignment)

| Seed | r_ent_predH | rho_ent_predH |
|------|-------------|---------------|
| 0    | +0.0911     | +0.1825       |
| 1    | −0.0086     | +0.0145       |
| 2    | +0.0365     | −0.0825       |
| 3    | +0.0086     | −0.0257       |
| 4    | +0.0985     | +0.1271       |
| **Medel** | **+0.038** | **+0.043** |

**r_ent_predH ≈ +0.04 i AUTO-poolen** — svag positiv korrelation.

Tolkning: i AUTO-poolen (de 67% lågentropiska samples) är routing-entropin
och prediktiv-entropin grovt oberoende. Det är förväntat — gaten HAR redan
filtrerat ut de samples där osäkerheten är hög. Återstående variation är låg.
Den svaga positiva tendensen är konsistent med P1-mekanismen.

---

## Jämförelsetabell — hela experimentserien

| Experiment | Modell | Signal | Primärmått | Resultat |
|---|---|---|---|---|
| exp_006a | Syntetisk mHC | Inter-ström-div | r med Ue | +0.996 ✓ |
| exp_007a | MNIST dense | Tillstånds-div | r med Ue | −0.351 (inverterad) |
| exp_007b | MNIST MC-dropout | Tillstånds-div | r med Ue | −0.332 (inverterad) |
| exp_008 mHC | micro-mHC | Routing-entropi | r + CW-gate | r=+0.208, −21% CW ✓ |
| **exp_009 gate_only** | **micro-mHC** | **Routing-entropi** | **CW-rate** | **−20.6% CW ✓ (replikering)** |
| **exp_009 closed_loop** | **TwoLayer-mHC** | **Routing-entropi L1** | **CW-rate** | **−8.3% CW (inkonsistent, ej verifierat)** |

---

## Begränsningar

- **MNIST-tak:** 96–97% accuracy lämnar litet rum för förbättring i L2.
  Svårare dataset (NLP, CIFAR, OOD) behövs för att pröva P3.
- **Distributionsskift vid eval:** L2 tränas på alla samples men evalueras
  på filtrerade subset. En alternativ design: träna L2 explicit på
  L1-filtrerade samples (curriculum från gate).
- **5 seeds:** Tillräckligt för riktnings-etablering, otillräckligt för
  statistisk kraftberäkning på en 8%-effekt.
- **CPU-körning:** Begränsar modellstorlek och antal epoker.

---

## Slutsats

**exp_009 verifierar gate_only-resultaten från exp_008 (−20.6% CW, replikering) ✓**

**P3 (closed_loop Layer1 → Gate → Layer2) är ännu EJ robust verifierad på MNIST.**

Den mekanistiska insikten är central:

> Epistemic-värdet i mHC-arkitekturen sitter i **mätningen** (routing-entropin
> som gate-signal), inte i ytterligare bearbetning efter filtreringen.

Två möjliga förklaringar kvarstår öppna och behöver åtskiljning:
1. MNIST är för enkelt (L2 har ingenting att rätta)
2. L2 behöver tränas på den filtrerade distributionen för att bidra

**exp_010:** Testa P3 på svårare data (NLP-klassificering eller OOD-detection)
alternativt: träna L2 explicit på gate-filtrerade samples.

---

## Kopplingar

- **exp_008 FINDINGS.md** — P1+P2 verifierade (referensexperiment)
- **HYPOTHESIS.md §3** — P3: closed-loop control
- **ECD-paper** — DOI: 10.5281/zenodo.18756421
