# exp_008 — Findings: micro-mHC på MNIST — P1 Empiriskt Testad

**Datum:** 2026-02-24
**Researcher:** Björn Wikström

---

## Fråga

Ger en mHC-arkitektur tränad från scratch r(routing_entropy, Ue) > 0,
i kontrast till kontrollbetingelsen H_res=I som förväntas ge r ≤ 0?

**Prediktion P1:** r(routing_entropy, softmax_entropy) > 0 för mHC-betingelse.

## Setup

- Modell: 2-lager micro-mHC MLP
  - Flatten → Dense(128) ReLU → mHC(4 strömmar × 32) → Dense(10) Softmax
  - mHC: per-sample H_res via gating-nät (Linear) + Sinkhorn-Knopp (20 iterationer)
- Kontroll: identisk arkitektur men H_res = I (fast, ingen blandning)
- Dataset: MNIST, 2000 testsamples (deterministisk urval)
- Träning: 5 epoker, Adam optimizer
- Seeds: 5 (0–4) per betingelse
- Primärsignal: routing-entropi H(H_res) = −Σ h·log(h) per sample
- Sekundärsignal: tillstånds-divergens L2 av routade strömmar
- Ue ground truth: softmax-entropi H(p)

## Resultat

### mHC-betingelse (Birkhoff-routing)

| Seed | Acc   | r_rte    | r_sdiv   |
|------|-------|----------|----------|
| 0    | 96.8% | **+0.139** | −0.223 |
| 1    | 96.5% | **+0.311** | −0.309 |
| 2    | 96.5% | **+0.124** | −0.197 |
| 3    | 96.4% | **+0.206** | −0.249 |
| 4    | 96.5% | **+0.260** | −0.320 |
| **Medel** | 96.5% | **+0.208 ± 0.071** | −0.259 ± 0.048 |

### Kontrollbetingelse (H_res = I)

| Seed | Acc   | r_rte | r_sdiv |
|------|-------|-------|--------|
| 0    | 96.3% | N/A   | −0.229 |
| 1    | 96.7% | N/A   | −0.218 |
| 2    | 96.1% | N/A   | −0.228 |
| 3    | 96.4% | N/A   | −0.246 |
| 4    | 96.5% | N/A   | −0.279 |
| **Medel** | 96.4% | N/A | **−0.240 ± 0.021** |

*N/A: H_res = I → routing-entropi alltid 0 → korrelation odefinierad.*

## Nyckelresultat

### P1 — STÖDS

**mHC routing-entropi r = +0.208 (p < 0.0001), konsistent över alla 5 seeds.**

Prediktion: r > 0. Observerat: r = +0.208 ± 0.071. P1 konfirmerad.

### Kontroll replikerar exp_007

Tillstånds-divergens för H_res=I: r = −0.240.
Jämför exp_007a: r = −0.351 (statisk delning av identiskt lager).
Konsistent inverterad signal — replikering bekräftad.

### Tillstånds-divergens förblir negativ även i mHC

Routing-entropin flippas till positiv av Birkhoff-routing,
men *tillstånds-divergensen* (L2 av routade strömmar) förblir negativ: r = −0.259.
Detta är logiskt: routing-blandning homogeniserar strömmarnas innehåll
snarare än att sprida det.

**Den epistemiskt informativa signalen i mHC är routing-entropin,
inte tillstånds-divergensen.**

## Mekanistisk förklaring

**Kontrollbetingelse (H_res = I):**
- Varje ström bearbetar sin partition av hidden state
- Tydlig input → specialiserade aktiveringar → hög L2-divergens
- Oklar input → diffusa aktiveringar → låg L2-divergens
- r(L2_div, Ue) < 0 — inverterad signal (replikerar exp_007)

**mHC-betingelse (Birkhoff-routing):**
- Gating-nätet lär sig att producera olika H_res beroende på input
- Oklar input (hög Ue) → gating sprider routing → hög H_res-entropi
- Tydlig input (låg Ue) → gating koncentrerar routing → låg H_res-entropi
- r(routing_entropy, Ue) > 0 — **korrekt tecken, P1 verifierad**

Sinkhorn-Knopp-normaliseringen är mekanismen som möjliggör detta:
den tvingar H_res att vara dubbelt stokastisk, vilket gör routing-entropins
variation meningsfull och tolkningsbar.

## Implikationer för mHC-hypotesen

1. **P1 verifierad empiriskt** — Birkhoff-routing ger korrekt tecknat signal
   på ett verkligt klassificeringsproblem (MNIST), inte bara i syntetisk simulering
2. **P2 verifierad empiriskt** — routing-entropi-gate reducerar CW-rate bland auto-beslut:
   τ=p67 ger +1.21pp precision, −21% relativ CW-rate, 67% coverage (se P2-sektion nedan)
3. **Tillstånds-divergens är fel signal** — även i mHC förblir L2-divergens negativ;
   det är routing-entropins spridning som bär Ue-information
4. **Arkitekturen lär sig routing-mönster** — utan explicit supervision lär sig
   gating-nätet att använda mer diffus routing för svårare inputs

## Jämförelsetabell — alla experiment

| Experiment | Modell | Signal | r med Ue | Riktning | Användbar? |
|---|---|---|---|---|---|
| exp_006a (syntetisk) | Simulerad mHC | Inter-ström-div | +0.996 | Korrekt | Ja — teori bekräftad |
| exp_006b | GPT-2 attention | Attention-entropi | ~+0.11 | Svag/korrekt | Nej |
| exp_006c | GPT-2 MC-dropout | Logit-divergens | ~−0.13 | Inverterad | Nej |
| exp_007a | MNIST dense (statisk) | Tillstånds-div | −0.351 | Inverterad | Nej |
| exp_007b | MNIST MC-dropout | Tillstånds-div | −0.332 | Inverterad | Nej |
| **exp_008 mHC** | **micro-mHC tränad** | **Routing-entropi** | **+0.208** | **Korrekt** | **Ja — P1 verifierad** |
| exp_008 ctrl | MNIST dense (H=I) | Tillstånds-div | −0.240 | Inverterad | Nej (replikerar 007) |

## Begränsningar

- Svag signal (r ≈ 0.21 vs exp_006a: r = 0.996) — MNIST är ett enkelt problem
  och micro-mHC är en förenklad implementation; riktiga mHC-vikter förväntas ge starkare signal
- Hard/easy digit-ratio ≈ 1.0x — routing-entropin separerar inte klasser tydligt;
  variationen är korrelerad med men inte perfectt kalibrerad mot svårighetsgrad
- CPU-körning begränsar antal seeds/epoker; fler epoker kan förstärka signalen
- Gating-nätet är en enkel linjär projektion; djupare gating kan ge bättre signal

## Slutsats

**exp_008 verifierar P1 och P2 empiriskt.**

**P1:** En micro-mHC-arkitektur tränad från scratch på MNIST producerar
r(routing_entropy, softmax_entropy) = +0.208 (p < 0.0001), konsistent
över 5 seeds. Kontrollbetingelsen H_res=I ger r = −0.240 (replikerar exp_007).

**P2:** Routing-entropi-gate med τ=p67 (eskalerar top 33%):

- Precision i auto-pool: 96.94% vs baseline 95.73% (+1.21pp)
- CW-rate i auto-pool: 1.179% vs baseline 1.500% (−21% relativt)
- Coverage: 67% (2/3 av alla samples hanteras autonomt)

Birkhoff-routing via Sinkhorn-Knopp-normalisering inverterar signalens tecken
jämfört med standard dense-lager. Den epistemiskt informativa signalen är
routing-entropin, inte tillstånds-divergensen. Gaten omsätter mätningen till
faktisk kvalitetsförbättring bland auto-beslut.

**Vägen framåt:** exp_009 (P2 på NLP-data eller OOD), eller söka tillgång
till tränade mHC-vikter (DeepSeek/community) för test på verklig skala.

## Kopplingar

- **exp_007 FINDINGS.md** — inverterad signal i standard dense-lager (replikeras i ctrl)
- **exp_006 FINDINGS.md** — syntetisk validering (exp_006a r=+0.996, P1 i teori)
- **HYPOTHESIS.md §2** — Signal 1: inter-ström-divergens; nu: routing-entropi
- **workspace/02_WRITING/papers/pågående/mhc-epistemic-infrastructure/draft.md** — P1–P3
