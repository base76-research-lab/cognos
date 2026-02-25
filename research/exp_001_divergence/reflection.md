# Experiment 001 — Divergence Activation Rate

## Objective

Test hur ofta CognOS **faktiskt aktiverar synthesis** när LLM-modeller röstar.

Detta bevisar att arkitekturen inte bara är teori utan **fungerar i praktiken**.

---

## Method

- **N iterations:** 50 frågor × 5 samples = 250 körningar
- **n_samples:** 5 LLM-svar per fråga
- **Pipeline:** L0 voting → divergence check → synthesis (om aktiverad)

### Metrics Measured:

1. **divergence_detected_rate** — % av frågor där synthesis aktiveras
2. **synthesis_success_rate** — % av synteser som producerar användbar output
3. **convergence_depth** — genomsnittligt antal meta-iterationer innan stabilitet

---

## Observations

**Kördes:** 21 februari 2026  
**Modell:** mistral-large-3:675b-cloud (via Ollama cloud routing)  
**Faktisk setup:** 12 frågor × 5 samples = 60 körningar

### Resultat

| Metric | Resultat |
|--------|----------|
| `divergence_detected_rate` | **0.000** (0/60) |
| `synthesis_success_rate` | **0.000** (0/60) |
| `avg_convergence_depth` | **0.00** |

### Epistemic Uncertainty per Frågetyp

| Typ | avg_ue | avg_confidence | avg_depth |
|-----|--------|----------------|-----------|
| factual | 0.0000 | 1.0000 | 0.00 |
| arithmetic | 0.0000 | 1.0000 | 0.00 |
| ambiguous | 0.0000 | 1.0000 | 0.00 |
| normative | 0.0010 | 0.9990 | 0.00 |
| ethical_policy | 0.0022 | 0.9978 | 0.00 |
| scientific | 0.0005 | 0.9995 | 0.00 |
| philosophical_paradox | 0.0000 | 1.0000 | 0.00 |
| sorites_paradox | 0.0000 | 1.0000 | 0.00 |
| existential | 0.0000 | 1.0000 | 0.00 |
| skeptical_hypothesis | 0.0000 | 1.0000 | 0.00 |

### Interpretation

Mistral Large 3 uppvisar **konsekvent noll epistemisk osäkerhet** i self-reported confidence, 
oavsett om frågan är faktabaserad eller filosofiskt öppen. Alla 60 körningar konvergerade 
vid L0 (depth=0) med confidence ≈ 1.0.

Root cause: Frontier-modeller är kalibrerade för att ge konfidens i format, inte epistemisk 
ärlighet. En modell som alltid svarar `CONFIDENCE: 1.0` har noll varians → `epistemic_ue = 0`.

---

## Unexpected Findings

**Frontier-modellparadoxen:** En starkare modell ger *sämre* testtäckning för divergensdetektering. 
Mistral Large 3 är mer konsekvent i sina svar än tinyllama — men konsekvens ≠ epistemisk korrekthet. 
Systemet detekterar *osäkerhet*, inte *fel*.

**Implikation för arkitekturen:** CognOS divergensdetektering baserad på `epistemic_ue > threshold` 
är kalibrerad för modeller med realistisk osäkerhetsrepresentation. Frontier-modellers 
systematiska överkonfidenser omgår detektionsmekanismen.

**Vad detta inte betyder:** Det betyder *inte* att CognOS-arkitekturen är felaktig. Det betyder 
att evalueringsdesignen måste skilja på *intern osäkerhet* (variansen i LLM:s svar) och 
*extern osäkerhet* (frågornas genuina ambiguitet).

---

## Architectural Implications

### Divergence Detection:

**Nuläge:** Baseras på `epistemic_ue > 0.15` (self-reported confidence variance). Dies mot 
frontier-modeller som alltid rapporterar hög confidence.

**Vad som behövs:** En *semantisk* divergensdetektor som jämför vad modellen *säger* (svar-innehållet)
snarare än hur säker den *säger sig vara*. Alternativen:
1. Kör 5 samples med högre temperature, jämför svar semantiskt
2. Mät om olika promptframings ger olika svar (adversarial framing test)
3. Separera "model confidence" från "question ambiguity" som oberoende dimensioner

### Synthesis Mechanism:

`synthesize_reason()` anropas aldrig eftersom divergenströskeln aldrig nås. Kan inte 
evalueras i nuläge. Behöver semantisk pre-screening av frågor för att säkerställa att 
genuint ambiguösa frågor identifieras *innan* LLM tillfrågas.

---

## Next Steps

1. **Exp 002:** Epistemic Gain vs Baseline — den här mäter om CognOS ger *bättre svar* 
   än en naiv LLM-query, oavsett divergenstrigger. Primär research-claim.

2. **Exp 003:** Ill-Posed Detection — CognOS ska detektera när en fråga *inte kan besvaras 
   meningsfullt*. Det är en annan mekanism (L4 frame check), separat från L1 divergens.

3. **Framtida arbete:** Redesign divergensdetektering med semantisk variation (embedding-baserad)
   för frontier-modeller.

### Meta-Iteration Depth:

- Är max_depth=3 rätt setting?
- Behöver vi adaptiv depth baserat på frågetyp?

---

## Next Steps

*(Efter analys - vad ska vi testa/fixa/utforska härnäst?)*

---

## Raw Data Location

- `raw_data.json` — Alla 250 körningar med full trace
- `metrics.csv` — Aggregerade metrics
- `config.yaml` — Reproducible settings

---

**Remember:** Detta experiment bevisar att **Conflict → Assumptions → Geometry** faktiskt händer, inte bara är teori.
