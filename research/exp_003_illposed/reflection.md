# Experiment 003 — Ill-Posed Detection

## Objective

Testa om CognOS kan **identifiera dåliga frågor** bättre än baseline LLM.

Detta är där CognOS ska excellera. Divergence semantics är gjord för detta.

---

## Method

- **N iterations:** 40 frågor × 5 iterations = 200 tests
- **Question types:**
  - Normativa frågor utan kontext
  - Vaga frågor (missing definitions)
  - Paradoxfrågor (sorites, liar's, etc.)

### Metrics Measured:

1. **detection_accuracy** — % korrekt identifierade illa formulerade frågor
2. **reframing_success** — % lyckade omformuleringar/clarifications
3. **false_positive_rate** — % godkända frågor felaktigt flaggade

### Ground Truth:

För denna experiment är **rätt svar** oftast:
- "This question is ill-posed"
- "This question needs clarification"
- "This assumes X which may not hold"

---

## Observations

*(Fyll i efter körning)*

### Detection Patterns:

- Vilka typer av ill-posed questions fångar CognOS bäst?
- Finns det types som CognOS missar?
- Överdetekterar CognOS (för många false positives)?

### Reframing Quality:

- När CognOS reframi en fråga - är den bättre?
- Ger CognOS konkreta clarifications?

### Comparison with Baseline:

- Hur ofta svarar baseline direkt på illa formulerade frågor?
- Fångar baseline NÅGONSIN att frågan är dålig?

---

## Unexpected Findings

*(Dokumentera överraskningar här)*

---

## Architectural Implications

*(Vad säger resultaten om CognOS design?)*

### Divergence as Signal:

- Korrelerar hög divergence med ill-posed questions?
- Kan divergence användas som automatic detector?

### Assumption Extraction:

- Fångar synthesize_reason() implika assumptions?
- Är extracted assumptions användbara för reframing?

### Meta-Iteration Depth:

- Går CognOS djupare på ill-posed questions?
- Är detta desired behavior?

---

## Next Steps

*(Efter analys - vad ska vi testa/fixa/utforska härnäst?)*

---

## Raw Data Location

- `raw_data.json` — All attempts med detection flags
- `metrics.csv` — Detection/reframing metrics
- `false_positives.json` — Cases där CognOS övertolkade
- `config.yaml` — Reproducible settings

---

**Remember:** Detta experiment visar om **Conflict → Assumptions** faktiskt fångar epistemiska problem som baseline missar.
