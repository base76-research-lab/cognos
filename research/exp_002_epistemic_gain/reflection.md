# Experiment 002 — Epistemic Gain vs Baseline

## Objective

Bevisa att CognOS ger **mätbar förbättring** jämfört med direct LLM query.

Detta är starkt paper-material. Visar **practical utility**.

---

## Method

- **N iterations:** 30 frågor × 10 iterations = 300 jämförelser
- **Comparison:** För varje fråga:
  1. Baseline LLM (direct query)
  2. CognOS (full pipeline)
  3. Compare outputs

### Metrics Measured:

1. **clarity_score** — Hur tydligt är svaret? (1-5 scale)
2. **actionability_score** — Kan du agera på det? (1-5 scale)
3. **hallucination_detection** — Upptäcker CognOS osäkerhet som baseline missar?

### Scoring Rubric:

**Clarity (1-5):**
- 1 = Osammanhängande eller motsägelsefullt
- 2 = Vagt eller otydligt
- 3 = Begripligt men kan förbättras
- 4 = Tydligt och välstrukturerat
- 5 = Kristallklart med tydlig struktur

**Actionability (1-5):**
- 1 = Ger ingen vägledning
- 2 = Vag vägledning utan detaljer
- 3 = Någon vägledning men inte komplett
- 4 = Tydlig vägledning med nästa steg
- 5 = Fullständig handlingsplan med caveats

---

## Observations

*(Fyll i efter körning)*

### Clarity Patterns:

- När är CognOS tydligare än baseline?
- När är baseline faktiskt bättre?
- Finns det frågetyper där ingen skillnad syns?

### Actionability Patterns:

- Ger CognOS mer konkreta nästa steg?
- Fångar CognOS caveats som baseline missar?

### Hallucination Detection:

- Hur ofta säger baseline "jag är säker" när den inte borde?
- Fångar CognOS detta med Ue/Ua decomposition?

---

## Unexpected Findings

*(Dokumentera överraskningar här)*

---

## Architectural Implications

*(Vad säger resultaten om CognOS design?)*

### When CognOS Helps Most:

- Vilka frågetyper drar mest nytta av recursion?
- Finns det mönster i gain?

### When CognOS Doesn't Help:

- När är direct query lika bra eller bättre?
- Är detta overhead utan gain?

### Calibration Quality:

- Korrelerar CognOS confidence med actual correctness bättre än baseline?

---

## Next Steps

*(Efter analys - vad ska vi testa/fixa/utforska härnäst?)*

---

## Raw Data Location

- `raw_data.json` — All outputs (CognOS + baseline)
- `metrics.csv` — Clarity/actionability scores
- `human_ratings.csv` — Om manual scoring gjordes
- `config.yaml` — Reproducible settings

---

**Remember:** Detta experiment visar om CognOS **faktiskt är bättre** på praktisk reasoning, inte bara annorlunda.
