# CognOS — Projektplan och Roadmap

**Skapad:** 2026-02-21
**Status:** Aktiv
**Läses av:** Modeller, agenter, Björn

---

## Nuläge (vad som är gjort)

### Implementerat ✅

| Komponent | Fil | Status |
|-----------|-----|--------|
| Konfidensformel v2 | `confidence.py` | Klar |
| Multimodal Ue-detektion | `confidence.py` | Klar |
| Fyra beslut (auto/synthesize/explore/escalate) | `confidence.py` | Klar |
| Jasper-integration (--cognos flag) | `jasper_brain.py` | Klar |
| MC sampling med temperaturvariationer | `jasper_cognos.py` | Klar |
| Structured choice-läge | `jasper_cognos.py` | Klar |
| Batch-komprimering (undviker rate limit) | `jasper_cognos.py` | Klar |
| Normalisering av Jaccard-likheter | `jasper_cognos.py` | Klar |
| Diagnostikprotokoll (tre prompttyper) | Experiment | Klar |
| Teoretiska insikter dokumenterade | `cognos_insights.md` | Klar |

### Empiriska fynd ✅

- CognOS v1.5 ger 40–60% högre safety gain vid matchad escalation (40–55%)
- Structured choice löser signal-mismatch (narrativa svar → stillikhet, inte positionslikhet)
- Tre promptformat ger radikalt olika svar och Ue (varierar 65×)
- Modell utan projektkontext svarar med "best practice" — inte situationsanpassat
- confidence=0 är signal om U_problem, inte U_model

---

## Kärn-hypotes (v0.1)

> LLM uncertainty-mätningar reflekterar inte modellens belief distribution.
> De reflekterar promptformat-conditionad distribution.

> En LLM med tillgång till kontext garanterar inte kontextförankrade svar.
> Det krävs ett verifieringslager för att detektera när modellen defaultar till
> statistisk best practice trots tillgänglig kontext.

**Tre separerbara osäkerhetskällor:**
- **U_model** — intern epistemisk osäkerhet
- **U_prompt** — format-inducerad varians
- **U_problem** — frågans inneboende ill-posedness

**CognOS säger om hypotesen (2026-02-21):**
- Falsifierbar? Delvis (B, 5/5, C=0.99) — kärnan stark, taxonomin behöver operationell definition
- Största svaghet? Empirisk grund för smal (A, 5/5, C=1.00) — ett experiment, en modell
- Redo för experiment? Ja (A, 5/5, C=1.00)

---

## Vad som behöver göras — i rätt ordning

### Steg 1 — Skärp hypotesen (30–60 min)

**Vad:** Lägg till operationella definitioner av U_model / U_prompt / U_problem.

Konkret: vad mäter vi som *bevis* för att ett utfall beror på U_prompt och inte U_model?

```
U_model:  Ue varierar med modellens träning, inte med promptformat
U_prompt: Ue varierar med promptformat på identiska frågor och modeller
U_problem: Ue är hög vid ALLA tre promptformat för samma fråga
```

**Output:** `HYPOTHESIS_V02.md` — max 300 ord, inkl. falsifieringskriterier.

---

### Steg 2 — Empirisk validering (1–2 dagar)

**Vad:** Kör diagnostikexperimentet systematiskt.

**Design:**
- 10 frågor × 3 typer × 3 promptformat = 90 körningar
- Frågetyper: fakta / beslut / värdering
- Promptformat: narrativ / forced binary / structured choice
- Modell: llama-3.1-8b-instant via Groq (reproducerbart)
- Mät: Ue per cell, majoritetssvar, confidence

**Hypotesen falsifieras om:**
- Ue inte varierar systematiskt med promptformat (U_prompt existerar inte)
- Majoritetssvar är stabila över format (ingen format-sensitivity)

**Output:** `VALIDATION_RESULTS.md` + data-tabell + tre figurer

**Verktyg:** Befintlig `ask_jasper_structured_choice()` + nytt experiment-script

---

### Steg 3 — Ompositionering (parallellt med steg 2)

**Vad:** Uppdatera hur CognOS beskrivs — internt och externt.

**Ny positionering:**

> CognOS är ett kontext-sensitivt metakognitivt verifieringslager.
> Det stoppar semantisk autopilot — detekterar när en LLM defaultar till
> statistisk best practice istället för kontextförankrade svar.

**Inte bara:**
- Decision gate
- Uncertainty engine
- Prompt analyzer

**Utan:** Verification layer för kontextförankring.

**Output:** Uppdaterad `PROJEKTBESKRIVNING.md` + LinkedIn-draft

---

### Steg 4 — Paper (3–4 dagar, efter validering)

**Målkonferens:** NeurIPS / ICML / ACL (ML Safety / Alignment track)
**Alternativ:** arXiv preprint → workshop

**Struktur:**
1. Introduction — problemet med format-conditionad uncertainty
2. Background — MC sampling, conformal prediction, befintliga metoder
3. Taxonomi — U_model / U_prompt / U_problem
4. Diagnostikprotokoll — metod för separation
5. Experiment — 10 × 3 × 3 design, resultat
6. CognOS — praktisk implementation av verifieringslagret
7. Implications — varför befintlig kalibrering kan vara felkalibrerad
8. Limitations och Future Work

**Nyckelcitat (redan formulerade):**
> *"Standard MC sampling measures format-conditioned variance, not belief variance."*
> *"A zero-confidence outcome may signal that the decision frame is incompatible
>  with the model's internal representation of the problem."*
> *"An LLM with access to context does not guarantee context-grounded responses."*

**Output:** `PAPER_DRAFT_V02.md` — ~4000 ord

---

## Vad CognOS säger om ordningen (2026-02-21, med projektkontext)

| Fråga | Svar | C |
|-------|------|---|
| Störst impact? | Ompositionering | 0.544 EXPLORE |
| Genomförbart nu? | Empirisk validering | 1.000 AUTO |
| Vad först? | Paper-hypotes | 0.544 EXPLORE |

**Tolkning:** AUTO på empirisk validering = kör det. EXPLORE på resten = mer information behövs innan beslut låses.

---

## Kontext-regel (viktig för alla modeller som läser detta)

Kör aldrig CognOS-frågor utan att ladda `cognos_insights.md` som systemkontext.

Utan kontext → modellen svarar med "best practice för AI-forskning"
Med kontext → modellen svarar situationsanpassat

Detta är inte bara en praktisk regel — det är projektets huvudtes demonstrerad i praktiken.

```python
insights = Path('/media/bjorn/iic/cognos/cognos_insights.md').read_text()
system = f"Du är vetenskaplig rådgivare för CognOS.\n\n{insights}\n\nSvara i angivet format."
```

---

## Filer — aktuell status

| Fil | Innehåll | Status |
|-----|---------|--------|
| `confidence.py` | CognOS v2-formel | Klar |
| `jasper_cognos.py` | Jasper-integration | Klar |
| `cognos_insights.md` | Teoretiska insikter, referens | Klar |
| `ROADMAP.md` | Denna fil | Aktiv |
| `HYPOTHESIS_V02.md` | Skärpt hypotes | Behövs |
| `VALIDATION_RESULTS.md` | Experimentresultat | Behövs |
| `PAPER_DRAFT_V02.md` | Uppdaterat paper | Behövs |
| `PAPER_DRAFT_V01.md` | Gammal draft | Obsolet |
| `PROJEKTBESKRIVNING.md` | Gammal beskrivning | Obsolet |
| `JASPER_INTEGRATION_PLAN.md` | Genomförd | Arkivera |

---

---

## Full Vision (uppdaterad 2026-02-21)

> CognOS som ett epistemiskt integritetslager för agentic AI-pipelines —
> levererat som öppen funktion till alla som bygger agenter.

**Tre delar som hänger ihop:**

| Del | Vad | Varför |
|-----|-----|--------|
| Paper | Scenarios + tester + data | Akademisk legitimitet |
| Kod | GitHub-repo, `pip install cognos` | Adoption |
| Data | Öppna testresultat på GitHub | Reproducerbarhet |

---

## Steg 5 — Open Source Release

**Vad:**

```
cognos/
├── confidence.py          ← Kärnformeln (redan klar)
├── structured_choice.py   ← Extracted från jasper_cognos.py
├── diagnostics.py         ← Diagnostikprotokoll (U_model/U_prompt/U_problem)
├── examples/
│   ├── basic_usage.py
│   ├── agentic_pipeline.py
│   └── paper_scenarios.py
├── data/                  ← Öppna testresultat
│   └── validation_results.jsonl
├── README.md
└── pyproject.toml         ← pip install cognos
```

**Minsta möjliga release (MVP):**
- `confidence.py` + `structured_choice.py` på GitHub
- README med ett fungerande exempel
- Länk från paper

**Full release:**
- PyPI-paket (`pip install cognos`)
- Dokumentation
- Reproducerbara experiment-scripts

---

## Steg 6 — Paper Scenarios (agentic pipelines)

Paper behöver minst 3 scenarios som visar CognOS i agentic kontext — inte bara enstaka beslut utan kedjor där ett fel propagerar.

**Scenario 1: Forskningssammanfattning**
Agent sammanfattar 5 papers → utan CognOS: U_prompt dominerar, svar speglar promptformat
→ med CognOS: divergens detekteras, syntes rekommenderas

**Scenario 2: Multi-steg-beslut**
Agent planerar ett projekt i 4 steg → fel i steg 1 propagerar till steg 4
→ CognOS fångar U_problem i steg 1 innan det sprider sig

**Scenario 3: Kunskapsintensiv fråga**
Agent svarar på medicinsk/vetenskaplig fråga utan och med domänkontext
→ CognOS visar skillnaden mellan U_model (modellen vet inte) och U_prompt (fel ram)

**Scenario 4: Jämförelse av konkurrerande papers**
Agent hittar dolda oenigheter i litteraturen som ingen explicit formulerat
→ CognOS exponerar underliggande konflikter via perspektivkonflikt-signal (SYNTHESIZE)

---

## Varför detta är starkt

Alla bygger agenter. Ingen har löst att agenter inte vet när de gissar.

CognOS gör en sak: det ger agenten **epistemisk ärlighet**.

> Agenten ska veta när den vet — och när den gissar.

Det är inte ett säkerhetsverktyg i teknisk mening.
Det är ett kognitivt integritetslager.

Och det levereras som en funktion alla kan importera.

*Denna fil uppdateras efter varje genomfört steg.*
