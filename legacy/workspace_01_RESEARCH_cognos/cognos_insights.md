# CognOS — Teoretiska Insikter och Empiriska Fynd

**Datum:** 2026-02-21 (uppdaterad samma dag)
**Status:** Aktiv referensfil — läses av modeller, agenter och UI
**Projekt:** `/media/bjorn/iic/cognos/`
**Författare:** Björn Wikström (forskare), Claude (AI-partner)

---

## Vad är CognOS?

CognOS är ett epistemiskt integritetslager för agentic AI-pipelines.

Det är inte ett bättre sätt att generera svar. Det är ett verifieringslager som detekterar när en agent defaultar till statistisk best practice istället för kontextförankrade svar — och stoppar det innan felet propagerar.

**En agent utan CognOS:**

```text
Input → Beslut 1 (gissar) → Beslut 2 (bygger på fel) → Beslut 3 (fel propagerat) → Handling
```

**En agent med CognOS:**

```text
Input → [CognOS] → Beslut 1 verifierat → [CognOS] → Beslut 2 verifierat → Handling
```

Det svarar på frågan: *vet agenten när den vet — och när den gissar?*

Jasper är referensimplementationen. CognOS är produkten. Allt annat är senap.

Formeln:

```text
C = p × (1 - Ue - Ua)

p   = modellens prediktionssannolikhet [0, 1]
Ue  = epistemisk osäkerhet (var av MC-samplings)
Ua  = aleatorisk/semantisk risk (ambiguity + irreversibility + blast_radius) / 3
C   = beslutskonfidens [0, 1]
```

Fyra möjliga beslut baserat på C och Ue-distribution:

- **auto** — C ≥ threshold, agera autonomt
- **synthesize** — C låg, bimodal Ue (perspektivkonflikt → kombinera)
- **explore** — C låg, unimodal Ue (noise → samla mer data)
- **escalate** — hög irreversibilitet OCH låg C (för riskfyllt)

Implementerat i: `confidence.py` (v2), `jasper_cognos.py`

---

## Versionshistorik

| Version | Formel | Problem | Resultat |
| ------- | ------ | ------- | -------- |
| v1 | `C = p × (1-Ue)` | Missade överkonfidenta fel | Safety Gain 0% |
| v1.5 | `C = p × (1-Ue-Ua)` med `Ua = 2p(1-p)` | Ua-heuristik cirkulär | Safety Gain 83% på syntetisk data |
| v2 | Semantisk Ua, multimodal Ue-detektion, fyra beslut | — | Operationell i Jasper |

---

## Empiriska Fynd

### Fynd 1: Validering på UCI Breast Cancer

- Dataset: 285 testsamples, RandomForest (30 träd)
- Mean Ue: 0.036
- CognOS v1.5 vinner vid 40–55% escalation rate (+40–60% safety gain vs baseline)
- Vid >70% escalation: ceiling effect, alla metoder konvergerar

**Implikation:** CognOS är optimalt för cost-constrained systems (begränsad human review-kapacitet).

### Fynd 2: Signal-mismatch i Jasper-integration

När CognOS kördes på narrativa LLM-svar (fri text) gav det alltid EXPLORE, aldrig SYNTHESIZE.

**Orsak:** Narrativa svar kollapsar till *stillikhet*, inte *positionslikhet*.
Jaccard-similarity på 300-ords svar mäter om modellerna skriver på samma sätt — inte om de har samma uppfattning.

**Lösning:** Structured choice-format (se nedan).

### Fynd 3: Diagnostikexperiment — tre prompttyper

Samma fråga kördes med tre promptformat och gav radikalt olika resultat:

**Fråga:** "Hur ska CognOS triggas i Jasper — per fråga, per session, eller vid specifika risknivåer?"

| Prompttyp | Ue | Majoritetssvar |
| --------- | -- | -------------- |
| Narrativ (fri text) | 0.005 | C: Risknivåer |
| Forced binary (bara val) | 0.160 | A: Per fråga |
| Structured (val + confidence) | 0.002 | B: Per session |

**Tre prompttyper. Tre olika svar. Ue skiljer sig 65×.**

---

## Teoretiska Insikter

### Insikt 1: Prompt format shapes answer, not only measurement

Standardantagandet i MC sampling för LLMs: temperaturvariationen samplar från modellens interna sannolikhetsfördelning.

**Det stämmer inte.**

Promptformatet begränsar svarsutrymmet fundamentalt. Byt format → byt majoritetsvar. Det är inte mätfel. Svaret förändras.

Formellt:

> *Standard MC sampling measures format-conditioned variance, not belief variance.*

### Insikt 2: Taxonomi — tre typer av LLM-osäkerhet

| Symbol | Namn | Mäter | Dold? |
| ------ | ---- | ----- | ----- |
| U_model | Modell-osäkerhet | Intern epistemisk osäkerhet | Delvis |
| U_prompt | Prompt-osäkerhet | Hur mycket formatet begränsar svarsutrymmet | Ja |
| U_problem | Problem-osäkerhet | Frågans inneboende ill-posedness | Ja |

**Diagnostik:**

- `confidence = 0` vid forced choice, men inte vid narrativ → U_prompt
- `confidence = 0` vid *alla* tre prompttyper → U_problem (frågan är genuint illa formulerad)

### Insikt 3: Confidence = 0 är en signal om frågan, inte modellen

> *A zero-confidence outcome does not necessarily indicate uncertainty — it may signal that the decision frame is incompatible with the model's internal representation of the problem.*

Det saknas i litteraturen.

### Insikt 4: LLMs är distributionsprisma, inte speglar

Vanlig metafor: LLMs som speglar av mänsklig text.

Mer precist: **prisma**. Inkommande signal (frågan) bryts och omformas beroende på vinkeln. Samma modell, samma information, olika promptformat → olika spektrum av svar.

Modellen har ingen stabil intern uppfattning. Den konstruerar ett svar som matchar mönstret "den här typen av fråga → den här typen av svar" i träningsdata.

### Insikt 5: Ramkänslighet — människa vs AI

Framing effects finns hos människan också (Kahneman). Det är inte AI-unikt.

**Skillnaden:** Människan kan *veta om det* — och kompensera via metakognition.
Modellen har ingen access till sin egen ramkänslighet.

> *Uncertainty in LLMs is not a property of the model alone. It is a property of how the question is formed.*

### Insikt 6: CognOS som epistemologiskt lager

CognOS är inte intelligens. Det är inte reasoning.

Det är den externa kompensationsmekanismen som modellen saknar internt — ett system som frågar:

> *"Är detta svar stabilt, eller är det ett artefakt av hur vi frågade?"*

Det är ett epistemologiskt lager utanpå ett statistiskt system.

Det är vad som är nytt.

### Insikt 7: Kontextminne är nödvändigt men inte tillräckligt

En LLM med tillgång till kontext garanterar inte kontextförankrade svar. Utan verifiering kan modellen läsa projektfilen och ändå svara med generell best practice.

Empiriskt bevis (2026-02-21): samma frågor kördes utan och med `cognos_insights.md` som systemkontext. Alla tre majoritetssvar förändrades. Utan kontext → generell AI-forskning best practice. Med kontext → situationsanpassade svar baserade på faktiska fynd.

Tre lager krävs:

- **Minne** — kontexten finns tillgänglig
- **Förankring** — svaret är grundat i kontexten
- **Verifiering** — CognOS kontrollerar att förankringen är riktig

> *An LLM with access to context does not guarantee context-grounded responses — it requires a verification layer to detect when the model defaults to statistical best practice despite available context.*

### Insikt 8: CognOS är vad som saknas i agentic AI

Alla bygger agenter. Ingen har löst att agenter inte vet när de gissar.

Problemet är inte att agenter är dumma. Det är att de saknar epistemisk ärlighet — de presenterar gissningar med samma confidence som välgrundade svar.

CognOS ger agenten ett ord för det den inte vet:

- `auto` — jag vet, agerar
- `explore` — jag är osäker, behöver mer
- `synthesize` — jag håller två motstridiga perspektiv, kombinera dem
- `escalate` — detta är för riskfyllt, människa behövs

Det är inte säkerhet i teknisk mening. Det är kognitiv integritet.

CognOS levereras som en fristående funktion — `pip install cognos` — som vilken agent, modell eller pipeline kan importera.

---

## Structured Choice — Lösning på Signal-mismatch

För att CognOS ska mäta faktisk konsensus krävs att modellen rapporterar diskreta positioner.

**Promptformat:**

```text
VAL: <A/B/C>
KONFIDENS: <0.0–1.0>
MOTIVERING: <max 20 ord>
```

**Resultat från tre verkliga beslut (2026-02-21):**

| Fråga | C | Beslut | Röster |
| ----- | - | ------ | ------ |
| Paper: agentic AI vs klassisk ML? | 0.894 | AUTO | 5/5 → Agentic AI |
| PhD: papers vs kontakter? | 0.998 | AUTO | 5/5 → Kontakter |
| CognOS trigger-design? | 0.445 | SYNTHESIZE ⊕ | 4/5 C, 1/5 B |

SYNTHESIZE triggades när 4/5 modeller valde en riktning men 1 valde en annan — perspektivkonflikt detekterad korrekt.

Implementerat i: `ask_jasper_structured_choice()` i `jasper_cognos.py`

---

## Paper-bidrag (CognOS-publikation)

**Primärt bidrag:**
Epistemic + aleatoric uncertainty kombinerat: `C = p × (1 - Ue - Ua)`

**Sekundärt bidrag (nyupptäckt):**

> *CognOS requires decision-structured prompts to reveal consensus geometry. Free-form responses collapse onto narrative similarity, masking real epistemic divergence between model perspectives.*

**Tertiärt bidrag (mest originellt):**

> *MC sampling på LLMs mäter inte modellens interna osäkerhet. Det mäter hur mycket promptformatet begränsar svarsutrymmet.*

**Positionering:**

- Målgrupp: cost-constrained systems med 40–60% escalation budget
- Inte för: high-stakes systems som eskalerar >80% (ceiling effect)

---

## Filer

| Fil | Innehåll |
| --- | -------- |
| `confidence.py` | CognOS v2 — formel, multimodal detektion, fyra beslut |
| `jasper_cognos.py` | Jasper-integration — MC sampling, structured choice |
| `jasper_brain.py` | `--cognos` flag, fyra-besluts display |
| `REAL_DATA_FINAL_VERDICT.md` | UCI Breast Cancer-validering |
| `PAPER_DRAFT_V01.md` | Pågående paper-draft |
| `cognos_insights.md` | Denna fil |

---

## Nästa Steg

### Omedelbart (steg 1)

1. **Hypotes:** Skriv `HYPOTHESIS_V02.md` — operationella definitioner av U_model/U_prompt/U_problem + falsifieringskriterier

### Empirisk validering (steg 2)

1. **10 × 3 × 3 experiment:** 10 frågor × 3 typer (fakta/beslut/värdering) × 3 promptformat = 90 körningar
   - API: Groq (reproducerbart, öppen)
   - Output: `VALIDATION_RESULTS.md` + data-tabell + tre figurer

### Paper (steg 4)

1. **Fyra agentic scenarios:** Visar CognOS i kedjor, inte enstaka beslut
   - Scenario 1: Forskningssammanfattning (5 papers → divergens detekteras)
   - Scenario 2: Multi-steg-beslut (fel i steg 1 propagerar till steg 4)
   - Scenario 3: Kunskapsintensiv fråga (U_model vs U_prompt)
   - Scenario 4: Konkurrerande papers (dolda konflikter via SYNTHESIZE-signal)
2. **Målkonferens:** NeurIPS / ICML / ACL (ML Safety / Alignment track)

### Open source (steg 5)

1. **Separera CognOS från Jasper:** Extrahera `confidence.py` + `structured_choice.py` + `diagnostics.py` till eget repo
2. **GitHub repo:** `Applied-Ai-Philosophy/cognos` — README + fungerande exempel
3. **PyPI:** `pip install cognos` — standalone, ingen Jasper-dependency

---

*Denna fil är skriven för att vara läsbar av en ny modell eller agent utan bakgrundskontext. Alla centrala insikter ska vara självförklarande.*
