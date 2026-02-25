# HYPOTHESIS_V02 — Operationalisering av osäkerhetskällor

**Datum:** 2026-02-21  
**Status:** Kandidathypotes för empirisk testning (10×3×3)

## Kärnhypotes
LLM-osäkerhet i beslutsfrågor är en blandning av tre separerbara komponenter:
- **U_model** (intern epistemisk osäkerhet)
- **U_prompt** (formatinducerad osäkerhet)
- **U_problem** (frågans inneboende ill-posedness)

CognOS ska kunna skilja dessa tillräckligt väl för att förbättra beslutsgating jämfört med enbart prediktionssannolikhet.

## Operationella definitioner
- **U_model:** Varians som kvarstår när promptformat hålls konstant men sampling varierar (temperatur/seed). Mått: inom-format varians i svar/konfidens.
- **U_prompt:** Varians som uppstår när fråga + modell hålls konstant men format ändras (narrativ, forced binary, structured choice). Mått: mellan-format skillnad i majoritetsval och/eller Ue.
- **U_problem:** Stabil hög osäkerhet oavsett format. Mått: låg konfidens i samtliga format + ingen robust majoritet.

## Prediktioner
1. Samma fråga ger signifikant större mellan-format-varians än inom-format-varians för en icke-trivial andel frågor (U_prompt finns).
2. Structured choice minskar mätartefakter och ger stabilare konsensusgeometri än fri text.
3. Frågor med genuin tvetydighet visar persistenta lågkonfidensutfall i alla format (U_problem).

## Falsifieringskriterier
Hypotesen förkastas helt eller delvis om något av följande gäller i 10×3×3-experimentet:
1. **Ingen formatkänslighet:** mellan-format-varians ≈ inom-format-varians för nästan alla frågor.
2. **Ingen val-instabilitet:** majoritetsval förblir oförändrat mellan format i nästan alla fall.
3. **Ingen robust U_problem-signal:** låg konfidens uppträder inte konsistent över alla format för ill-posed frågor.

## Beslutsregel för CognOS
- Dominant mellan-format-varians ⇒ flagga **U_prompt-risk**.
- Persistenta lågkonfidensutfall över format ⇒ flagga **U_problem-risk**.
- Låg inom-format-stabilitet under fixerat format ⇒ flagga **U_model-risk**.