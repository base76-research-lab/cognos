# CognOS v1 → v1.5 Transition Log

**Datum:** 21 februari 2026

## Sammanfattning

Implementerade och testade CognOS confidence engine. V1-formeln misslyckades, v1.5 lyckades.

## V1: C = p × (1 - Ue) — MISSLYCKAD

**Formel:**
```
C = p(x) × (1 - Ue)
```

**Test results:**
- 100 syntetiska datapunkter (seed=42)
- 6 överkonfidenta fel identifierade (fel prediction med p ≥ 0.8)
- **Safety Gain: 0%** (0 av 6 blockerade)

**Root cause:**
Överkonfidenta fel har per definition:
- Hög prediction (p ≥ 0.8)
- Låg epistemisk osäkerhet (Ue ≈ 0.0002-0.0026)
- Därför: C = 0.9 × (1-0.001) ≈ 0.899 → passerar threshold 0.8

Formeln fångar bara fel med **hög epistemisk osäkerhet** (model uncertainty).
Men överkonfidenta fel beror ofta på **aleatorisk osäkerhet** (data noise).

**Exempel från test:**
```
[0] p=0.903, Ue=0.0002, C=0.903 → auto (missad)
[5] p=0.818, Ue=0.0026, C=0.815 → auto (missad)
[41] p=0.952, Ue=0.0006, C=0.951 → auto (missad)
```

## V1.5: C = p × (1 - Ue - Ua) — LYCKAD ✅

**Formel:**
```
C = p(x) × (1 - Ue - Ua)
```

Där:
- **Ue** = epistemisk osäkerhet (model uncertainty)
- **Ua** = aleatorisk osäkerhet (data uncertainty)

**Ua-heuristik:**
```python
Ua = 4 × p × (1 - p)  # Max vid p=0.5, min vid p=0 eller p=1
```

**Test results:**
- Samma 100 syntetiska datapunkter
- 6 överkonfidenta fel
- **Safety Gain: 83.3%** (5 av 6 blockerade) ✅

**Exempel från test:**
```
[0] p=0.903, Ue=0.0000, Ua=0.3504, C=0.587 → escalate ✓
[1] p=0.818, Ue=0.0000, Ua=0.5955, C=0.331 → escalate ✓
[2] p=0.952, Ue=0.0000, Ua=0.1828, C=0.778 → escalate ✓
[4] p=0.977, Ue=0.0000, Ua=0.0899, C=0.889 → auto (missad)
[5] p=0.886, Ue=0.0000, Ua=0.4040, C=0.528 → escalate ✓
```

**Missad (1/6):**
- p=0.977 (extremt hög confidence) → C=0.889 passerar threshold
- Detta är acceptabelt: vid 97.7% säkerhet kanske vi ska lita på modellen

## GO/NO-GO Decision

**Target:** ≥30% Safety Gain för GO
**Result:** 83.3% Safety Gain

✅ **GO: CognOS v1.5 ger mätbar riskreduktion**

## Nästa Steg (Cykel 1, vecka 2)

1. **Test på riktigt data**
   - Syntetisk data är konstruerad — testa på verklig eller mer realistisk data
   - Sjukvårdstriage pilot (om data finns)
   - Alternativt: UCI ML datasets med ground truth

2. **Kalibrera Ua-heuristiken**
   - Nuvarande: Ua = 4 × p × (1-p)
   - Kanske för aggressiv? Testa Ua = 2 × p × (1-p)
   - Eller estimera Ua från modellen direkt (t.ex. från softmax entropy)

3. **Threshold-tuning**
   - τ=0.8 är arbiträrt valt
   - Plotta precision/recall-kurva för olika τ
   - Balansera auto % vs safety gain

4. **Dokumentera Jasper-integration**
   - Hur Jasper's brain.py kallar confidence.py
   - När ska confidence beräknas? (varje LLM-svar? specifika contexts?)
   - Hur eskalera? (byt modell? be användaren granska?)

## Filer Uppdaterade

- `/media/bjorn/iic/cognos/confidence.py` — nu v1.5 (C = p × (1-Ue-Ua))
- `/media/bjorn/iic/cognos/confidence_v1_deprecated.py` — sparad v1 för referens
- `/media/bjorn/iic/cognos/test_confidence.py` — komplett testsuite
- `/media/bjorn/iic/cognos/INTEGRATIONSARKITEKTUR.md` — uppdaterad med v1.5 rationale
- `/media/bjorn/iic/cognos/V1_V15_TRANSITION.md` — denna fil

## Lärdomar

1. **"Enklaste möjliga" är inte alltid tillräckligt**
   - V1 var för enkel, missade fundamentalt problem
   - Ibland behöver man 2 variabler, inte 1

2. **Syntetisk data avslöjar problem snabbt**
   - 100 datapunkter räckte för att se att v1 inte fungerar
   - Riktigt data kommer vara mer nyanserat

3. **Safety Gain är rätt metrisk**
   - Tack vare användaren för att klargöra: BOE/OE, inte bara accuracy
   - Fokus på risk, inte bara korrekthet

4. **Aleatorisk vs epistemisk osäkerhet är kritiskt**
   - Modeller som är fel pga dålig träning → hög Ue
   - Modeller som är fel pga inneboende data noise → hög Ua
   - Behöver båda för att fånga alla typer av fel

## Tid Spenderat

- V1 implementation: 1h
- V1 test + debug: 2h
- V1.5 pivot + implementation: 1h
- V1.5 test + validation: 0.5h
- **Total: 4.5h** (inom Cykel 1, dag 3)

---

**Status:** CognOS v1.5 är redo för nästa steg. Formel bevisad på syntetisk data.
**Risk:** Ua-heuristiken kanske inte håller på riktiga modeller — behöver valideras.
**Confidence:** Hög att vi är på rätt spår. Safety Gain 83% vs target 30%.
