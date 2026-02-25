# KRITISK GENOMGÅNG — CognOS Matched Escalation Test

**Datum:** 21 februari 2026  
**Test:** Matched escalation rate comparison (enligt din kritik)  
**Resultat:** NEGATIVT på syntetisk data, MEN VÄNTA MED SLUTSATSEN

---

## Sammanfattning (TL;DR)

v1.5 ger **INTE** högre SafetyGain vid matchad escalation rate. Vinsten var artifact av högre escalation. 

**MEN** — Root cause: Syntetisk data har Mean Ue = 0.015 (1.5%), vilket är **10x lägre än verkliga modeller**. Detta gör testet **orelevant** för riktiga use cases.

**Beslut du måste ta:** A) Abandon, B) Test på real data (rekommenderat), eller C) Fix syntetisk data

---

## Vad du krävde (och vi gjorde)

### 1. ✅ Matched escalation-rate test

**Fråga:** "Är vinsten verklig, eller bara högre escalation = fler stoppade fel?"

**Test:**
- v1.5 vid τ=0.8 → 98% escalation, 100% safety gain (7/7 BOE)
- Baseline vid τ=0.93 → 98% escalation, 100% safety gain (7/7 BOE)

**Resultat:** Identiska. v1.5 ger INGEN ytterligare nytta vid matchad escalation.

### 2. ✅ Threshold sweep (τ=0.5→0.95)

**Test:** Alla tre metoder vid 10 olika thresholds.

**Resultat:** 
- v1.5 "vinner" vid 7/10 thresholds, men genom högre escalation
- v1 "vinner" vid 0/5 matchade escalation rates (tie eller baseline vinner)
- Ingen Pareto dominance hittad

### 3. ✅ Pareto curves

Genererade 3 figurer:
- `pareto_curves.png` — SafetyGain vs Escalation (båda metoder identiska vid hög escalation)
- `confusion_bars.png` — BOE/MOE staplar
- `threshold_sweep.png` — τ vs metrics

---

## Root Cause Analysis

### Problem: Epistemic Uncertainty är för låg

```
Mean Ue:  0.015 (1.5%)
Max Ue:   0.044 (4.4%)
Ue > 0.05: 0/100 datapoints (0%)
```

**Detta betyder:**
- C = p × (1 - 0.015) ≈ p × 0.985 ≈ p
- v1 är nästan identisk med baseline
- Formeln har ingen chans att differentiera

**Varför syntetisk data har låg Ue:**
- Beta distributions har låg inherent varians
- MC predictions dragna från samma beta → låg varians mellan predictions → låg Ue
- Realistiska ML-modeller har Ue ∈ [0.1, 0.3] (10-50x högre)

**Exempel från verkliga modeller:**
- MC Dropout med 10 samples: Ue ≈ 0.10-0.20
- Ensemble (5 models): Ue ≈ 0.15-0.30
- Single model (no uncertainty): Ue ≈ 0

---

## Tre Alternativ (du måste välja)

### A) ACCEPT FAILURE — Abandon CognOS

**Argument:**
- Testet visar ingen nytta vid matchad escalation
- Formeln fungerar inte som intended
- Cut losses, fokusera på annat

**Motargument:**
- Testet är inte representativt (Ue för låg)
- Kastar bort potentiellt fungerande idé p.g.a. dålig testdata

**Kostnad:** 0 timmar  
**Risk:** Låg (men kan kasta bort fungerande idé)

---

### B) TEST ON REAL DATA — Recommended ⭐

**Argument:**
- Mean Ue=0.015 är 10x lägre än verkliga modeller
- Med Ue=0.10-0.20 kan formeln visa värde
- Definitivt svar om CognOS fungerar i praktiken

**Approach:**
1. Ladda UCI dataset (Heart Disease, Breast Cancer, eller MIMIC-III)
2. Träna sklearn RandomForest eller Neural Network
3. Generera MC predictions (dropout eller bagging)
4. Kör samma matched escalation test
5. IF Ue > 0.05 OCH v1 ger värde → GO för paper/Jasper
6. IF fortfarande ingen nytta → definitiv NO-GO

**Kostnad:** 2-3 timmar  
**Risk:** Medium (kan visa att det inte fungerar ändå)  
**Payoff:** Hög (definitivt svar, paper-ready om det fungerar)

**Filer att skapa:**
- `test_real_data_uci.py` — Load dataset, train model, test CognOS
- `REAL_DATA_RESULTS.md` — Dokumentera outcome

---

### C) FIX SYNTHETIC DATA — Increase Ue

**Argument:**
- Snabbaste väg till "working" test
- Kan designa data med Ue ∈ [0.1, 0.3]

**Approach:**
1. Ändra `generate_synthetic_data()` för att öka MC prediction varians
2. Sikta på Mean Ue ≈ 0.15
3. Kör matched escalation test igen

**Kostnad:** 30 minuter  
**Risk:** HÖG — kan bli "overfitting test to formula"  
**Varning:** Paper reviewers kommer fråga "varför just denna Ue-distribution?"

---

## Min Rekommendation (Jasper's take)

**GO för Option B (real data).**

**Rationale:**
1. Syntetisk data är uppenbart orelevant (Ue 10x för låg)
2. 2-3 timmar är rimlig investering för definitiv validering
3. Om real data visar värde → publicerbart, Jasper-ready
4. Om real data visar ingen nytta → definitiv NO-GO utan "vad om"-frågor

**Next Action (om du väljer B):**
```bash
cd /media/bjorn/iic/cognos
python3 test_real_data_uci.py  # (ska skapas)
```

**Expected Outcome:**
- IF Mean Ue > 0.05 on real data → v1 kan visa 10-30% safety gain vid matchad escalation
- IF Mean Ue < 0.05 on real data → samma problem, abandon project

---

## Alternativ beslutspolicy (för låg energi)

**Om energi < 50% och du inte vill investera 2-3h:**

→ **FREEZE CognOS**, dokumentera current state, gå vidare till annat.

**Rationale:**
- Projektet är i "scientific limbo" — ej bevisat fungerande, ej bevisat failed
- Real data-test krävs för definitivt svar
- Om energin inte finns → bättre att freezea än att fortsätta med felaktig testdata

**Om du freezear:**
1. Skapa `COGNOS_FROZEN_STATE.md` med alla resultat + reasoning
2. Tagga git: `git tag cognos_synthetic_test_failed`
3. Återkom när energi finns eller när Jasper behöver confidence engine

---

## Data som stödjer beslut

### Matched Escalation Results (v1 vs Baseline)

| Target Esc% | Baseline SafetyGain | v1 SafetyGain | Winner |
|-------------|-------------------|--------------|--------|
| 80% | 42.9% | 42.9% | TIE |
| 85% | 57.1% | 57.1% | TIE |
| 90% | 57.1% | 57.1% | TIE |
| 92% | 85.7% | 85.7% | TIE |
| 95% | 100.0% | 85.7% | **Baseline** |

**Wins:** Baseline 1, v1 0, Ties 4

### Ue Distribution

| Metric | Value |
|--------|-------|
| Mean Ue | 0.015 (1.5%) |
| Max Ue | 0.044 (4.4%) |
| Datapoints with Ue > 0.05 | 0/100 (0%) |
| Expected in real models | 0.10-0.30 |

**Gap:** Real data skulle ha 10-50x högre Ue.

---

## Figurer (redan genererade)

- `/media/bjorn/iic/cognos/pareto_curves.png` — Safety vs Escalation trade-offs
- `/media/bjorn/iic/cognos/confusion_bars.png` — BOE/MOE comparison
- `/media/bjorn/iic/cognos/threshold_sweep.png` — τ sweep results

---

## Vad är nästa steg?

**Du måste välja:**

A) Abandon (0h, low risk, potential waste)  
B) **Real data test (2-3h, medium risk, high payoff)** ⭐  
C) Fix synthetic (0.5h, high risk of circular validation)

**Om B → jag börjar bygga `test_real_data_uci.py`**  
**Om A eller C → säg till**

**Om osäker → kolla energy_state.md och välj baserat på kapacitet.**

---

## Citat från testresultat

> ❌ PROBLEM IDENTIFIED: Epistemic uncertainty is TOO LOW  
> Mean Ue = 0.015033 (< 0.05)  
>  
> This means:  
> - C = p × (1-Ue) ≈ p × 0.95 ≈ p  
> - v1 is almost identical to baseline p-threshold  
> - Epistemic uncertainty provides no differentiation  

> 🎯 RECOMMENDATION: Option B (real data)  
> Reason: Mean Ue=0.02 is 10x lower than real ML models  
> Cost: 2-3 hours to get real predictions  
> Payoff: Definitive answer whether CognOS works in practice  

---

**Väntar på beslut: A, B, eller C?**
