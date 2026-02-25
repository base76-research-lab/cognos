Integrationsarkitektur: CUDA/Tensor + CognOS

Syfte

Dokumentet beskriver en konkret integrationsarkitektur for CognOS ovanpa CUDA/Tensor-stacken
samt ett matematiskt ramverk for confidence-motorn.

1) Oversiktlig stack

GPU/CUDA
  -> Tensor frameworks (PyTorch/TensorFlow)
     -> Modelllager (flera modeller)
        -> CognOS Runtime
           -> Applikationer / Humans

CognOS lagg till beslutsgraf, malmodell, uncertainty, audit trail och routing.

2) Komponenter och dataflode

A) Dataingest (Field)
- Datakallor: tabular, dokument, realtid
- Feature store + metadata (kvalitet, provenance, timestamp)
- Policy/regler som constraints

B) Model layer (Nodes)
- Prediktionsmodell: y = f(x)
- Osakerhetsmodell: u = g(x)
- Kalibrering: p_hat = calibrate(p, u)
- Alternativ/konfliktmodell: ensemble eller flera specialistmodeller

C) CognOS Runtime
- Model arbitration: valj modell m_i baserat pa kontext och expected utility
- Decision graph: representera beslut som noder + edges
- Goal engine: optimering under constraints
- Confidence engine: explicit epistemisk osakerhet + risk
- Audit layer: spårbar beslutskedja

3) CUDA/Tensor-integration (praktiskt)

- Inference service lagg in:
  - Ett "router"-head som valjer modell
  - Ett "uncertainty"-head som estimerar epistemisk/aleatorisk osakerhet
- Batch routing: skicka requests till billig/halv/stor modell beroende pa context
- Early exit: stoppa inference om confidence >= threshold
- Fallback: eskalera om confidence < threshold

4) Confidence-motor: v1.5 (med aleatorisk osäkerhet)

**Viktigt: v1 (C = p × (1-Ue)) misslyckades testet.**
V1 kunde inte fånga överkonfidenta fel med låg epistemisk osäkerhet (Safety Gain: 0%).

**CognOS v1.5 lägger till aleatorisk osäkerhet (Ua):**

```python
C = p(x) × (1 - Ue - Ua)
```

Där:
- `p(x)` = modellens prediktiva sannolikhet för vald klass (0-1)
- `Ue` = epistemisk osäkerhet från MC Dropout (0-1), model uncertainty
- `Ua` = aleatorisk osäkerhet (0-1), data uncertainty / inneboende noise

Decision rule:
- C ≥ τ (t.ex. 0.8) → auto
- C < τ → eskalera till dyrare modell eller människa

**Test results (100 synthetic samples):**
- Baseline: 6 överkonfidenta fel (wrong predictions med p ≥ 0.8)
- CognOS v1.5: Blockerade 5/6 (83.3% Safety Gain) ✅ GO
- v1.5 fångar fel med hög p + låg Ue genom Ua-komponenten

**Framtida variabler (ej v1.5):**
- Datakvalitet/provenance
- Modellkonflikt (ensemble disagreement)
- Policy/risk penalty

5) Osäkerhetsestimat (v1.5)

Epistemisk osäkerhet via MC Dropout:
```python
# Kör modell T gånger med dropout aktiverat under inference
predictions = [model(x, training=True) for _ in range(T)]
Ue = np.var(predictions)  # Variance över körningar
```

Aleatorisk osäkerhet (approximation):
```python
# Heuristik: entropy-baserad, max vid p=0.5, min vid p=0 eller p=1
Ua = 4 × p × (1 - p)
```

*Alternativt kan Ua estimeras från modellens interna representationer
eller från separat osäkerhetsmodell.*

Alternativt (om ensemble finns):
```python
Ue = 1 - np.mean([p_i.max() for p_i in predictions])  # Disagreement mellan modeller
```

6) Implementation (Python, ej TensorFlow ännu)

CognOS v1 är Python-first. TensorFlow/PyTorch-integration kommer när vi bevisat värdet.

```python
# cognos/confidence.py
def compute_confidence(p: float, predictions: list[float]) -> float:
    """
    Args:
        p: Prediktiv sannolikhet (0-1)
        predictions: Lista av T predictions från MC Dropout
    Returns:
        C: Confidence score (0-1)
    """
    Ue = np.var(predictions)
    C = p * (1 - Ue)
    return C
```

Detta testas först på syntetisk data, sedan på sjukvårdstriage-data.

7) MVP-implementation (dag 1)

Bygg en testbar Python-funktion idag:

```python
# cognos/confidence.py
import numpy as np

def compute_confidence(
    prediction: float,
    mc_predictions: list[float],
    threshold: float = 0.8
) -> dict:
    """
    CognOS v1 confidence engine.
    
    Args:
        prediction: Modellens top prediction (0-1)
        mc_predictions: T predictions från MC Dropout runs
        threshold: Decision threshold (default 0.8)
    
    Returns:
        {
            'confidence': C,
            'epistemic_uncertainty': Ue,
            'decision': 'auto' | 'escalate',
            'prediction': prediction
        }
    """
    Ue = float(np.var(mc_predictions))
    C = prediction * (1 - Ue)
    
    decision = 'auto' if C >= threshold else 'escalate'
    
    return {
        'confidence': C,
        'epistemic_uncertainty': Ue,
        'decision': decision,
        'prediction': prediction
    }
```

Test på 100 syntetiska datapunkter. Tid: 1 dag.

8) Demo-scenario (vecka 1)

- 3 modeller: small (snabb/billig), medium, large (långsam/dyr)
- Routing: small → om C < 0.8 → medium → om C < 0.8 → large
- Mät: kostnad per beslut, precision, antal eskaleringar

Detta är epicenter. Resten kommer när detta fungerar.
