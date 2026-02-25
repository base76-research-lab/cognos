#!/usr/bin/env python3
"""
CognOS v2 — Confidence Engine med Semantisk Risk och Multimodal Detektion

Versionshistorik:
  v1:   C = p × (1-Ue)             — missade överkonfidenta fel
  v1.5: C = p × (1-Ue-Ua)         — Ua som heuristik (2p(1-p)), domänbegränsad
  v2:   C = p × (1-Ue-Ua)         — Ua som semantisk handlingsrisk
        + multimodal Ue-detektion  — skiljer noise från perspektivkonflikt
        + fyra beslut              — auto / synthesize / explore / escalate

Formler:
  Ue  = var(mc_predictions)                         — epistemisk osäkerhet
  Ua  = (ambiguity + irreversibility + blast_radius) / 3  — semantisk risk
  C   = p × (1 - Ue - Ua)                          — beslutskonfidens

Fyra beslut:
  auto       — C ≥ threshold, agera autonomt
  synthesize — C < threshold, bimodal Ue (perspektivkonflikt → kombinera)
  explore    — C < threshold, unimodal Ue (random noise → sampla mer)
  escalate   — irreversibility hög OCH C låg (för riskfyllt att syntetisera)
"""

import numpy as np
from typing import List, Dict, Optional


def _is_multimodal(mc_predictions: List[float], separation_threshold: float = 0.20) -> bool:
    """
    Avgör om MC-samplingarna representerar perspektivkonflikt (bimodal)
    eller random noise (unimodal).

    Metod: dela vid medelvärdet, mät separation mellan kluster-medel.
    Bimodal = separation > threshold → signal för syntes, inte bara eskalering.

    Args:
        mc_predictions: MC Dropout-samplings
        separation_threshold: Minsta kluster-separation för bimodal (default 0.20)

    Returns:
        True om bimodal (perspektivkonflikt), False om unimodal (noise)
    """
    mc = np.array(mc_predictions)
    if len(mc) < 4:
        return False

    mean = mc.mean()
    cluster_high = mc[mc > mean]
    cluster_low  = mc[mc <= mean]

    if len(cluster_high) == 0 or len(cluster_low) == 0:
        return False

    separation = abs(cluster_high.mean() - cluster_low.mean())
    return float(separation) > separation_threshold


def compute_confidence(
    prediction: float,
    mc_predictions: List[float],
    ambiguity: Optional[float] = None,
    irreversibility: Optional[float] = None,
    blast_radius: Optional[float] = None,
    aleatoric_uncertainty: Optional[float] = None,
    threshold: float = 0.72,
    irreversibility_override: float = 0.85,
    synthesis_separation: float = 0.20,
) -> Dict:
    """
    Beräknar beslutskonfidens och returnerar ett av fyra beslut.

    Ua-prioritering (första som matchar används):
      1. Semantisk: om ambiguity + irreversibility + blast_radius ges
      2. Direkt:    om aleatoric_uncertainty ges
      3. Legacy:    2 × p × (1-p) heuristik (bakåtkompatibilitet)

    Args:
        prediction:             Modellens top prediction probability [0, 1]
        mc_predictions:         T samplings från MC Dropout [0, 1]
        ambiguity:              Instruktionstvetydighet [0, 1]
        irreversibility:        Handlingens reversibilitet [0, 1]
        blast_radius:           Räckvidd av konsekvenser [0, 1]
        aleatoric_uncertainty:  Direkt Ua-override [0, 1]
        threshold:              C-tröskel för auto (default 0.72)
        irreversibility_override: Irr-nivå som alltid eskalerar (default 0.85)
        synthesis_separation:   Bimodal-separation för synthesize (default 0.20)

    Returns:
        {
            'confidence': C ∈ [0, 1],
            'decision': 'auto' | 'synthesize' | 'explore' | 'escalate',
            'is_multimodal': bool,
            'epistemic_uncertainty': Ue,
            'aleatoric_uncertainty': Ua,
            'ua_source': 'semantic' | 'direct' | 'legacy',
            'prediction': float,
            'components': {ambiguity, irreversibility, blast_radius} | None
        }
    """
    if not (0 <= prediction <= 1):
        raise ValueError(f"prediction måste vara i [0, 1], fick {prediction}")
    if not all(0 <= p <= 1 for p in mc_predictions):
        raise ValueError("Alla mc_predictions måste vara i [0, 1]")
    if len(mc_predictions) < 4:
        raise ValueError("mc_predictions kräver minst 4 samplings för multimodal-detektion")

    # --- Epistemisk osäkerhet ---
    Ue = float(np.var(mc_predictions))

    # --- Aleatorisk / semantisk risk ---
    semantic_given = all(x is not None for x in [ambiguity, irreversibility, blast_radius])

    if semantic_given:
        Ua = (ambiguity + irreversibility + blast_radius) / 3.0
        ua_source = 'semantic'
        components = {
            'ambiguity': ambiguity,
            'irreversibility': irreversibility,
            'blast_radius': blast_radius
        }
    elif aleatoric_uncertainty is not None:
        Ua = aleatoric_uncertainty
        ua_source = 'direct'
        components = None
    else:
        Ua = 2 * prediction * (1 - prediction)
        ua_source = 'legacy'
        components = None

    Ua = float(min(max(0.0, Ua), 1.0))

    # --- Beslutskonfidens ---
    C = float(max(0.0, min(1.0, prediction * (1 - Ue - Ua))))

    # --- Multimodal detektion ---
    multimodal = _is_multimodal(mc_predictions, synthesis_separation)

    # --- Beslutslogik (prioritetsordning) ---
    irr_val = irreversibility if irreversibility is not None else 0.0
    if irr_val >= irreversibility_override and C < threshold:
        decision = 'escalate'
    elif C >= threshold:
        decision = 'auto'
    elif multimodal:
        decision = 'synthesize'
    else:
        decision = 'explore'

    return {
        'confidence': C,
        'decision': decision,
        'is_multimodal': multimodal,
        'epistemic_uncertainty': Ue,
        'aleatoric_uncertainty': Ua,
        'ua_source': ua_source,
        'prediction': prediction,
        'components': components,
    }


if __name__ == '__main__':
    print("CognOS v2 — Smoke test")
    print("=" * 56)

    tests = [
        {
            'label': 'Kodläsning (låg risk, konsensus)',
            'p': 0.92, 'mc': [0.91, 0.93, 0.92, 0.90, 0.93, 0.91],
            'kw': dict(ambiguity=0.1, irreversibility=0.05, blast_radius=0.05),
            'expect': 'auto'
        },
        {
            'label': 'Filradering (hög irreversibilitet)',
            'p': 0.80, 'mc': [0.79, 0.81, 0.80, 0.78, 0.82, 0.80],
            'kw': dict(ambiguity=0.5, irreversibility=0.95, blast_radius=0.90),
            'expect': 'escalate'
        },
        {
            'label': 'Perspektivkonflikt (bimodal)',
            'p': 0.72, 'mc': [0.91, 0.89, 0.90, 0.88, 0.38, 0.42, 0.40, 0.41],
            'kw': dict(ambiguity=0.4, irreversibility=0.2, blast_radius=0.2),
            'expect': 'synthesize'
        },
        {
            'label': 'Random noise (unimodal, låg C)',
            'p': 0.60, 'mc': [0.58, 0.62, 0.59, 0.61, 0.60, 0.63],
            'kw': dict(ambiguity=0.5, irreversibility=0.1, blast_radius=0.1),
            'expect': 'explore'
        },
    ]

    for t in tests:
        r = compute_confidence(t['p'], t['mc'], **t['kw'])
        ok = '✅' if r['decision'] == t['expect'] else '❌'
        print(f"\n{ok} {t['label']}")
        print(f"   p={r['prediction']:.2f}  Ue={r['epistemic_uncertainty']:.4f}  "
              f"Ua={r['aleatoric_uncertainty']:.3f}  C={r['confidence']:.3f}")
        print(f"   multimodal={r['is_multimodal']}  "
              f"decision={r['decision']}  (förväntat: {t['expect']})")

    print("\n" + "=" * 56)
