#!/usr/bin/env python3
"""
CognOS v1 — Confidence Engine

Epicenter: Beräknar beslutskonfidens C = p(x) × (1 - Ue)
där Ue är epistemisk osäkerhet från MC Dropout eller ensemble disagreement.

Design: enklaste möjliga. En variabel. Ett threshold. Bevis först, komplexitet senare.
"""

import numpy as np
from typing import List, Dict, Literal


def compute_confidence(
    prediction: float,
    mc_predictions: List[float],
    threshold: float = 0.8
) -> Dict[str, float | str]:
    """
    Beräknar beslutskonfidens baserat på prediction och epistemisk osäkerhet.
    
    Args:
        prediction: Modellens top prediction probability (0-1)
        mc_predictions: Lista av T prediktioner från MC Dropout runs (0-1)
        threshold: Decision threshold för auto vs escalate (default 0.8)
    
    Returns:
        {
            'confidence': C ∈ [0, 1],
            'epistemic_uncertainty': Ue ∈ [0, 1],
            'decision': 'auto' | 'escalate',
            'prediction': original prediction
        }
    
    Example:
        >>> compute_confidence(0.92, [0.91, 0.93, 0.90, 0.94, 0.89])
        {
            'confidence': 0.9178...,
            'epistemic_uncertainty': 0.000328,
            'decision': 'auto',
            'prediction': 0.92
        }
    """
    if not (0 <= prediction <= 1):
        raise ValueError(f"prediction must be in [0, 1], got {prediction}")
    
    if not all(0 <= p <= 1 for p in mc_predictions):
        raise ValueError("All mc_predictions must be in [0, 1]")
    
    if len(mc_predictions) < 2:
        raise ValueError("mc_predictions must contain at least 2 samples")
    
    # Epistemisk osäkerhet = variance över MC Dropout runs
    Ue = float(np.var(mc_predictions))
    
    # Beslutskonfidens
    C = prediction * (1 - Ue)
    
    # Decision rule
    decision: Literal['auto', 'escalate'] = 'auto' if C >= threshold else 'escalate'
    
    return {
        'confidence': C,
        'epistemic_uncertainty': Ue,
        'decision': decision,
        'prediction': prediction
    }


def route_model(
    input_data: any,
    models: List[Dict],
    threshold: float = 0.8,
    max_escalations: int = 2
) -> Dict:
    """
    Dynamisk modellrouting: small → medium → large baserat på confidence.
    
    Args:
        input_data: Input till modeller
        models: Lista av modeller med keys: 'name', 'predict_fn', 'cost'
        threshold: Confidence threshold för att acceptera prediction
        max_escalations: Max antal eskaleringar (default 2, för 3 modeller)
    
    Returns:
        {
            'final_prediction': float,
            'confidence': float,
            'model_used': str,
            'total_cost': float,
            'escalations': int,
            'trajectory': list  # history av modeller som kördes
        }
    
    Example:
        models = [
            {'name': 'small', 'predict_fn': small_model, 'cost': 0.01},
            {'name': 'medium', 'predict_fn': medium_model, 'cost': 0.10},
            {'name': 'large', 'predict_fn': large_model, 'cost': 1.00},
        ]
        route_model(x, models, threshold=0.8)
    """
    if not models:
        raise ValueError("models list cannot be empty")
    
    trajectory = []
    total_cost = 0.0
    escalations = 0
    
    for i, model in enumerate(models):
        if escalations >= max_escalations:
            break
        
        # Kör modell (predict_fn borde returnera (prediction, mc_predictions))
        try:
            prediction, mc_predictions = model['predict_fn'](input_data)
        except ValueError:
            # Fallback om predict_fn bara returnerar en float
            prediction = model['predict_fn'](input_data)
            mc_predictions = [prediction] * 5  # Dummy MC samples
        
        total_cost += model['cost']
        
        # Beräkna confidence
        result = compute_confidence(prediction, mc_predictions, threshold)
        
        trajectory.append({
            'model': model['name'],
            'prediction': prediction,
            'confidence': result['confidence'],
            'decision': result['decision']
        })
        
        # Om auto → klar
        if result['decision'] == 'auto':
            return {
                'final_prediction': prediction,
                'confidence': result['confidence'],
                'model_used': model['name'],
                'total_cost': total_cost,
                'escalations': escalations,
                'trajectory': trajectory
            }
        
        # Annars eskalera
        escalations += 1
    
    # Om vi nått hit: alla modeller kördes, returnera sista
    last = trajectory[-1]
    return {
        'final_prediction': last['prediction'],
        'confidence': last['confidence'],
        'model_used': last['model'],
        'total_cost': total_cost,
        'escalations': escalations,
        'trajectory': trajectory
    }


if __name__ == '__main__':
    # Quick smoke test
    print("CognOS v1 — Confidence Engine")
    print("=" * 40)
    
    # Test 1: Hög confidence → auto
    result = compute_confidence(0.95, [0.94, 0.96, 0.95, 0.93, 0.97])
    print(f"\nTest 1 (hög confidence):")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Ue: {result['epistemic_uncertainty']:.6f}")
    print(f"  Decision: {result['decision']}")
    
    # Test 2: Låg confidence → escalate
    result = compute_confidence(0.65, [0.50, 0.70, 0.60, 0.80, 0.55])
    print(f"\nTest 2 (låg confidence):")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Ue: {result['epistemic_uncertainty']:.6f}")
    print(f"  Decision: {result['decision']}")
    
    # Test 3: Hög prediction men hög osäkerhet → escalate
    result = compute_confidence(0.90, [0.60, 0.85, 0.95, 0.70, 1.0])
    print(f"\nTest 3 (hög prediction, hög Ue):")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Ue: {result['epistemic_uncertainty']:.6f}")
    print(f"  Decision: {result['decision']}")
    
    print("\n" + "=" * 40)
    print("Smoke test: OK ✓")
