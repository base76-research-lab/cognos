#!/usr/bin/env python3
"""
test_confidence.py — CognOS v1 test suite

Testar confidence engine på 100 syntetiska datapunkter.
Mål: bevisa att C = p × (1 - Ue) ger mätbar nytta.
"""

import numpy as np
from confidence import compute_confidence, route_model


def generate_synthetic_data(n=100, seed=42):
    """
    Genererar syntetisk data med varierad confidence.
    Inkluderar medvetet överkonfidenta felaktiga prediktioner.
    
    CognOS hypotes: Modeller som är fel är ofta osäkra (hög Ue).
    Men ibland är de fel OCH säkra (överkonfidenta) → farligt.
    
    Returns:
        List of (prediction, mc_predictions, ground_truth)
    """
    np.random.seed(seed)
    data = []
    
    for i in range(n):
        # Ground truth (0 eller 1)
        ground_truth = np.random.choice([0, 1])
        
        # 30% av tiden: generera osäker eller felaktig prediction
        scenario = np.random.choice(['correct_confident', 'correct_uncertain', 
                                     'wrong_confident', 'wrong_uncertain'], 
                                    p=[0.5, 0.2, 0.15, 0.15])
        
        if scenario == 'correct_confident':
            # Rätt prediction, låg osäkerhet (bästa fallet)
            if ground_truth == 1:
                prediction = np.random.beta(8, 2)
            else:
                prediction = np.random.beta(2, 8)
            mc_noise = np.random.normal(0, 0.01, 5)
            uncertainty_level = 'low'
            
        elif scenario == 'correct_uncertain':
            # Rätt prediction, hög osäkerhet (bra att eskalera)
            if ground_truth == 1:
                prediction = np.random.beta(6, 4)  # Lite lägre confidence
            else:
                prediction = np.random.beta(4, 6)
            mc_noise = np.random.normal(0, 0.12, 5)
            uncertainty_level = 'high'
            
        elif scenario == 'wrong_confident':
            # FEL prediction, men inte helt övertygad (lite osäkerhet)
            # Realistiskt: modeller som är fel tenderar att ha lite mer variance
            if ground_truth == 1:
                prediction = np.random.beta(2, 8)  # Säger nej när det är ja
            else:
                prediction = np.random.beta(8, 2)  # Säger ja när det är nej
            # Högre osäkerhet än correct_confident (realistiskt beteende)
            mc_noise = np.random.normal(0, 0.05, 5)  # Lite högre än 0.01
            uncertainty_level = 'medium'
            
        else:  # wrong_uncertain
            # FEL prediction, hög osäkerhet (modellen vet att den inte vet)
            if ground_truth == 1:
                prediction = np.random.beta(3, 7)
            else:
                prediction = np.random.beta(7, 3)
            mc_noise = np.random.normal(0, 0.15, 5)
            uncertainty_level = 'high'
        
        mc_base = prediction
        mc_predictions = np.clip(mc_base + mc_noise, 0, 1).tolist()
        
        data.append({
            'prediction': prediction,
            'mc_predictions': mc_predictions,
            'ground_truth': ground_truth,
            'uncertainty': uncertainty_level,
            'scenario': scenario
        })
    
    return data


def evaluate_baseline(data, threshold=0.5):
    """
    Baseline: använd bara prediction p(x), inget confidence.
    Räknar ALLA överkonfidenta fel (ground truth risk).
    """
    correct = 0
    overconfident_errors = []  # Lista med alla farliga fall
    
    for i, d in enumerate(data):
        predicted_class = 1 if d['prediction'] >= threshold else 0
        is_correct = predicted_class == d['ground_truth']
        
        if is_correct:
            correct += 1
        elif d['prediction'] >= 0.8:  # Fel med hög confidence = FARLIGT
            overconfident_errors.append(i)
    
    return {
        'accuracy': correct / len(data),
        'overconfident_errors': overconfident_errors,  # Index av farliga fall
        'total': len(data)
    }


def evaluate_cognos(data, overconfident_error_indices, threshold=0.8):
    """
    CognOS: använd C = p × (1 - Ue) för att filtrera.
    Räknar hur många av de FARLIGA fallen (OE) som blockeras.
    """
    auto_decisions = 0
    escalated = 0
    correct_auto = 0
    
    # För varje överkonfident fel: blockerat eller missat?
    blocked_overconfident_errors = []  # BOE
    missed_overconfident_errors = []   # MOE
    
    for i, d in enumerate(data):
        result = compute_confidence(
            d['prediction'],
            d['mc_predictions'],
            threshold
        )
        
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        is_overconfident_error = i in overconfident_error_indices
        
        if result['decision'] == 'auto':
            auto_decisions += 1
            if is_correct:
                correct_auto += 1
            if is_overconfident_error:
                missed_overconfident_errors.append(i)  # FARLIGT: slapp igenom
        else:
            escalated += 1
            if is_overconfident_error:
                blocked_overconfident_errors.append(i)  # BRA: blockerat
    
    return {
        'auto_decisions': auto_decisions,
        'escalated': escalated,
        'accuracy_on_auto': correct_auto / auto_decisions if auto_decisions > 0 else 0,
        'blocked_overconfident_errors': blocked_overconfident_errors,  # BOE
        'missed_overconfident_errors': missed_overconfident_errors,    # MOE
        'total': len(data)
    }


def test_model_routing():
    """Test model routing med dummy-modeller."""
    
    # Dummy models
    def small_model(x):
        pred = 0.75
        mc = [0.73, 0.76, 0.74, 0.77, 0.75]
        return pred, mc
    
    def medium_model(x):
        pred = 0.85
        mc = [0.84, 0.86, 0.85, 0.85, 0.86]
        return pred, mc
    
    def large_model(x):
        pred = 0.95
        mc = [0.94, 0.95, 0.96, 0.95, 0.94]
        return pred, mc
    
    models = [
        {'name': 'small', 'predict_fn': small_model, 'cost': 0.01},
        {'name': 'medium', 'predict_fn': medium_model, 'cost': 0.10},
        {'name': 'large', 'predict_fn': large_model, 'cost': 1.00},
    ]
    
    # Test case 1: Small model räcker
    result = route_model(None, models, threshold=0.7)
    assert result['model_used'] == 'small'
    assert result['escalations'] == 0
    assert result['total_cost'] == 0.01
    
    # Test case 2: Behöver eskalera till medium
    result = route_model(None, models, threshold=0.8)
    assert result['model_used'] == 'medium'
    assert result['escalations'] == 1
    assert result['total_cost'] == 0.11  # small + medium
    
    # Test case 3: Behöver eskalera till large
    result = route_model(None, models, threshold=0.9)
    assert result['model_used'] == 'large'
    assert result['escalations'] == 2
    assert result['total_cost'] == 1.11  # small + medium + large
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("CognOS v1 — Test Suite")
    print("=" * 60)
    
    # Generera data
    print("\n📊 Genererar 100 syntetiska datapunkter...")
    data = generate_synthetic_data(100)
    
    # Analysera distributionen
    scenarios = {}
    for d in data:
        s = d['scenario']
        scenarios[s] = scenarios.get(s, 0) + 1
    
    print("\n📋 Scenario-distribution:")
    for s, count in scenarios.items():
        print(f"  {s}: {count}")
    
    # Baseline
    print("\n🎯 Baseline (bara prediction):")
    baseline = evaluate_baseline(data)
    OE = len(baseline['overconfident_errors'])
    print(f"  Accuracy: {baseline['accuracy']:.2%}")
    print(f"  Overconfident Errors (OE): {OE}")
    
    # Debug: visa de överkonfidenta felen
    print("\n🔍 Debug: Överkonfidenta fel i detalj:")
    for idx in baseline['overconfident_errors']:
        d = data[idx]
        result = compute_confidence(d['prediction'], d['mc_predictions'], threshold=0.8)
        print(f"  [{idx}] p={d['prediction']:.3f}, Ue={result['epistemic_uncertainty']:.4f}, "
              f"C={result['confidence']:.3f}, decision={result['decision']}, scenario={d['scenario']}")
    
    # CognOS
    print("\n🧠 CognOS (confidence filtering, threshold=0.8):")
    cognos = evaluate_cognos(data, baseline['overconfident_errors'], threshold=0.8)
    BOE = len(cognos['blocked_overconfident_errors'])
    MOE = len(cognos['missed_overconfident_errors'])
    
    print(f"  Auto decisions: {cognos['auto_decisions']}")
    print(f"  Escalated: {cognos['escalated']}")
    print(f"  Accuracy on auto: {cognos['accuracy_on_auto']:.2%}")
    print(f"  Blocked Overconfident Errors (BOE): {BOE}")
    print(f"  Missed Overconfident Errors (MOE): {MOE}")
    
    # Testa med lägre threshold
    print("\n🧠 CognOS (confidence filtering, threshold=0.7):")
    cognos_07 = evaluate_cognos(data, baseline['overconfident_errors'], threshold=0.7)
    BOE_07 = len(cognos_07['blocked_overconfident_errors'])
    MOE_07 = len(cognos_07['missed_overconfident_errors'])
    
    print(f"  Auto decisions: {cognos_07['auto_decisions']}")
    print(f"  Escalated: {cognos_07['escalated']}")
    print(f"  Accuracy on auto: {cognos_07['accuracy_on_auto']:.2%}")
    print(f"  Blocked Overconfident Errors (BOE): {BOE_07}")
    print(f"  Missed Overconfident Errors (MOE): {MOE_07}")
    
    # Comparison
    print("\n📈 Safety Gain (threshold=0.8):")
    if OE > 0:
        safety_gain = (BOE / OE) * 100
        print(f"  Total Overconfident Errors (OE): {OE}")
        print(f"  Blocked by CognOS (BOE): {BOE}")
        print(f"  Missed by CognOS (MOE): {MOE}")
        print(f"  Safety Gain: {safety_gain:.1f}% (BOE/OE)")
    else:
        print(f"  N/A: Inga överkonfidenta fel i datasetet")
    print(f"  Eskalerade fall totalt: {cognos['escalated']}")
    
    print("\n📈 Safety Gain (threshold=0.7):")
    if OE > 0:
        safety_gain_07 = (BOE_07 / OE) * 100
        print(f"  Blocked by CognOS (BOE): {BOE_07}")
        print(f"  Missed by CognOS (MOE): {MOE_07}")
        print(f"  Safety Gain: {safety_gain_07:.1f}% (BOE/OE)")
    print(f"  Eskalerade fall totalt: {cognos_07['escalated']}")
    
    # Model routing test
    print("\n🚦 Test: Model Routing...")
    try:
        test_model_routing()
        print("  ✓ Model routing fungerar")
    except AssertionError as e:
        print(f"  ✗ Model routing misslyckades: {e}")
    
    # Go/No-Go decision
    print("\n" + "=" * 60)
    if OE > 0:
        safety_gain = (BOE / OE) * 100
        
        if safety_gain >= 30:
            print(f"✅ GO: CognOS blockerade {BOE}/{OE} överkonfidenta fel (Safety Gain: {safety_gain:.1f}%)")
        else:
            print(f"⚠️  NO-GO: Endast {BOE}/{OE} blockerade (Safety Gain: {safety_gain:.1f}% < 30% threshold)")
    else:
        print("⚠️  N/A: Datasetet hade inga överkonfidenta fel att reducera")
    print("=" * 60)
