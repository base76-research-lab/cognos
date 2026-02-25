#!/usr/bin/env python3
"""Test suite för CognOS v1.5 (med Ua)"""

import sys
sys.path.insert(0, '.')

# Kopiera evaluation-funktioner från test_confidence.py men använd v1.5
import numpy as np
from confidence_v15 import compute_confidence

# Återanvänd datagen från test_confidence.py
exec(open('test_confidence.py').read().split('if __name__')[0])

if __name__ == '__main__':
    print("=" * 60)
    print("CognOS v1.5 — Test Suite (med aleatorisk osäkerhet)")
    print("=" * 60)
    
    # Generera data
    print("\n📊 Genererar 100 syntetiska datapunkter...")
    data = generate_synthetic_data(100)
    
    # Baseline
    print("\n🎯 Baseline (bara prediction):")
    baseline = evaluate_baseline(data)
    OE = len(baseline['overconfident_errors'])
    print(f"  Accuracy: {baseline['accuracy']:.2%}")
    print(f"  Overconfident Errors (OE): {OE}")
    
    # Debug: visa de överkonfidenta felen
    print("\n🔍 Debug: Överkonfidenta fel")
    for idx in baseline['overconfident_errors'][:3]:  # Visa bara 3 första
        d = data[idx]
        result = compute_confidence(d['prediction'], d['mc_predictions'], threshold=0.8)
        print(f"  [{idx}] p={d['prediction']:.3f}, Ue={result['epistemic_uncertainty']:.4f}, "
              f"Ua={result['aleatoric_uncertainty']:.4f}, C={result['confidence']:.3f}, "
              f"decision={result['decision']}, scenario={d['scenario']}")
    if OE > 3:
        print(f"  ... och {OE-3} till")
    
    # CognOS v1.5
    print("\n🧠 CognOS v1.5 (confidence filtering, threshold=0.8):")
    cognos = evaluate_cognos(data, baseline['overconfident_errors'], threshold=0.8)
    BOE = len(cognos['blocked_overconfident_errors'])
    MOE = len(cognos['missed_overconfident_errors'])
    
    print(f"  Auto decisions: {cognos['auto_decisions']}")
    print(f"  Escalated: {cognos['escalated']}")
    print(f"  Accuracy on auto: {cognos['accuracy_on_auto']:.2%}")
    print(f"  Blocked Overconfident Errors (BOE): {BOE}")
    print(f"  Missed Overconfident Errors (MOE): {MOE}")
    
    # Results
    print("\n📈 Safety Gain:")
    if OE > 0:
        safety_gain = (BOE / OE) * 100
        print(f"  Total Overconfident Errors (OE): {OE}")
        print(f"  Blocked by CognOS (BOE): {BOE}")
        print(f"  Missed by CognOS (MOE): {MOE}")
        print(f"  Safety Gain: {safety_gain:.1f}% (BOE/OE)")
    else:
        print(f"  N/A: Inga överkonfidenta fel i datasetet")
    print(f"  Eskalerade fall totalt: {cognos['escalated']}")
    
    # GO/NO-GO
    print("\n" + "=" * 60)
    if OE > 0:
        safety_gain = (BOE / OE) * 100
        
        if safety_gain >= 30:
            print(f"✅ GO: CognOS v1.5 blockerade {BOE}/{OE} överkonfidenta fel (Safety Gain: {safety_gain:.1f}%)")
            print(f"   Formeln C = p × (1 - Ue - Ua) ger mätbar riskreduktion.")
        else:
            print(f"⚠️  NO-GO: Endast {BOE}/{OE} blockerade (Safety Gain: {safety_gain:.1f}% < 30% threshold)")
    else:
        print("⚠️  N/A: Datasetet hade inga överkonfidenta fel att reducera")
    print("=" * 60)
