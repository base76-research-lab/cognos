#!/usr/bin/env python3
"""
TEST v1 vs Baseline at matched escalation rates
"""

import numpy as np
from test_matched_escalation import generate_synthetic_data, method_baseline, method_v1, method_v15

np.random.seed(42)
data = generate_synthetic_data(n=100)

print("=" * 80)
print("v1 vs BASELINE: Matched Escalation Test")
print("=" * 80)

# Test key escalation rates
test_escalations = [80, 85, 90, 92, 95]

for target_esc in test_escalations:
    print(f"\n{'=' * 80}")
    print(f"Target Escalation: {target_esc}%")
    print(f"{'=' * 80}")
    
    # Find thresholds for baseline and v1 to match escalation
    baseline_thresholds = np.linspace(0.5, 0.99, 100)
    v1_thresholds = np.linspace(0.5, 0.99, 100)
    
    # Find baseline threshold
    best_baseline = None
    best_baseline_diff = float('inf')
    for tau in baseline_thresholds:
        result = method_baseline(data, threshold=tau)
        diff = abs(result['escalation_rate'] - target_esc)
        if diff < best_baseline_diff:
            best_baseline_diff = diff
            best_baseline = result
    
    # Find v1 threshold
    best_v1 = None
    best_v1_diff = float('inf')
    for tau in v1_thresholds:
        result = method_v1(data, threshold=tau)
        diff = abs(result['escalation_rate'] - target_esc)
        if diff < best_v1_diff:
            best_v1_diff = diff
            best_v1 = result
    
    print(f"\n{'Method':<15} {'τ':<8} {'ActualEsc%':<12} {'SafetyGain%':<15} {'BOE/MOE':<10} {'AutoAcc%':<10}")
    print("-" * 80)
    print(f"{'Baseline':<15} {best_baseline['threshold']:<8.4f} {best_baseline['escalation_rate']:<12.1f} {best_baseline['safety_gain']:<15.1f} {best_baseline['BOE']}/{best_baseline['MOE']:<8} {best_baseline['auto_accuracy']:<10.1f}")
    print(f"{'v1':<15} {best_v1['threshold']:<8.4f} {best_v1['escalation_rate']:<12.1f} {best_v1['safety_gain']:<15.1f} {best_v1['BOE']}/{best_v1['MOE']:<8} {best_v1['auto_accuracy']:<10.1f}")
    
    # Verdict
    if best_v1['safety_gain'] > best_baseline['safety_gain']:
        diff = best_v1['safety_gain'] - best_baseline['safety_gain']
        print(f"\n✅ v1 WINS: +{diff:.1f}% safety gain at matched escalation!")
    elif best_v1['safety_gain'] < best_baseline['safety_gain']:
        diff = best_baseline['safety_gain'] - best_v1['safety_gain']
        print(f"\n❌ Baseline wins: +{diff:.1f}% safety gain")
    else:
        print(f"\n⚖️  TIE")
        if best_v1['auto_accuracy'] > best_baseline['auto_accuracy']:
            print(f"   But v1 has better auto accuracy: {best_v1['auto_accuracy']:.1f}% vs {best_baseline['auto_accuracy']:.1f}%")

print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

# Count wins
v1_wins = 0
baseline_wins = 0
ties = 0

for target_esc in test_escalations:
    baseline_thresholds = np.linspace(0.5, 0.99, 100)
    v1_thresholds = np.linspace(0.5, 0.99, 100)
    
    best_baseline = None
    best_baseline_diff = float('inf')
    for tau in baseline_thresholds:
        result = method_baseline(data, threshold=tau)
        diff = abs(result['escalation_rate'] - target_esc)
        if diff < best_baseline_diff:
            best_baseline_diff = diff
            best_baseline = result
    
    best_v1 = None
    best_v1_diff = float('inf')
    for tau in v1_thresholds:
        result = method_v1(data, threshold=tau)
        diff = abs(result['escalation_rate'] - target_esc)
        if diff < best_v1_diff:
            best_v1_diff = diff
            best_v1 = result
    
    if best_v1['safety_gain'] > best_baseline['safety_gain']:
        v1_wins += 1
    elif best_v1['safety_gain'] < best_baseline['safety_gain']:
        baseline_wins += 1
    else:
        ties += 1

print(f"\nv1 wins: {v1_wins}/{len(test_escalations)}")
print(f"Baseline wins: {baseline_wins}/{len(test_escalations)}")
print(f"Ties: {ties}/{len(test_escalations)}")

if v1_wins > 0:
    print("\n✅ GO: v1 (epistemic only) provides REAL value over baseline!")
    print("   This suggests epistemic uncertainty matters, but Ua (aleatoric heuristic) is too aggressive.")
    print("\n📝 RECOMMENDATION:")
    print("   - v1 is the winner (not v1.5)")
    print("   - Or: redesign Ua in v1.5 to be less aggressive")
    print("   - Test on real data with v1 as main candidate")
else:
    print("\n❌ NO-GO: Even v1 doesn't beat baseline at matched escalation")
