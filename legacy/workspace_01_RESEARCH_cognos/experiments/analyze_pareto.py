#!/usr/bin/env python3
"""
DEEP ANALYSIS: Find operating points where v1.5 beats baseline
"""

import numpy as np
from test_matched_escalation import generate_synthetic_data, method_baseline, method_v1, method_v15

np.random.seed(42)
data = generate_synthetic_data(n=100)

print("=" * 80)
print("DEEP ANALYSIS: Pareto Curve Key Points")
print("=" * 80)

# Test thresholds from 0.5 to 0.95
thresholds = np.linspace(0.5, 0.95, 10)

print("\n" + "=" * 80)
print("FULL THRESHOLD SWEEP")
print("=" * 80)

print(f"\n{'τ':<8} {'Method':<12} {'Esc%':<10} {'SafetyGain%':<15} {'BOE':<8} {'MOE':<8} {'AutoAcc%':<10}")
print("-" * 90)

baseline_results = []
v1_results = []
v15_results = []

for tau in thresholds:
    b = method_baseline(data, threshold=tau)
    v1 = method_v1(data, threshold=tau)
    v15 = method_v15(data, threshold=tau)
    
    baseline_results.append(b)
    v1_results.append(v1)
    v15_results.append(v15)
    
    print(f"{tau:<8.2f} {'Baseline':<12} {b['escalation_rate']:<10.1f} {b['safety_gain']:<15.1f} {b['BOE']:<8} {b['MOE']:<8} {b['auto_accuracy']:<10.1f}")
    print(f"{tau:<8.2f} {'v1':<12} {v1['escalation_rate']:<10.1f} {v1['safety_gain']:<15.1f} {v1['BOE']:<8} {v1['MOE']:<8} {v1['auto_accuracy']:<10.1f}")
    print(f"{tau:<8.2f} {'v1.5':<12} {v15['escalation_rate']:<10.1f} {v15['safety_gain']:<15.1f} {v15['BOE']:<8} {v15['MOE']:<8} {v15['auto_accuracy']:<10.1f}")
    print()

# Find interesting operating points
print("\n" + "=" * 80)
print("CRITICAL OPERATING POINTS")
print("=" * 80)

# Find escalation rates where we have variation in safety gain
target_escalations = [60, 70, 80, 90]

print("\n📊 COMPARISON AT DIFFERENT ESCALATION RATES:\n")

for target_esc in target_escalations:
    print(f"\nTarget Escalation: {target_esc}%")
    print("-" * 70)
    
    # Find closest results for each method
    b_closest = min(baseline_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    v1_closest = min(v1_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    v15_closest = min(v15_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    
    print(f"{'Method':<12} {'ActualEsc%':<12} {'SafetyGain%':<15} {'BOE/MOE':<12} {'AutoAcc%':<10} {'τ':<8}")
    print(f"{'Baseline':<12} {b_closest['escalation_rate']:<12.1f} {b_closest['safety_gain']:<15.1f} {b_closest['BOE']}/{b_closest['MOE']:<10} {b_closest['auto_accuracy']:<10.1f} {b_closest['threshold']:<8.2f}")
    print(f"{'v1':<12} {v1_closest['escalation_rate']:<12.1f} {v1_closest['safety_gain']:<15.1f} {v1_closest['BOE']}/{v1_closest['MOE']:<10} {v1_closest['auto_accuracy']:<10.1f} {v1_closest['threshold']:<8.2f}")
    print(f"{'v1.5':<12} {v15_closest['escalation_rate']:<12.1f} {v15_closest['safety_gain']:<15.1f} {v15_closest['BOE']}/{v15_closest['MOE']:<10} {v15_closest['auto_accuracy']:<10.1f} {v15_closest['threshold']:<8.2f}")
    
    # Analyze winner
    if v15_closest['safety_gain'] > max(b_closest['safety_gain'], v1_closest['safety_gain']):
        print(f"  ✅ WINNER: v1.5 (ΔSafety: +{v15_closest['safety_gain'] - max(b_closest['safety_gain'], v1_closest['safety_gain']):.1f}%)")
    elif b_closest['safety_gain'] > v15_closest['safety_gain']:
        print(f"  ⚠️  WARNING: Baseline beats v1.5 (ΔSafety: +{b_closest['safety_gain'] - v15_closest['safety_gain']:.1f}%)")
    else:
        print(f"  ⚖️  TIE: All methods equal")

# Final verdict based on Pareto dominance
print("\n" + "=" * 80)
print("PARETO DOMINANCE ANALYSIS")
print("=" * 80)

print("\n🔍 Looking for Pareto-optimal points where v1.5 dominates...")

dominated_points = 0
for i, (b, v15) in enumerate(zip(baseline_results, v15_results)):
    # v1.5 dominates if: better SafetyGain at equal/lower Escalation, or equal SafetyGain at lower Escalation
    if v15['safety_gain'] > b['safety_gain'] and v15['escalation_rate'] <= b['escalation_rate']:
        print(f"\n✅ v1.5 Dominates at τ={v15['threshold']:.2f}:")
        print(f"   v1.5:     {v15['escalation_rate']:.1f}% esc, {v15['safety_gain']:.1f}% safety")
        print(f"   Baseline: {b['escalation_rate']:.1f}% esc, {b['safety_gain']:.1f}% safety")
        dominated_points += 1
    elif v15['safety_gain'] == b['safety_gain'] and v15['escalation_rate'] < b['escalation_rate'] - 2:
        print(f"\n➕ v1.5 More Efficient at τ={v15['threshold']:.2f}:")
        print(f"   v1.5:     {v15['escalation_rate']:.1f}% esc → {v15['safety_gain']:.1f}% safety")
        print(f"   Baseline: {b['escalation_rate']:.1f}% esc → {b['safety_gain']:.1f}% safety")
        print(f"   (Same safety, {b['escalation_rate'] - v15['escalation_rate']:.1f}% lower escalation)")
        dominated_points += 1

if dominated_points == 0:
    print("\n❌ NO PARETO DOMINANCE FOUND")
    print("   v1.5 does not beat baseline at any operating point.")
    print("   The formula provides no additional value beyond p-threshold.")

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Count how many thresholds each method wins at
baseline_wins = sum(1 for b, v15 in zip(baseline_results, v15_results) if b['safety_gain'] > v15['safety_gain'])
v15_wins = sum(1 for b, v15 in zip(baseline_results, v15_results) if v15['safety_gain'] > b['safety_gain'])
ties = len(baseline_results) - baseline_wins - v15_wins

print(f"\nAcross {len(thresholds)} tested thresholds:")
print(f"  Baseline wins: {baseline_wins}")
print(f"  v1.5 wins: {v15_wins}")
print(f"  Ties: {ties}")

if v15_wins > baseline_wins:
    print("\n✅ GO: v1.5 provides value at some operating points")
elif v15_wins == 0 and baseline_wins == 0:
    print("\n⚖️  NEUTRAL: All methods identical (dataset too simple?)")
else:
    print("\n❌ NO-GO: Baseline dominates v1.5")

# Recommendation
print("\n📝 RECOMMENDATION:")
if v15_wins > 0 or dominated_points > 0:
    print("   Continue to real data testing — there may be value in more complex scenarios.")
    print("   Current synthetic data might be too simple (only 7 OE, binary scenarios).")
else:
    print("   Current formula does not provide value over p-threshold.")
    print("   Consider: (1) test on real data, (2) redesign Ua heuristic, (3) accept as failed experiment.")
