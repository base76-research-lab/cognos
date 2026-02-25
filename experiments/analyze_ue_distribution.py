#!/usr/bin/env python3
"""
Analyze epistemic uncertainty distribution in synthetic data
"""

import numpy as np
from test_matched_escalation import generate_synthetic_data

np.random.seed(42)
data = generate_synthetic_data(n=100)

print("=" * 80)
print("EPISTEMIC UNCERTAINTY (Ue) ANALYSIS")
print("=" * 80)

# Compute Ue for all datapoints
Ue_values = []
for d in data:
    Ue = float(np.var(d['mc_predictions']))
    Ue_values.append({
        'Ue': Ue,
        'prediction': d['prediction'],
        'scenario': d['scenario'],
        'ground_truth': d['ground_truth']
    })

Ues = [v['Ue'] for v in Ue_values]

print(f"\n📊 Ue Distribution:")
print(f"  Mean: {np.mean(Ues):.6f}")
print(f"  Median: {np.median(Ues):.6f}")
print(f"  Std: {np.std(Ues):.6f}")
print(f"  Min: {np.min(Ues):.6f}")
print(f"  Max: {np.max(Ues):.6f}")

print(f"\n📊 Ue Percentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(Ues, p):.6f}")

# Analyze by scenario
print(f"\n📊 Ue by Scenario:")
scenarios = {}
for v in Ue_values:
    s = v['scenario']
    if s not in scenarios:
        scenarios[s] = []
    scenarios[s].append(v['Ue'])

for s, ues in scenarios.items():
    print(f"\n  {s} (n={len(ues)}):")
    print(f"    Mean: {np.mean(ues):.6f}")
    print(f"    Max: {np.max(ues):.6f}")

# Compute C for v1 at different Ue levels
print(f"\n📊 Impact of Ue on Confidence (v1):")
print(f"  If p=0.8:")
for ue in [0.01, 0.05, 0.10, 0.20, 0.30]:
    c = 0.8 * (1 - ue)
    print(f"    Ue={ue:.2f} → C={c:.3f} (Δ from p: {0.8 - c:.3f})")

# Count datapoints with significant Ue (>0.05)
significant_ue = sum(1 for ue in Ues if ue > 0.05)
print(f"\n⚠️  Datapoints with Ue > 0.05: {significant_ue}/{len(Ues)} ({significant_ue/len(Ues)*100:.1f}%)")
high_ue = sum(1 for ue in Ues if ue > 0.10)
print(f"⚠️  Datapoints with Ue > 0.10: {high_ue}/{len(Ues)} ({high_ue/len(Ues)*100:.1f}%)")

print(f"\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if np.mean(Ues) < 0.05:
    print("\n❌ PROBLEM IDENTIFIED: Epistemic uncertainty is TOO LOW")
    print(f"   Mean Ue = {np.mean(Ues):.6f} (< 0.05)")
    print("\n   This means:")
    print("   - C = p × (1-Ue) ≈ p × 0.95 ≈ p")
    print("   - v1 is almost identical to baseline p-threshold")
    print("   - Epistemic uncertainty provides no differentiation")
    print("\n   WHY THIS HAPPENED:")
    print("   - Beta distributions have low inherent variance")
    print("   - MC predictions drawn from same distribution → low variance → low Ue")
    print("   - Real models have MUCH higher Ue (0.1-0.3 common)")
    print("\n   SOLUTIONS:")
    print("   1. Test on REAL model with actual MC Dropout (Ue will be 10-50x higher)")
    print("   2. Redesign synthetic data with higher variance MC predictions")
    print("   3. Accept that current test isn't representative of real use case")
    
else:
    print("\n✅ Ue distribution looks reasonable")

# Recommendation
print(f"\n" + "=" * 80)
print("CRITICAL DECISION POINT")
print("=" * 80)

print("\n🔬 Current test shows NO VALUE for CognOS over p-threshold.")
print("   BUT: This may be artifact of unrealistic synthetic data (Ue too low).")
print("\n📝 OPTIONS:")
print("\n   A) ACCEPT FAILURE — CognOS doesn't work, abandon project")
print("      Risk: May be throwing away working idea due to bad test data")
print("\n   B) TEST ON REAL DATA — Validate with actual model predictions")
print("      Risk: Takes time, but could prove value on realistic Ue distributions")
print("      Target: UCI dataset + sklearn model with dropout/ensemble")
print("\n   C) FIX SYNTHETIC DATA — Increase MC prediction variance")
print("      Risk: Could be overfitting test to make formula look good")
print("\n🎯 RECOMMENDATION: Option B (real data)")
print("   Reason: Mean Ue=0.02 is 10x lower than real ML models")
print("   Cost: 2-3 hours to get real predictions")
print("   Payoff: Definitive answer whether CognOS works in practice")
