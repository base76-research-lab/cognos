#!/usr/bin/env python3
"""
Debug: Inspect prediction distribution and fix matched escalation test
"""

import numpy as np
from test_matched_escalation import generate_synthetic_data

np.random.seed(42)
data = generate_synthetic_data(n=100)

# Extract predictions
predictions = [d['prediction'] for d in data]

print("=" * 80)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("=" * 80)

print(f"\nTotal datapoints: {len(predictions)}")
print(f"Mean prediction: {np.mean(predictions):.3f}")
print(f"Median prediction: {np.median(predictions):.3f}")
print(f"Std prediction: {np.std(predictions):.3f}")
print(f"Min prediction: {np.min(predictions):.3f}")
print(f"Max prediction: {np.max(predictions):.3f}")

# Percentiles
percentiles = [10, 25, 50, 75, 90, 95, 98, 99]
print("\nPercentiles:")
for p in percentiles:
    val = np.percentile(predictions, p)
    print(f"  {p}th: {val:.3f}")

# Count predictions above various thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
print("\nPredictions BELOW threshold (would be escalated by baseline):")
for tau in thresholds:
    below = sum(1 for p in predictions if p < tau)
    pct = (below / len(predictions)) * 100
    print(f"  τ={tau:.2f}: {below} predictions ({pct:.1f}% escalation)")

# To get 98% escalation, we need threshold where only 2% of predictions are above
target_threshold = np.percentile(predictions, 98)
print(f"\nTo achieve 98% escalation:")
print(f"  Baseline needs τ={target_threshold:.4f}")
print(f"  (98th percentile of prediction distribution)")

# Count by scenario
scenarios = {}
for d in data:
    s = d['scenario']
    if s not in scenarios:
        scenarios[s] = []
    scenarios[s].append(d['prediction'])

print("\nPrediction statistics by scenario:")
for s, preds in scenarios.items():
    print(f"\n  {s} (n={len(preds)}):")
    print(f"    Mean: {np.mean(preds):.3f}")
    print(f"    Min: {np.min(preds):.3f}")
    print(f"    Max: {np.max(preds):.3f}")
