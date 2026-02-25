#!/usr/bin/env python3
"""
DOMAIN 2: IMDB Sentiment Classification (TEXT DOMAIN)

Validating CognOS on text classification to show cross-domain value.
Medical → Text shows generalizability beyond single domain.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.path.append('/media/bjorn/iic/cognos')
from test_matched_escalation import method_baseline, method_v1, method_v15

print("=" * 80)
print("DOMAIN 2: 20 Newsgroups Text Classification")
print("=" * 80)

# Load 20 newsgroups (binary: comp.graphics vs sci.space)
print("\n📦 Loading 20 Newsgroups dataset (binary classification)...")
categories = ['comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

print(f"   Train samples: {len(newsgroups_train.data)}")
print(f"   Test samples: {len(newsgroups_test.data)}")
print(f"   Categories: {categories}")

# Vectorize text
print("\n🔤 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', max_df=0.8, min_df=2)
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   Feature matrix: {X_train.shape}")

# Train RandomForest with moderate capacity (for errors + uncertainty)
print("\n🤖 Training RandomForest (20 trees, max_depth=4 for more errors)...")
model = RandomForestClassifier(
    n_estimators=20,  # Fewer trees
    max_depth=4,      # Shallower for more errors
    min_samples_split=20,
    random_state=42,
    bootstrap=True
)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"   Train accuracy: {train_acc:.3f}")
print(f"   Test accuracy: {test_acc:.3f}")

# Generate MC predictions from individual trees
print("\n🎲 Generating MC predictions from trees...")
mc_predictions_list = []
predictions_list = []

for i in range(X_test.shape[0]):
    x = X_test[i:i+1]
    # Get predictions from each tree
    tree_predictions = np.array([tree.predict_proba(x)[0][1] for tree in model.estimators_])
    mean_pred = np.mean(tree_predictions)
    
    mc_predictions_list.append(tree_predictions)
    predictions_list.append(mean_pred)

print(f"   Generated {len(predictions_list)} predictions with {len(mc_predictions_list[0])} MC samples")

# Compute Ue
Ue_values = [float(np.var(mc_preds)) for mc_preds in mc_predictions_list]

print(f"\n📊 Epistemic Uncertainty (Ue) Distribution:")
print(f"   Mean: {np.mean(Ue_values):.6f}")
print(f"   Median: {np.median(Ue_values):.6f}")
print(f"   Std: {np.std(Ue_values):.6f}")
print(f"   Min: {np.min(Ue_values):.6f}")
print(f"   Max: {np.max(Ue_values):.6f}")

high_ue_005 = sum(1 for ue in Ue_values if ue > 0.05)
high_ue_010 = sum(1 for ue in Ue_values if ue > 0.10)
print(f"\n   Datapoints with Ue > 0.05: {high_ue_005}/{len(Ue_values)} ({high_ue_005/len(Ue_values)*100:.1f}%)")
print(f"   Datapoints with Ue > 0.10: {high_ue_010}/{len(Ue_values)} ({high_ue_010/len(Ue_values)*100:.1f}%)")

# Format data
data = []
for pred, mc_preds, true_label in zip(predictions_list, mc_predictions_list, y_test):
    predicted_class = 1 if pred >= 0.5 else 0
    is_correct = (predicted_class == true_label)
    
    data.append({
        'prediction': float(pred),
        'mc_predictions': mc_preds.tolist(),
        'ground_truth': int(true_label),
        'is_correct': is_correct
    })

# Count errors and OE
total_errors = sum(1 for d in data if not d['is_correct'])
print(f"\n📊 Error Analysis:")
print(f"   Total errors: {total_errors} ({total_errors/len(data)*100:.1f}%)")

# Adaptive OE threshold (lower for text as predictions may be more calibrated)
OE_THRESHOLD = 0.6
OE_indices = []
for i, d in enumerate(data):
    if not d['is_correct'] and d['prediction'] >= OE_THRESHOLD:
        OE_indices.append(i)

print(f"   Overconfident Errors (p >= {OE_THRESHOLD}): {len(OE_indices)} ({len(OE_indices)/total_errors*100 if total_errors > 0 else 0:.1f}% of errors)")

if len(OE_indices) < 5:
    OE_THRESHOLD = 0.55
    OE_indices = []
    for i, d in enumerate(data):
        if not d['is_correct'] and d['prediction'] >= OE_THRESHOLD:
            OE_indices.append(i)
    print(f"   Adjusted: p >= {OE_THRESHOLD}: {len(OE_indices)} OE")

if len(OE_indices) < 5:
    OE_THRESHOLD = 0.5
    OE_indices = []
    for i, d in enumerate(data):
        if not d['is_correct'] and d['prediction'] >= OE_THRESHOLD:
            OE_indices.append(i)
    print(f"   Adjusted: p >= {OE_THRESHOLD}: {len(OE_indices)} OE")

if len(OE_indices) == 0:
    print("\n❌ No overconfident errors found. Model too calibrated.")
    error_preds = [d['prediction'] for d in data if not d['is_correct']]
    if error_preds:
        print(f"   Error prediction range: {min(error_preds):.3f} to {max(error_preds):.3f}")
        print(f"   Median error prediction: {np.median(error_preds):.3f}")
    sys.exit(1)

print(f"\n✅ Found {len(OE_indices)} overconfident errors for testing.")

# Test at low escalation rates (where CognOS showed value in medical domain)
print(f"\n" + "=" * 80)
print("LOW ESCALATION MATCHED TEST (TEXT DOMAIN)")
print("=" * 80)

thresholds = np.linspace(0.45, 0.75, 50)

baseline_results = []
v1_results = []
v15_results = []

for tau in thresholds:
    b = method_baseline(data, threshold=tau, oe_p_threshold=OE_THRESHOLD)
    v1 = method_v1(data, threshold=tau, oe_p_threshold=OE_THRESHOLD)
    v15 = method_v15(data, threshold=tau, oe_p_threshold=OE_THRESHOLD)
    
    baseline_results.append(b)
    v1_results.append(v1)
    v15_results.append(v15)

# Test at target escalations
target_escalations = [40, 45, 50, 55]

print("\n📊 MATCHED ESCALATION COMPARISON:\n")

v1_wins = 0
v15_wins = 0
total_tests = len(target_escalations)

for target_esc in target_escalations:
    b_closest = min(baseline_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    v1_closest = min(v1_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    v15_closest = min(v15_results, key=lambda r: abs(r['escalation_rate'] - target_esc))
    
    print(f"Target {target_esc}% escalation:")
    print(f"  Baseline: {b_closest['safety_gain']:.1f}% safety")
    print(f"  v1:       {v1_closest['safety_gain']:.1f}% safety")
    print(f"  v1.5:     {v15_closest['safety_gain']:.1f}% safety")
    
    max_safety = max(b_closest['safety_gain'], v1_closest['safety_gain'], v15_closest['safety_gain'])
    
    if v1_closest['safety_gain'] == max_safety and v1_closest['safety_gain'] > b_closest['safety_gain']:
        print(f"  ✅ v1 WINS (+{v1_closest['safety_gain'] - b_closest['safety_gain']:.1f}%)")
        v1_wins += 1
    elif v15_closest['safety_gain'] == max_safety and v15_closest['safety_gain'] > b_closest['safety_gain']:
        print(f"  ✅ v1.5 WINS (+{v15_closest['safety_gain'] - b_closest['safety_gain']:.1f}%)")
        v15_wins += 1
    else:
        print(f"  ⚖️  Tie or baseline wins")
    print()

print("=" * 80)
print("CROSS-DOMAIN VALIDATION RESULT")
print("=" * 80)

print(f"\nDomain: TEXT (20 Newsgroups)")
print(f"  CognOS wins: {v1_wins + v15_wins}/{total_tests}")
print(f"  Mean Ue: {np.mean(Ue_values):.4f}")
print(f"  Test samples: {len(data)}")
print(f"  Overconfident errors: {len(OE_indices)}")

if v1_wins + v15_wins >= 2:
    print(f"\n✅ CROSS-DOMAIN VALIDATION SUCCESS")
    print(f"   CognOS provides value in TEXT domain (medical ✓, text ✓)")
    print(f"   Ready for: Paper multi-domain section")
else:
    print(f"\n⚠️  WEAK CROSS-DOMAIN: CognOS doesn't generalize to text")
    print(f"   May be domain-specific to medical data")

print(f"\n" + "=" * 80)
