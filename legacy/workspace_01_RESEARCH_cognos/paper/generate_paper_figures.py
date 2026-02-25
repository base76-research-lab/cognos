#!/usr/bin/env python3
"""
Generate paper-quality figures from CognOS test results.

Figures to generate:
1. Pareto curves (Safety Gain vs Escalation Rate) - both domains
2. Operating regime plot (convergence at >70%)
3. Ue distribution comparison (synthetic vs real)
4. BOE/MOE bars at matched escalation points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_breast_cancer, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('/media/bjorn/iic/cognos')
from test_matched_escalation import method_baseline, method_v1, method_v15, generate_synthetic_data

# Set style for paper-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

print("=" * 80)
print("GENERATING PAPER FIGURES")
print("=" * 80)

# ============================================================================
# PREPARE DATA FROM BOTH DOMAINS
# ============================================================================

print("\n📦 Loading medical domain data...")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
model_med = RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_split=10, random_state=42, bootstrap=True)
model_med.fit(X_train, y_train)

medical_data = []
for x, true_label in zip(X_test, y_test):
    tree_predictions = np.array([tree.predict_proba([x])[0][1] for tree in model_med.estimators_])
    mean_pred = np.mean(tree_predictions)
    medical_data.append({
        'prediction': float(mean_pred),
        'mc_predictions': tree_predictions.tolist(),
        'ground_truth': int(true_label),
        'is_correct': (1 if mean_pred >= 0.5 else 0) == true_label
    })

print(f"   Medical: {len(medical_data)} samples")

print("\n📦 Loading text domain data...")
categories = ['comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', max_df=0.8, min_df=2)
X_train_text = vectorizer.fit_transform(newsgroups_train.data)
X_test_text = vectorizer.transform(newsgroups_test.data)
y_train_text = newsgroups_train.target
y_test_text = newsgroups_test.target

model_text = RandomForestClassifier(n_estimators=20, max_depth=4, min_samples_split=20, random_state=42, bootstrap=True)
model_text.fit(X_train_text, y_train_text)

text_data = []
for i in range(X_test_text.shape[0]):
    x = X_test_text[i:i+1]
    tree_predictions = np.array([tree.predict_proba(x)[0][1] for tree in model_text.estimators_])
    mean_pred = np.mean(tree_predictions)
    text_data.append({
        'prediction': float(mean_pred),
        'mc_predictions': tree_predictions.tolist(),
        'ground_truth': int(y_test_text[i]),
        'is_correct': (1 if mean_pred >= 0.5 else 0) == y_test_text[i]
    })

print(f"   Text: {len(text_data)} samples")

# ============================================================================
# FIGURE 1: PARETO CURVES (BOTH DOMAINS)
# ============================================================================

print("\n📊 Generating Figure 1: Pareto Curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Medical domain
thresholds = np.linspace(0.45, 0.95, 20)
baseline_med = [method_baseline(medical_data, tau, 0.7) for tau in thresholds]
v1_med = [method_v1(medical_data, tau, 0.7) for tau in thresholds]
v15_med = [method_v15(medical_data, tau, 0.7) for tau in thresholds]

ax1.plot([r['escalation_rate'] for r in baseline_med], [r['safety_gain'] for r in baseline_med], 
         'o-', label='Baseline (p-threshold)', color='#1f77b4', linewidth=2, markersize=4)
ax1.plot([r['escalation_rate'] for r in v1_med], [r['safety_gain'] for r in v1_med], 
         's-', label='v1 (Epistemic only)', color='#ff7f0e', linewidth=2, markersize=4)
ax1.plot([r['escalation_rate'] for r in v15_med], [r['safety_gain'] for r in v15_med], 
         '^-', label='CognOS (v1.5)', color='#2ca02c', linewidth=2, markersize=5)

# Highlight operating regime
ax1.axvspan(40, 55, alpha=0.15, color='green', label='Optimal regime')
ax1.axvspan(70, 100, alpha=0.1, color='gray', label='Ceiling effect')

ax1.set_xlabel('Escalation Rate (%)', fontweight='bold')
ax1.set_ylabel('Safety Gain (%)', fontweight='bold')
ax1.set_title('Medical Domain (UCI Breast Cancer)', fontweight='bold')
ax1.legend(loc='lower right', framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(35, 100)
ax1.set_ylim(-5, 105)

# Text domain
baseline_text = [method_baseline(text_data, tau, 0.55) for tau in thresholds]
v1_text = [method_v1(text_data, tau, 0.55) for tau in thresholds]
v15_text = [method_v15(text_data, tau, 0.55) for tau in thresholds]

ax2.plot([r['escalation_rate'] for r in baseline_text], [r['safety_gain'] for r in baseline_text], 
         'o-', label='Baseline', color='#1f77b4', linewidth=2, markersize=4)
ax2.plot([r['escalation_rate'] for r in v1_text], [r['safety_gain'] for r in v1_text], 
         's-', label='v1', color='#ff7f0e', linewidth=2, markersize=4)
ax2.plot([r['escalation_rate'] for r in v15_text], [r['safety_gain'] for r in v15_text], 
         '^-', label='CognOS', color='#2ca02c', linewidth=2, markersize=5)

ax2.axvspan(40, 55, alpha=0.15, color='green')
ax2.axvspan(70, 100, alpha=0.1, color='gray')

ax2.set_xlabel('Escalation Rate (%)', fontweight='bold')
ax2.set_ylabel('Safety Gain (%)', fontweight='bold')
ax2.set_title('Text Domain (20 Newsgroups)', fontweight='bold')
ax2.legend(loc='lower right', framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(35, 100)
ax2.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('figure1_pareto_curves.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: figure1_pareto_curves.png")
plt.close()

# ============================================================================
# FIGURE 2: OPERATING REGIME ANALYSIS
# ============================================================================

print("\n📊 Generating Figure 2: Operating Regime...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate advantage at each escalation rate (CognOS - Baseline)
escalation_rates_med = [r['escalation_rate'] for r in baseline_med]
advantages_med = [v15_med[i]['safety_gain'] - baseline_med[i]['safety_gain'] 
                  for i in range(len(baseline_med))]

escalation_rates_text = [r['escalation_rate'] for r in baseline_text]
advantages_text = [v15_text[i]['safety_gain'] - baseline_text[i]['safety_gain'] 
                   for i in range(len(baseline_text))]

ax.plot(escalation_rates_med, advantages_med, 'o-', label='Medical Domain', 
        color='#d62728', linewidth=2.5, markersize=6)
ax.plot(escalation_rates_text, advantages_text, 's-', label='Text Domain', 
        color='#9467bd', linewidth=2.5, markersize=6)

# Mark zero line
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Highlight regimes
ax.axvspan(40, 55, alpha=0.2, color='green', label='Value regime (40-55%)')
ax.axvspan(70, 100, alpha=0.15, color='gray', label='Convergence (>70%)')

# Add annotations
ax.annotate('CognOS provides\n40-100% advantage', xy=(47, 50), xytext=(47, 70),
            fontsize=10, ha='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax.annotate('Methods converge\n(ceiling effect)', xy=(85, 5), xytext=(85, 20),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax.set_xlabel('Escalation Rate (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Safety Gain Advantage\n(CognOS - Baseline, %)', fontweight='bold', fontsize=12)
ax.set_title('Operating Regime: Where CognOS Provides Value', fontweight='bold', fontsize=13)
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(35, 100)

plt.tight_layout()
plt.savefig('figure2_operating_regime.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: figure2_operating_regime.png")
plt.close()

# ============================================================================
# FIGURE 3: EPISTEMIC UNCERTAINTY DISTRIBUTIONS
# ============================================================================

print("\n📊 Generating Figure 3: Ue Distributions...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

# Synthetic data
np.random.seed(42)
synthetic_data = generate_synthetic_data(n=100)
Ue_synthetic = [float(np.var(d['mc_predictions'])) for d in synthetic_data]

# Real data
Ue_medical = [float(np.var(d['mc_predictions'])) for d in medical_data]
Ue_text = [float(np.var(d['mc_predictions'])) for d in text_data]

# Histogram 1: Synthetic
ax1.hist(Ue_synthetic, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.axvline(np.mean(Ue_synthetic), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(Ue_synthetic):.4f}')
ax1.set_xlabel('Epistemic Uncertainty (Ue)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Synthetic Data (Too Low)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Histogram 2: Medical
ax2.hist(Ue_medical, bins=25, color='#3498db', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(Ue_medical), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(Ue_medical):.4f}')
ax2.set_xlabel('Epistemic Uncertainty (Ue)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Medical Domain (Realistic)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Histogram 3: Text
ax3.hist(Ue_text, bins=25, color='#2ecc71', alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(Ue_text), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(Ue_text):.4f}')
ax3.set_xlabel('Epistemic Uncertainty (Ue)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Text Domain (Realistic)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_ue_distributions.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: figure3_ue_distributions.png")
plt.close()

# ============================================================================
# FIGURE 4: MATCHED ESCALATION BARS
# ============================================================================

print("\n📊 Generating Figure 4: Matched Escalation Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Medical domain - find results at ~45% escalation
target_esc = 45
baseline_45 = min(baseline_med, key=lambda r: abs(r['escalation_rate'] - target_esc))
v1_45 = min(v1_med, key=lambda r: abs(r['escalation_rate'] - target_esc))
v15_45 = min(v15_med, key=lambda r: abs(r['escalation_rate'] - target_esc))

methods = ['Baseline', 'v1', 'CognOS']
safety_gains_med = [baseline_45['safety_gain'], v1_45['safety_gain'], v15_45['safety_gain']]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars1 = ax1.bar(methods, safety_gains_med, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Safety Gain (%)', fontweight='bold', fontsize=11)
ax1.set_title(f'Medical Domain (Matched ~{target_esc}% Escalation)', fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, safety_gains_med):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Text domain
baseline_45_text = min(baseline_text, key=lambda r: abs(r['escalation_rate'] - target_esc))
v1_45_text = min(v1_text, key=lambda r: abs(r['escalation_rate'] - target_esc))
v15_45_text = min(v15_text, key=lambda r: abs(r['escalation_rate'] - target_esc))

safety_gains_text = [baseline_45_text['safety_gain'], v1_45_text['safety_gain'], v15_45_text['safety_gain']]

bars2 = ax2.bar(methods, safety_gains_text, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Safety Gain (%)', fontweight='bold', fontsize=11)
ax2.set_title(f'Text Domain (Matched ~{target_esc}% Escalation)', fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, safety_gains_text):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_matched_escalation.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: figure4_matched_escalation.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ALL FIGURES GENERATED")
print("=" * 80)
print("\nPaper-ready figures:")
print("  1. figure1_pareto_curves.png — Pareto frontiers (both domains)")
print("  2. figure2_operating_regime.png — Where CognOS provides value")
print("  3. figure3_ue_distributions.png — Synthetic vs real Ue comparison")
print("  4. figure4_matched_escalation.png — Matched escalation bars")
print("\nAll figures saved at 300 DPI in /media/bjorn/iic/cognos/")
print("Ready for paper insertion.")
