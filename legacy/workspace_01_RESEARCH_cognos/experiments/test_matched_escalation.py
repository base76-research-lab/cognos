#!/usr/bin/env python3
"""
KRITISK TEST: Matched Escalation Rate Comparison

Fråga: Är v1.5 bättre för att den har bättre formel, eller bara för att den eskalerar mer?

Test:
1. Justera baseline threshold tills den matchar v1.5's escalation rate (~93%)
2. Jämför SafetyGain vid SAMMA escalation rate
3. Threshold sweep (τ=0.5→0.95) för alla metoder
4. Plotta Pareto curves: SafetyGain vs EscalationRate

Om v1.5 fortfarande har högre SafetyGain vid matchad escalation → STARKT resultat, publicerbart.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Seed för reproducerbarhet
np.random.seed(42)

def generate_synthetic_data(n: int = 100) -> List[Dict]:
    """
    Generera syntetisk data med 4 scenarios:
    - correct_confident: 50% (hög p, låg Ue, rätt)
    - correct_uncertain: 20% (medel p, hög Ue, rätt)
    - wrong_confident: 15% (hög p, låg Ue, FEL) ← FARLIGA
    - wrong_uncertain: 15% (låg p, hög Ue, fel)
    """
    data = []
    
    # Correct confident (50)
    for _ in range(50):
        p = np.random.beta(8, 2)  # Hög confidence
        mc_preds = np.random.beta(8, 2, size=10)  # Låg varians
        data.append({
            'prediction': float(p),
            'mc_predictions': mc_preds.tolist(),
            'ground_truth': 1 if p >= 0.5 else 0,
            'scenario': 'correct_confident'
        })
    
    # Correct uncertain (20)
    for _ in range(20):
        p = np.random.beta(3, 3)  # Medel confidence
        mc_preds = np.random.beta(3, 3, size=10)  # Hög varians
        data.append({
            'prediction': float(p),
            'mc_predictions': mc_preds.tolist(),
            'ground_truth': 1 if p >= 0.5 else 0,
            'scenario': 'correct_uncertain'
        })
    
    # Wrong confident (15) ← KRITISKA: överkonfidenta fel
    for _ in range(15):
        p = np.random.beta(8, 2)
        mc_preds = np.random.beta(8, 2, size=10)
        # Flip ground truth så den blir fel
        gt = 0 if p >= 0.5 else 1
        data.append({
            'prediction': float(p),
            'mc_predictions': mc_preds.tolist(),
            'ground_truth': gt,
            'scenario': 'wrong_confident'
        })
    
    # Wrong uncertain (15)
    for _ in range(15):
        p = np.random.beta(2, 8)  # Låg confidence
        mc_preds = np.random.beta(2, 8, size=10)
        gt = 0 if p >= 0.5 else 1
        data.append({
            'prediction': float(p),
            'mc_predictions': mc_preds.tolist(),
            'ground_truth': gt,
            'scenario': 'wrong_uncertain'
        })
    
    return data

def method_baseline(data: List[Dict], threshold: float = 0.8, oe_p_threshold: float = 0.8) -> Dict:
    """Method A: Simple p-threshold"""
    decisions = []
    
    for d in data:
        decision = 'auto' if d['prediction'] >= threshold else 'escalate'
        decisions.append({
            'decision': decision,
            'confidence': d['prediction'],
            'is_correct': (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        })
    
    # Compute metrics
    OE_indices = []
    for i, d in enumerate(data):
        is_correct = (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        if not is_correct and d['prediction'] >= oe_p_threshold:
            OE_indices.append(i)
    
    BOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'escalate')
    MOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'auto')
    
    num_escalate = sum(1 for dec in decisions if dec['decision'] == 'escalate')
    auto_decisions = [dec for dec in decisions if dec['decision'] == 'auto']
    auto_accuracy = sum(1 for dec in auto_decisions if dec['is_correct']) / len(auto_decisions) if auto_decisions else 0
    
    return {
        'method': 'Baseline',
        'threshold': threshold,
        'OE': len(OE_indices),
        'BOE': BOE,
        'MOE': MOE,
        'safety_gain': (BOE / len(OE_indices) * 100) if OE_indices else 0,
        'escalation_rate': (num_escalate / len(data)) * 100,
        'auto_accuracy': auto_accuracy * 100
    }

def method_v1(data: List[Dict], threshold: float = 0.8, oe_p_threshold: float = 0.8) -> Dict:
    """Method B: v1 (epistemic only)"""
    decisions = []
    
    for d in data:
        Ue = float(np.var(d['mc_predictions']))
        C = max(0.0, d['prediction'] * (1 - Ue))
        decision = 'auto' if C >= threshold else 'escalate'
        decisions.append({
            'decision': decision,
            'confidence': C,
            'is_correct': (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        })
    
    # Compute metrics (uniform OE definition)
    OE_indices = []
    for i, d in enumerate(data):
        is_correct = (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        if not is_correct and d['prediction'] >= oe_p_threshold:
            OE_indices.append(i)
    
    BOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'escalate')
    MOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'auto')
    
    num_escalate = sum(1 for dec in decisions if dec['decision'] == 'escalate')
    auto_decisions = [dec for dec in decisions if dec['decision'] == 'auto']
    auto_accuracy = sum(1 for dec in auto_decisions if dec['is_correct']) / len(auto_decisions) if auto_decisions else 0
    
    return {
        'method': 'v1',
        'threshold': threshold,
        'OE': len(OE_indices),
        'BOE': BOE,
        'MOE': MOE,
        'safety_gain': (BOE / len(OE_indices) * 100) if OE_indices else 0,
        'escalation_rate': (num_escalate / len(data)) * 100,
        'auto_accuracy': auto_accuracy * 100
    }

def method_v15(data: List[Dict], threshold: float = 0.8, oe_p_threshold: float = 0.8) -> Dict:
    """Method C: v1.5 (epistemic + aleatoric)"""
    decisions = []
    
    for d in data:
        Ue = float(np.var(d['mc_predictions']))
        Ua = 2 * d['prediction'] * (1 - d['prediction'])
        C = max(0.0, d['prediction'] * (1 - Ue - Ua))
        decision = 'auto' if C >= threshold else 'escalate'
        decisions.append({
            'decision': decision,
            'confidence': C,
            'is_correct': (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        })
    
    # Compute metrics (uniform OE definition)
    OE_indices = []
    for i, d in enumerate(data):
        is_correct = (1 if d['prediction'] >= 0.5 else 0) == d['ground_truth']
        if not is_correct and d['prediction'] >= oe_p_threshold:
            OE_indices.append(i)
    
    BOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'escalate')
    MOE = sum(1 for i in OE_indices if decisions[i]['decision'] == 'auto')
    
    num_escalate = sum(1 for dec in decisions if dec['decision'] == 'escalate')
    auto_decisions = [dec for dec in decisions if dec['decision'] == 'auto']
    auto_accuracy = sum(1 for dec in auto_decisions if dec['is_correct']) / len(auto_decisions) if auto_decisions else 0
    
    return {
        'method': 'v1.5',
        'threshold': threshold,
        'OE': len(OE_indices),
        'BOE': BOE,
        'MOE': MOE,
        'safety_gain': (BOE / len(OE_indices) * 100) if OE_indices else 0,
        'escalation_rate': (num_escalate / len(data)) * 100,
        'auto_accuracy': auto_accuracy * 100
    }

def find_threshold_for_escalation_rate(data: List[Dict], method_func, target_escalation: float, tolerance: float = 1.0) -> Tuple[float, Dict]:
    """
    Hitta threshold som ger önskad escalation rate (inom tolerance %).
    Test multiple thresholds and pick closest match.
    """
    # Test a range of thresholds
    test_thresholds = np.linspace(0.5, 0.99, 50)
    
    best_threshold = 0.8
    best_result = None
    best_diff = float('inf')
    
    for tau in test_thresholds:
        result = method_func(data, threshold=tau)
        diff = abs(result['escalation_rate'] - target_escalation)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = tau
            best_result = result
            
            if diff <= tolerance:
                return best_threshold, best_result
    
    return best_threshold, best_result

def threshold_sweep(data: List[Dict], thresholds: List[float]) -> Dict[str, List[Dict]]:
    """
    Kör alla metoder för varje threshold och returnera resultat.
    """
    results = {
        'baseline': [],
        'v1': [],
        'v1.5': []
    }
    
    for tau in thresholds:
        results['baseline'].append(method_baseline(data, threshold=tau))
        results['v1'].append(method_v1(data, threshold=tau))
        results['v1.5'].append(method_v15(data, threshold=tau))
    
    return results

def plot_pareto_curves(sweep_results: Dict[str, List[Dict]], save_path: str = 'pareto_curves.png'):
    """
    Plotta Pareto curves: SafetyGain vs EscalationRate
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'baseline': 'blue', 'v1': 'orange', 'v1.5': 'green'}
    
    # Plot 1: SafetyGain vs EscalationRate
    for method, results in sweep_results.items():
        esc_rates = [r['escalation_rate'] for r in results]
        safety_gains = [r['safety_gain'] for r in results]
        ax1.plot(esc_rates, safety_gains, marker='o', label=method, color=colors[method], linewidth=2)
    
    ax1.set_xlabel('Escalation Rate (%)', fontsize=12)
    ax1.set_ylabel('Safety Gain (%)', fontsize=12)
    ax1.set_title('Safety Gain vs Escalation Rate (Pareto Frontier)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AutoAccuracy vs EscalationRate
    for method, results in sweep_results.items():
        esc_rates = [r['escalation_rate'] for r in results]
        auto_accs = [r['auto_accuracy'] for r in results]
        ax2.plot(esc_rates, auto_accs, marker='o', label=method, color=colors[method], linewidth=2)
    
    ax2.set_xlabel('Escalation Rate (%)', fontsize=12)
    ax2.set_ylabel('Auto Accuracy (%)', fontsize=12)
    ax2.set_title('Auto Accuracy vs Escalation Rate', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Pareto curves saved to: {save_path}")
    plt.close()

def plot_confusion_bars(baseline_result: Dict, v1_result: Dict, v15_result: Dict, save_path: str = 'confusion_bars.png'):
    """
    Plotta BOE/MOE stacked bars för alla metoder.
    """
    methods = ['Baseline\n(matched)', 'v1\n(τ=0.8)', 'v1.5\n(τ=0.8)']
    BOE = [baseline_result['BOE'], v1_result['BOE'], v15_result['BOE']]
    MOE = [baseline_result['MOE'], v1_result['MOE'], v15_result['MOE']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(methods))
    width = 0.5
    
    p1 = ax.bar(x, BOE, width, label='BOE (Blocked)', color='green', alpha=0.8)
    p2 = ax.bar(x, MOE, width, bottom=BOE, label='MOE (Missed)', color='red', alpha=0.8)
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overconfident Error Handling: BOE vs MOE', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (b, m) in enumerate(zip(BOE, MOE)):
        total = b + m
        ax.text(i, total + 0.2, f'{total}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion bars saved to: {save_path}")
    plt.close()

def plot_threshold_sweep(sweep_results: Dict[str, List[Dict]], save_path: str = 'threshold_sweep.png'):
    """
    Plotta threshold (τ) vs SafetyGain och EscalationRate.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'baseline': 'blue', 'v1': 'orange', 'v1.5': 'green'}
    
    # Plot 1: τ vs SafetyGain
    for method, results in sweep_results.items():
        thresholds = [r['threshold'] for r in results]
        safety_gains = [r['safety_gain'] for r in results]
        ax1.plot(thresholds, safety_gains, marker='o', label=method, color=colors[method], linewidth=2)
    
    ax1.set_xlabel('Threshold (τ)', fontsize=12)
    ax1.set_ylabel('Safety Gain (%)', fontsize=12)
    ax1.set_title('Threshold vs Safety Gain', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: τ vs EscalationRate
    for method, results in sweep_results.items():
        thresholds = [r['threshold'] for r in results]
        esc_rates = [r['escalation_rate'] for r in results]
        ax2.plot(thresholds, esc_rates, marker='o', label=method, color=colors[method], linewidth=2)
    
    ax2.set_xlabel('Threshold (τ)', fontsize=12)
    ax2.set_ylabel('Escalation Rate (%)', fontsize=12)
    ax2.set_title('Threshold vs Escalation Rate', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Threshold sweep saved to: {save_path}")
    plt.close()

def main():
    print("=" * 80)
    print("KRITISK TEST: Matched Escalation Rate Comparison")
    print("=" * 80)
    
    # Generate data
    data = generate_synthetic_data(n=100)
    print(f"\n✅ Generated {len(data)} synthetic datapoints")
    
    # Part 1: Find v1.5's escalation rate at τ=0.8
    print("\n" + "=" * 80)
    print("PART 1: Find v1.5's escalation rate (baseline measurement)")
    print("=" * 80)
    
    v15_baseline = method_v15(data, threshold=0.8)
    target_escalation = v15_baseline['escalation_rate']
    
    print(f"\nv1.5 (τ=0.8):")
    print(f"  Escalation Rate: {v15_baseline['escalation_rate']:.1f}%")
    print(f"  Safety Gain: {v15_baseline['safety_gain']:.1f}%")
    print(f"  BOE/MOE: {v15_baseline['BOE']}/{v15_baseline['MOE']}")
    
    # Part 2: Match baseline to same escalation rate
    print("\n" + "=" * 80)
    print(f"PART 2: Match baseline escalation rate to ~{target_escalation:.1f}%")
    print("=" * 80)
    
    matched_threshold, baseline_matched = find_threshold_for_escalation_rate(
        data, method_baseline, target_escalation, tolerance=1.0
    )
    
    print(f"\n🎯 Found baseline threshold: τ={matched_threshold:.4f}")
    print(f"  Escalation Rate: {baseline_matched['escalation_rate']:.1f}% (target: {target_escalation:.1f}%)")
    print(f"  Safety Gain: {baseline_matched['safety_gain']:.1f}%")
    print(f"  BOE/MOE: {baseline_matched['BOE']}/{baseline_matched['MOE']}")
    
    # Part 3: Compare at matched escalation rate
    print("\n" + "=" * 80)
    print("PART 3: CRITICAL COMPARISON (Matched Escalation Rate)")
    print("=" * 80)
    
    # Also get v1 at τ=0.8 for comparison
    v1_baseline = method_v1(data, threshold=0.8)
    
    print("\n📊 RESULTS AT MATCHED ESCALATION (~93%):\n")
    print(f"{'Method':<20} {'Esc%':<10} {'SafetyGain%':<15} {'BOE':<8} {'MOE':<8} {'AutoAcc%':<10}")
    print("-" * 80)
    print(f"{'Baseline (matched)':<20} {baseline_matched['escalation_rate']:<10.1f} {baseline_matched['safety_gain']:<15.1f} {baseline_matched['BOE']:<8} {baseline_matched['MOE']:<8} {baseline_matched['auto_accuracy']:<10.1f}")
    print(f"{'v1 (τ=0.8)':<20} {v1_baseline['escalation_rate']:<10.1f} {v1_baseline['safety_gain']:<15.1f} {v1_baseline['BOE']:<8} {v1_baseline['MOE']:<8} {v1_baseline['auto_accuracy']:<10.1f}")
    print(f"{'v1.5 (τ=0.8)':<20} {v15_baseline['escalation_rate']:<10.1f} {v15_baseline['safety_gain']:<15.1f} {v15_baseline['BOE']:<8} {v15_baseline['MOE']:<8} {v15_baseline['auto_accuracy']:<10.1f}")
    
    # CRITICAL VERDICT
    print("\n" + "=" * 80)
    print("🔬 CRITICAL VERDICT")
    print("=" * 80)
    
    if v15_baseline['safety_gain'] > baseline_matched['safety_gain']:
        diff = v15_baseline['safety_gain'] - baseline_matched['safety_gain']
        print(f"\n✅ GO: v1.5 ger {diff:.1f}% HÖGRE SafetyGain vid SAMMA escalation rate.")
        print(f"   Detta bevisar att vinsten INTE bara är 'mer escalation = fler stoppade fel'.")
        print(f"   v1.5's formel är BÄTTRE på att välja VILKA predictions som ska eskaleras.")
        print("\n📝 PUBLICERBART RESULTAT — Detta är core finding för paper.")
    else:
        print(f"\n❌ NO-GO: v1.5 ger INTE högre SafetyGain vid matchad escalation.")
        print(f"   Vinsten var bara artifact av högre escalation rate.")
        print(f"   Formeln behöver omarbetas.")
    
    # Part 4: Threshold sweep
    print("\n" + "=" * 80)
    print("PART 4: Threshold Sweep (τ=0.5→0.95)")
    print("=" * 80)
    
    thresholds = np.linspace(0.5, 0.95, 10)
    sweep_results = threshold_sweep(data, thresholds)
    
    print("\n✅ Completed threshold sweep for all methods")
    
    # Part 5: Generate figures
    print("\n" + "=" * 80)
    print("PART 5: Generate Paper-Ready Figures")
    print("=" * 80)
    
    plot_pareto_curves(sweep_results, 'pareto_curves.png')
    plot_confusion_bars(baseline_matched, v1_baseline, v15_baseline, 'confusion_bars.png')
    plot_threshold_sweep(sweep_results, 'threshold_sweep.png')
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - pareto_curves.png (SafetyGain vs Escalation, AutoAcc vs Escalation)")
    print("  - confusion_bars.png (BOE/MOE stacked bars)")
    print("  - threshold_sweep.png (τ vs SafetyGain, τ vs Escalation)")

if __name__ == "__main__":
    main()
