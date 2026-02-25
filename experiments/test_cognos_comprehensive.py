#!/usr/bin/env python3
"""
CognOS v1.5 — Comprehensive Test Suite

Testar confidence engine med korrekta metrics:
- OE definierat som fel + C ≥ threshold (inte p ≥ 0.8)
- Threshold sweep 0.5-0.95
- Reliability diagram
- Cost-safety tradeoff

Paper-ready results.
"""

import numpy as np
from confidence import compute_confidence
from typing import List, Dict, Tuple


def generate_synthetic_data(n=100, seed=42):
    """
    Genererar syntetisk data med varierad confidence.
    
    CognOS hypotes: Modeller som är fel är ofta osäkra (hög Ue).
    Men ibland är de fel OCH säkra (överkonfidenta) → farligt.
    
    Returns:
        List of (prediction, mc_predictions, ground_truth)
    """
    np.random.seed(seed)
    data = []
    
    for i in range(n):
        ground_truth = np.random.choice([0, 1])
        
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
                prediction = np.random.beta(6, 4)
            else:
                prediction = np.random.beta(4, 6)
            mc_noise = np.random.normal(0, 0.12, 5)
            uncertainty_level = 'high'
            
        elif scenario == 'wrong_confident':
            # FEL prediction, men inte helt övertygad
            if ground_truth == 1:
                prediction = np.random.beta(2, 8)
            else:
                prediction = np.random.beta(8, 2)
            mc_noise = np.random.normal(0, 0.05, 5)
            uncertainty_level = 'medium'
            
        else:  # wrong_uncertain
            # FEL prediction, hög osäkerhet
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


def evaluate_cognos(data: List[Dict], threshold: float = 0.8) -> Dict:
    """
    Evaluerar CognOS med KORREKT OE-definition:
    OE = fel + C ≥ threshold (inte p ≥ 0.8)
    
    Detta är ärligt: vi definierar OE baserat på CognOS egen confidence-score.
    """
    total = len(data)
    correct = 0
    wrong = 0
    
    # Räkna C för alla datapunkter
    all_confidences = []
    all_decisions = []
    all_correct = []
    
    for d in data:
        result = compute_confidence(d['prediction'], d['mc_predictions'], threshold=threshold)
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        
        all_confidences.append(result['confidence'])
        all_decisions.append(result['decision'])
        all_correct.append(is_correct)
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
    
    # OE = fel + hög confidence (C ≥ threshold)
    OE_indices = []
    for i, (conf, is_correct) in enumerate(zip(all_confidences, all_correct)):
        if not is_correct and conf >= threshold:
            OE_indices.append(i)
    
    # BOE = OE som eskalerades (blockerat)
    # MOE = OE som gick till auto (missat)
    BOE_indices = []
    MOE_indices = []
    
    for idx in OE_indices:
        if all_decisions[idx] == 'escalate':
            BOE_indices.append(idx)
        else:
            MOE_indices.append(idx)
    
    # Auto/escalate stats
    auto_count = sum(1 for d in all_decisions if d == 'auto')
    escalate_count = sum(1 for d in all_decisions if d == 'escalate')
    
    auto_correct = sum(1 for i, (d, c) in enumerate(zip(all_decisions, all_correct)) 
                       if d == 'auto' and c)
    
    OE = len(OE_indices)
    BOE = len(BOE_indices)
    MOE = len(MOE_indices)
    
    safety_gain = (BOE / OE * 100) if OE > 0 else 0
    auto_accuracy = (auto_correct / auto_count) if auto_count > 0 else 0
    escalation_rate = escalate_count / total
    
    return {
        'total': total,
        'correct': correct,
        'wrong': wrong,
        'OE': OE,
        'BOE': BOE,
        'MOE': MOE,
        'safety_gain': safety_gain,
        'auto_count': auto_count,
        'escalate_count': escalate_count,
        'auto_accuracy': auto_accuracy,
        'escalation_rate': escalation_rate,
        'all_confidences': all_confidences,
        'all_decisions': all_decisions,
        'all_correct': all_correct,
        'OE_indices': OE_indices,
        'BOE_indices': BOE_indices,
        'MOE_indices': MOE_indices
    }


def threshold_sweep(data: List[Dict], thresholds: List[float]) -> List[Dict]:
    """Experiment A: Threshold sweep 0.5-0.95"""
    results = []
    for tau in thresholds:
        result = evaluate_cognos(data, threshold=tau)
        results.append({
            'threshold': tau,
            'safety_gain': result['safety_gain'],
            'escalation_rate': result['escalation_rate'],
            'auto_accuracy': result['auto_accuracy'],
            'OE': result['OE'],
            'BOE': result['BOE']
        })
    return results


def reliability_diagram(data: List[Dict], threshold: float = 0.8, n_bins: int = 10):
    """Experiment B: Reliability diagram (confidence vs accuracy)"""
    result = evaluate_cognos(data, threshold=threshold)
    
    confidences = np.array(result['all_confidences'])
    correct = np.array(result['all_correct'], dtype=float)
    
    # Binning
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
    
    return {
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts
    }


def cost_safety_tradeoff(data: List[Dict], thresholds: List[float],
                          cost_auto: float = 0.01, cost_escalate: float = 0.10):
    """Experiment C: Cost-safety tradeoff"""
    results = []
    for tau in thresholds:
        result = evaluate_cognos(data, threshold=tau)
        
        total_cost = (result['auto_count'] * cost_auto + 
                      result['escalate_count'] * cost_escalate)
        avg_cost = total_cost / result['total']
        
        results.append({
            'threshold': tau,
            'safety_gain': result['safety_gain'],
            'avg_cost': avg_cost,
            'escalation_rate': result['escalation_rate']
        })
    return results


if __name__ == '__main__':
    print("=" * 70)
    print("CognOS v1.5 — Comprehensive Test Suite")
    print("=" * 70)
    
    # Generate data
    print("\n📊 Genererar 100 syntetiska datapunkter...")
    data = generate_synthetic_data(100)
    
    # Baseline evaluation (threshold=0.8)
    print("\n" + "=" * 70)
    print("1️⃣  Baseline Evaluation (τ=0.8)")
    print("=" * 70)
    
    result = evaluate_cognos(data, threshold=0.8)
    
    print(f"\n📈 Overall Stats:")
    print(f"  Total: {result['total']}")
    print(f"  Correct: {result['correct']} ({result['correct']/result['total']*100:.1f}%)")
    print(f"  Wrong: {result['wrong']} ({result['wrong']/result['total']*100:.1f}%)")
    
    print(f"\n🎯 Overconfident Errors (OE = fel + C ≥ τ):")
    print(f"  OE: {result['OE']}")
    print(f"  BOE (blocked): {result['BOE']}")
    print(f"  MOE (missed): {result['MOE']}")
    print(f"  Safety Gain: {result['safety_gain']:.1f}%")
    
    print(f"\n🚦 Decision Stats:")
    print(f"  Auto: {result['auto_count']} ({result['auto_count']/result['total']*100:.1f}%)")
    print(f"  Escalated: {result['escalate_count']} ({result['escalation_rate']*100:.1f}%)")
    print(f"  Auto Accuracy: {result['auto_accuracy']:.1%}")
    
    # Experiment A: Threshold sweep
    print("\n" + "=" * 70)
    print("2️⃣  Experiment A: Threshold Sweep (0.5 → 0.95)")
    print("=" * 70)
    
    thresholds = np.linspace(0.5, 0.95, 10)
    sweep_results = threshold_sweep(data, thresholds)
    
    print(f"\n{'Threshold':<12} {'Safety Gain':<15} {'Escalation Rate':<18} {'Auto Accuracy'}")
    print("-" * 70)
    for r in sweep_results:
        print(f"{r['threshold']:.2f}         {r['safety_gain']:>6.1f}%         "
              f"{r['escalation_rate']*100:>6.1f}%              {r['auto_accuracy']:.1%}")
    
    # Experiment B: Reliability diagram
    print("\n" + "=" * 70)
    print("3️⃣  Experiment B: Reliability Diagram")
    print("=" * 70)
    
    reliability = reliability_diagram(data, threshold=0.8)
    
    print(f"\n{'Confidence Bin':<18} {'Accuracy':<12} {'Count'}")
    print("-" * 50)
    for center, acc, count in zip(reliability['bin_centers'], 
                                   reliability['bin_accuracies'], 
                                   reliability['bin_counts']):
        if not np.isnan(acc):
            print(f"{center:.2f}              {acc:.1%}        {count}")
    
    # Experiment C: Cost-safety tradeoff
    print("\n" + "=" * 70)
    print("4️⃣  Experiment C: Cost-Safety Tradeoff")
    print("=" * 70)
    print("(cost_auto=0.01, cost_escalate=0.10)\n")
    
    cost_results = cost_safety_tradeoff(data, thresholds)
    
    print(f"{'Threshold':<12} {'Safety Gain':<15} {'Avg Cost':<12} {'Escalation Rate'}")
    print("-" * 70)
    for r in cost_results:
        print(f"{r['threshold']:.2f}         {r['safety_gain']:>6.1f}%         "
              f"{r['avg_cost']:>6.4f}       {r['escalation_rate']*100:>6.1f}%")
    
    # GO/NO-GO decision
    print("\n" + "=" * 70)
    print("5️⃣  GO/NO-GO Decision")
    print("=" * 70)
    
    baseline_result = result  # τ=0.8
    
    if baseline_result['OE'] > 0:
        if baseline_result['safety_gain'] >= 30:
            print(f"\n✅ GO: CognOS v1.5 achieves {baseline_result['safety_gain']:.1f}% Safety Gain")
            print(f"   (Target: ≥30%)")
            print(f"   Blocked {baseline_result['BOE']}/{baseline_result['OE']} overconfident errors")
            print(f"   Escalation rate: {baseline_result['escalation_rate']*100:.1f}%")
            print(f"   Auto accuracy: {baseline_result['auto_accuracy']:.1%}")
            print(f"\n📄 Paper-Ready: 'Confidence-Driven Model Arbitration for Safer Autonomous Systems'")
        else:
            print(f"\n⚠️  NO-GO: Safety Gain {baseline_result['safety_gain']:.1f}% < 30% threshold")
    else:
        print("\n⚠️  N/A: No overconfident errors in dataset")
    
    print("\n" + "=" * 70)
