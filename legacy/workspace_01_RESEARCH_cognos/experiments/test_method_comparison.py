#!/usr/bin/env python3
"""
CognOS — Critical Comparison Test

Testar tre metoder för att bevisa värde:
- Method A (baseline): Simple p-threshold
- Method B (v1): C = p × (1 - Ue)  
- Method C (v1.5): C = p × (1 - Ue - Ua)

Fråga: Ger CognOS bättre riskkontroll än enkla heuristiker?
"""

import numpy as np
from typing import List, Dict, Tuple


def generate_synthetic_data(n=100, seed=42):
    """Samma datagen som tidigare"""
    np.random.seed(seed)
    data = []
    
    for i in range(n):
        ground_truth = np.random.choice([0, 1])
        scenario = np.random.choice(['correct_confident', 'correct_uncertain', 
                                     'wrong_confident', 'wrong_uncertain'], 
                                    p=[0.5, 0.2, 0.15, 0.15])
        
        if scenario == 'correct_confident':
            if ground_truth == 1:
                prediction = np.random.beta(8, 2)
            else:
                prediction = np.random.beta(2, 8)
            mc_noise = np.random.normal(0, 0.01, 5)
            
        elif scenario == 'correct_uncertain':
            if ground_truth == 1:
                prediction = np.random.beta(6, 4)
            else:
                prediction = np.random.beta(4, 6)
            mc_noise = np.random.normal(0, 0.12, 5)
            
        elif scenario == 'wrong_confident':
            if ground_truth == 1:
                prediction = np.random.beta(2, 8)
            else:
                prediction = np.random.beta(8, 2)
            mc_noise = np.random.normal(0, 0.05, 5)
            
        else:  # wrong_uncertain
            if ground_truth == 1:
                prediction = np.random.beta(3, 7)
            else:
                prediction = np.random.beta(7, 3)
            mc_noise = np.random.normal(0, 0.15, 5)
        
        mc_base = prediction
        mc_predictions = np.clip(mc_base + mc_noise, 0, 1).tolist()
        
        data.append({
            'prediction': prediction,
            'mc_predictions': mc_predictions,
            'ground_truth': ground_truth,
            'scenario': scenario
        })
    
    return data


def method_a_simple_threshold(data: List[Dict], p_threshold: float = 0.8, 
                                confidence_threshold: float = 0.8) -> Dict:
    """
    Method A (Baseline): Simple p-threshold
    
    OE = fel + p ≥ p_threshold
    Blockera allt med p ≥ confidence_threshold
    """
    total = len(data)
    correct = 0
    wrong = 0
    
    # Identify all wrong predictions
    wrong_predictions = []
    for i, d in enumerate(data):
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
            wrong_predictions.append(i)
    
    # OE = fel + p ≥ p_threshold (classic definition)
    OE_indices = []
    for idx in wrong_predictions:
        d = data[idx]
        if d['prediction'] >= p_threshold:
            OE_indices.append(idx)
    
    # Decision: escalate if p < confidence_threshold
    BOE_indices = []
    MOE_indices = []
    auto_count = 0
    escalate_count = 0
    auto_correct = 0
    
    for i, d in enumerate(data):
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        
        if d['prediction'] >= confidence_threshold:
            auto_count += 1
            if is_correct:
                auto_correct += 1
            if i in OE_indices:
                MOE_indices.append(i)  # Missed
        else:
            escalate_count += 1
            if i in OE_indices:
                BOE_indices.append(i)  # Blocked
    
    OE = len(OE_indices)
    BOE = len(BOE_indices)
    MOE = len(MOE_indices)
    
    safety_gain = (BOE / OE * 100) if OE > 0 else 0
    auto_accuracy = (auto_correct / auto_count) if auto_count > 0 else 0
    escalation_rate = escalate_count / total
    
    return {
        'method': 'Simple p-threshold',
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
        'escalation_rate': escalation_rate
    }


def method_b_epistemic_only(data: List[Dict], threshold: float = 0.8, 
                             oe_p_threshold: float = 0.8) -> Dict:
    """
    Method B (v1): C = p × (1 - Ue)
    
    OE = fel + p ≥ oe_p_threshold (SAMMA för alla metoder)
    """
    total = len(data)
    correct = 0
    wrong = 0
    
    # Compute C for all
    all_confidences = []
    all_decisions = []
    all_correct = []
    
    for d in data:
        # C = p × (1 - Ue)
        Ue = float(np.var(d['mc_predictions']))
        C = d['prediction'] * (1 - Ue)
        
        decision = 'auto' if C >= threshold else 'escalate'
        
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        
        all_confidences.append(C)
        all_decisions.append(decision)
        all_correct.append(is_correct)
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
    
    # OE = fel + p ≥ oe_p_threshold (UNIFORM definition)
    OE_indices = []
    for i, d in enumerate(data):
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        if not is_correct and d['prediction'] >= oe_p_threshold:
            OE_indices.append(i)
    
    # BOE/MOE
    BOE_indices = []
    MOE_indices = []
    
    for idx in OE_indices:
        if all_decisions[idx] == 'escalate':
            BOE_indices.append(idx)
        else:
            MOE_indices.append(idx)
    
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
        'method': 'Epistemic only (v1)',
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
        'escalation_rate': escalation_rate
    }


def method_c_epistemic_aleatoric(data: List[Dict], threshold: float = 0.8,
                                  oe_p_threshold: float = 0.8) -> Dict:
    """
    Method C (v1.5): C = p × (1 - Ue - Ua)
    
    OE = fel + p ≥ oe_p_threshold (SAMMA för alla metoder)
    """
    total = len(data)
    correct = 0
    wrong = 0
    
    # Compute C for all
    all_confidences = []
    all_decisions = []
    all_correct = []
    
    for d in data:
        # C = p × (1 - Ue - Ua)
        Ue = float(np.var(d['mc_predictions']))
        Ua = 2 * d['prediction'] * (1 - d['prediction'])  # Current heuristic
        C = max(0.0, d['prediction'] * (1 - Ue - Ua))
        
        decision = 'auto' if C >= threshold else 'escalate'
        
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        
        all_confidences.append(C)
        all_decisions.append(decision)
        all_correct.append(is_correct)
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
    
    # OE = fel + p ≥ oe_p_threshold (UNIFORM definition)
    OE_indices = []
    for i, d in enumerate(data):
        predicted_class = 1 if d['prediction'] >= 0.5 else 0
        is_correct = predicted_class == d['ground_truth']
        if not is_correct and d['prediction'] >= oe_p_threshold:
            OE_indices.append(i)
    
    # BOE/MOE
    BOE_indices = []
    MOE_indices = []
    
    for idx in OE_indices:
        if all_decisions[idx] == 'escalate':
            BOE_indices.append(idx)
        else:
            MOE_indices.append(idx)
    
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
        'method': 'Epistemic + Aleatoric (v1.5)',
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
        'escalation_rate': escalation_rate
    }


if __name__ == '__main__':
    print("=" * 80)
    print("CognOS — Critical Comparison: Does it beat simple heuristics?")
    print("=" * 80)
    
    # Generate data
    print("\n📊 Genererar 100 syntetiska datapunkter (seed=42)...")
    data = generate_synthetic_data(100)
    
    scenarios = {}
    for d in data:
        s = d['scenario']
        scenarios[s] = scenarios.get(s, 0) + 1
    
    print(f"\nScenario-distribution:")
    for s, count in scenarios.items():
        print(f"  {s}: {count}")
    
    # Test all three methods
    threshold = 0.8
    oe_p_threshold = 0.8  # UNIFORM OE definition for fair comparison
    
    print(f"\n" + "=" * 80)
    print(f"Method Comparison (threshold τ={threshold})")
    print(f"OE definition (UNIFORM): fel + p ≥ {oe_p_threshold} for ALL methods")
    print("=" * 80)
    
    results = []
    
    # Method A: Simple p-threshold
    print("\n🅰️  Method A: Simple p-threshold (baseline)")
    print("-" * 80)
    result_a = method_a_simple_threshold(data, p_threshold=oe_p_threshold, confidence_threshold=threshold)
    results.append(result_a)
    
    print(f"  Decision rule: escalate if p < {threshold}")
    print(f"  OE: {result_a['OE']}")
    print(f"  BOE: {result_a['BOE']}, MOE: {result_a['MOE']}")
    print(f"  Safety Gain: {result_a['safety_gain']:.1f}%")
    print(f"  Escalation rate: {result_a['escalation_rate']*100:.1f}%")
    print(f"  Auto accuracy: {result_a['auto_accuracy']:.1%}")
    
    # Method B: v1 (epistemic only)
    print("\n🅱️  Method B: Epistemic only (v1)")
    print("-" * 80)
    result_b = method_b_epistemic_only(data, threshold=threshold, oe_p_threshold=oe_p_threshold)
    results.append(result_b)
    
    print(f"  Formula: C = p × (1 - Ue)")
    print(f"  Decision rule: escalate if C < {threshold}")
    print(f"  OE: {result_b['OE']}")
    print(f"  BOE: {result_b['BOE']}, MOE: {result_b['MOE']}")
    print(f"  Safety Gain: {result_b['safety_gain']:.1f}%")
    print(f"  Escalation rate: {result_b['escalation_rate']*100:.1f}%")
    print(f"  Auto accuracy: {result_b['auto_accuracy']:.1%}")
    
    # Method C: v1.5 (epistemic + aleatoric)
    print("\n🅾️  Method C: Epistemic + Aleatoric (v1.5)")
    print("-" * 80)
    result_c = method_c_epistemic_aleatoric(data, threshold=threshold, oe_p_threshold=oe_p_threshold)
    results.append(result_c)
    
    print(f"  Formula: C = p × (1 - Ue - Ua), Ua = 2×p×(1-p)")
    print(f"  Decision rule: escalate if C < {threshold}")
    print(f"  OE: {result_c['OE']}")
    print(f"  BOE: {result_c['BOE']}, MOE: {result_c['MOE']}")
    print(f"  Safety Gain: {result_c['safety_gain']:.1f}%")
    print(f"  Escalation rate: {result_c['escalation_rate']*100:.1f}%")
    print(f"  Auto accuracy: {result_c['auto_accuracy']:.1%}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 Summary Comparison")
    print("=" * 80)
    
    print(f"\n{'Method':<30} {'OE':<8} {'Safety Gain':<15} {'Escalation':<15} {'Auto Acc'}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<30} {r['OE']:<8} {r['safety_gain']:>6.1f}%         "
              f"{r['escalation_rate']*100:>6.1f}%         {r['auto_accuracy']:.1%}")
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("🏆 Winner Analysis")
    print("=" * 80)
    
    # Compare at equal escalation rates (approximately)
    print(f"\nAt similar escalation rates (~{results[0]['escalation_rate']*100:.0f}%):")
    
    best_safety = max(r['safety_gain'] for r in results)
    winners = [r for r in results if r['safety_gain'] == best_safety]
    
    if len(winners) == 1:
        winner = winners[0]
        print(f"\n✅ WINNER: {winner['method']}")
        print(f"   Safety Gain: {winner['safety_gain']:.1f}%")
        print(f"   Escalation: {winner['escalation_rate']*100:.1f}%")
        print(f"   Auto Accuracy: {winner['auto_accuracy']:.1%}")
        
        if winner['method'] == 'Simple p-threshold':
            print(f"\n⚠️  CognOS tillför INGET värde över baseline!")
            print(f"   Ua-heuristiken är redundant.")
        else:
            print(f"\n🎯 CognOS ger {winner['safety_gain'] - results[0]['safety_gain']:.1f}% "
                  f"bättre safety gain än baseline!")
    else:
        print(f"\n🤝 TIE: Flera metoder är lika bra ({best_safety:.1f}% safety gain)")
        for w in winners:
            print(f"   - {w['method']}")
    
    # Decision
    print("\n" + "=" * 80)
    print("🎯 GO/NO-GO Decision")
    print("=" * 80)
    
    if best_safety >= 30:
        winner = [r for r in results if r['safety_gain'] == best_safety][0]
        print(f"\n✅ GO: {winner['method']} achieves {best_safety:.1f}% Safety Gain")
        
        if winner['method'] != 'Simple p-threshold':
            print(f"   CognOS confidence formula tillför mätbart värde.")
        else:
            print(f"   ⚠️  Men baseline är lika bra — behöver bättre data eller formel.")
    else:
        print(f"\n⚠️  NO-GO: Bästa metoden ger bara {best_safety:.1f}% Safety Gain (< 30%)")
        print(f"   Problem: antingen datasetet eller formlerna behöver revideras.")
    
    print("\n" + "=" * 80)
