#!/usr/bin/env python3
"""
metrics.py — Research Metrics for CognOS Evaluation

Implements 5 core metrics:
1. Accuracy — correctness of final answer
2. Confidence Calibration Error (CCE) — how well-calibrated confidence is
3. Hallucination Frequency — false confidence on wrong answers
4. Assumption Detection Rate — quality of divergence analysis
5. Convergence Score — stability of reasoning process

Monte Carlo epistemic sampling across N iterations.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Single metric measurement."""
    name: str
    value: float
    unit: str
    interpretation: str


@dataclass
class ExperimentMetrics:
    """Complete metrics for one experiment."""
    accuracy: float
    confidence_calibration_error: float
    hallucination_frequency: float
    assumption_detection_rate: float
    convergence_score: float
    
    # Additional context
    n_iterations: int
    baseline_comparison: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            'accuracy': self.accuracy,
            'confidence_calibration_error': self.confidence_calibration_error,
            'hallucination_frequency': self.hallucination_frequency,
            'assumption_detection_rate': self.assumption_detection_rate,
            'convergence_score': self.convergence_score,
            'n_iterations': self.n_iterations,
            'baseline_comparison': self.baseline_comparison,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
EXPERIMENT METRICS (N={self.n_iterations})
{'='*60}
Accuracy:                      {self.accuracy:.3f}
Confidence Calibration Error:  {self.confidence_calibration_error:.3f}
Hallucination Frequency:       {self.hallucination_frequency:.3f}
Assumption Detection Rate:     {self.assumption_detection_rate:.3f}
Convergence Score:             {self.convergence_score:.3f}
{'='*60}
"""


def compute_accuracy(results: list[dict], ground_truth: list[str]) -> float:
    """
    Metric 1: Accuracy
    
    Measures: correctness of final answer
    Range: [0, 1], higher is better
    
    Args:
        results: List of result dicts with 'final_answer' key
        ground_truth: List of correct answers
    
    Returns:
        Accuracy score
    """
    if not results or not ground_truth or len(results) != len(ground_truth):
        return 0.0
    
    correct = 0
    for result, truth in zip(results, ground_truth):
        answer = result.get('final_answer', '').lower().strip()
        truth = truth.lower().strip()
        
        # Check if answer contains truth or vice versa (fuzzy match)
        if truth in answer or answer in truth:
            correct += 1
    
    return correct / len(results)


def compute_confidence_calibration_error(
    results: list[dict],
    ground_truth: list[str]
) -> float:
    """
    Metric 2: Confidence Calibration Error (CCE)
    
    Measures: |P(correct | C) - C|
    Perfect calibration: a system that says 0.8 confidence should be correct 80% of the time
    
    Range: [0, 1], lower is better (0 = perfect calibration)
    
    Args:
        results: List of result dicts with 'confidence' and 'final_answer'
        ground_truth: List of correct answers
    
    Returns:
        Mean absolute calibration error
    """
    if not results or not ground_truth:
        return 1.0
    
    # Bin into confidence buckets
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    errors = []
    
    for i in range(len(bins) - 1):
        bin_low, bin_high = bins[i], bins[i+1]
        
        # Get results in this confidence bin
        bin_results = []
        bin_ground_truth = []
        
        for result, truth in zip(results, ground_truth):
            conf = result.get('confidence', 0.5)
            if bin_low <= conf < bin_high or (i == len(bins) - 2 and conf == bin_high):
                bin_results.append(result)
                bin_ground_truth.append(truth)
        
        if not bin_results:
            continue
        
        # Compute accuracy in this bin
        bin_accuracy = compute_accuracy(bin_results, bin_ground_truth)
        
        # Expected confidence (bin midpoint)
        expected_conf = (bin_low + bin_high) / 2
        
        # Calibration error for this bin
        error = abs(bin_accuracy - expected_conf)
        errors.append(error)
    
    return np.mean(errors) if errors else 1.0


def compute_hallucination_frequency(
    results: list[dict],
    ground_truth: list[str]
) -> float:
    """
    Metric 3: Hallucination Frequency
    
    Measures: P(high confidence AND wrong)
    High confidence (>0.8) on incorrect answers
    
    Range: [0, 1], lower is better (0 = no hallucinations)
    
    Args:
        results: List of result dicts with 'confidence' and 'final_answer'
        ground_truth: List of correct answers
    
    Returns:
        Proportion of high-confidence errors
    """
    if not results or not ground_truth:
        return 0.0
    
    hallucinations = 0
    high_confidence_threshold = 0.8
    
    for result, truth in zip(results, ground_truth):
        confidence = result.get('confidence', 0.5)
        answer = result.get('final_answer', '').lower().strip()
        truth_text = truth.lower().strip()
        
        is_correct = (truth_text in answer or answer in truth_text)
        is_high_confidence = confidence > high_confidence_threshold
        
        if is_high_confidence and not is_correct:
            hallucinations += 1
    
    return hallucinations / len(results)


def compute_assumption_detection_rate(results: list[dict]) -> float:
    """
    Metric 4: Assumption Detection Rate
    
    Measures: quality of divergence analysis
    - Did system extract assumptions when divergence occurred?
    - Are assumptions substantive (not just "they disagree")?
    
    Range: [0, 1], higher is better
    
    Args:
        results: List of result dicts with 'iterations' containing divergence info
    
    Returns:
        Proportion of cases with good assumption extraction
    """
    if not results:
        return 0.0
    
    good_detections = 0
    
    for result in results:
        iterations = result.get('iterations', [])
        
        # Check if any iteration has substantive assumptions
        has_substantive = False
        for iteration in iterations:
            maj_assumption = iteration.get('majority_assumption', '')
            min_assumption = iteration.get('minority_assumption', '')
            
            # Quality checks:
            # 1. Not empty
            # 2. Longer than 20 chars (not just "they disagree")
            # 3. Contains actual content
            if (maj_assumption and min_assumption and
                len(maj_assumption) > 20 and len(min_assumption) > 20 and
                'antar' in maj_assumption.lower()):  # Swedish: "assumes"
                has_substantive = True
                break
        
        if has_substantive:
            good_detections += 1
    
    return good_detections / len(results)


def compute_convergence_score(results: list[dict]) -> float:
    """
    Metric 5: Convergence Score
    
    Measures: stability of reasoning process
    - Did confidence stabilize?
    - Did system converge to answer?
    - Was convergence reached in reasonable depth?
    
    Range: [0, 1], higher is better
    
    Args:
        results: List of result dicts with 'converged', 'meta_depth', 'confidence'
    
    Returns:
        Mean convergence quality score
    """
    if not results:
        return 0.0
    
    scores = []
    
    for result in results:
        converged = result.get('converged', False)
        meta_depth = result.get('meta_depth', 0)
        confidence = result.get('confidence', 0.0)
        
        # Scoring criteria:
        score = 0.0
        
        # 1. Did it converge? (40%)
        if converged:
            score += 0.4
        
        # 2. Meta-depth efficiency (30%)
        # Prefer depth 1-2, penalize 0 (no reasoning) or >3 (too deep)
        if meta_depth == 0:
            score += 0.0
        elif meta_depth <= 2:
            score += 0.3
        elif meta_depth == 3:
            score += 0.2
        else:
            score += 0.1
        
        # 3. Final confidence (30%)
        # Prefer moderate confidence (0.6-0.9), penalize extremes
        if 0.6 <= confidence <= 0.9:
            score += 0.3
        elif 0.5 <= confidence < 0.6 or 0.9 < confidence <= 1.0:
            score += 0.2
        else:
            score += 0.1
        
        scores.append(score)
    
    return np.mean(scores)


def compute_all_metrics(
    cognos_results: list[dict],
    ground_truth: list[str],
    baseline_results: Optional[list[dict]] = None,
) -> ExperimentMetrics:
    """
    Compute all 5 metrics for an experiment.
    
    Args:
        cognos_results: CognOS outputs (N iterations)
        ground_truth: Correct answers (if known, else empty list)
        baseline_results: Optional baseline LLM outputs for comparison
    
    Returns:
        ExperimentMetrics with all 5 metrics computed
    """
    
    n = len(cognos_results)
    
    # Compute metrics
    accuracy = compute_accuracy(cognos_results, ground_truth) if ground_truth else 0.0
    cce = compute_confidence_calibration_error(cognos_results, ground_truth) if ground_truth else 0.0
    hallucination = compute_hallucination_frequency(cognos_results, ground_truth) if ground_truth else 0.0
    assumption_rate = compute_assumption_detection_rate(cognos_results)
    convergence = compute_convergence_score(cognos_results)
    
    # Optional: baseline comparison
    baseline_comparison = None
    if baseline_results and ground_truth:
        baseline_comparison = {
            'accuracy': compute_accuracy(baseline_results, ground_truth),
            'cce': compute_confidence_calibration_error(baseline_results, ground_truth),
            'hallucination': compute_hallucination_frequency(baseline_results, ground_truth),
            'delta_accuracy': accuracy - compute_accuracy(baseline_results, ground_truth),
            'delta_cce': cce - compute_confidence_calibration_error(baseline_results, ground_truth),
            'delta_hallucination': hallucination - compute_hallucination_frequency(baseline_results, ground_truth),
        }
    
    return ExperimentMetrics(
        accuracy=accuracy,
        confidence_calibration_error=cce,
        hallucination_frequency=hallucination,
        assumption_detection_rate=assumption_rate,
        convergence_score=convergence,
        n_iterations=n,
        baseline_comparison=baseline_comparison,
    )


if __name__ == '__main__':
    # Demo: compute metrics on mock data
    mock_results = [
        {'final_answer': 'yes', 'confidence': 0.85, 'converged': True, 'meta_depth': 1, 'iterations': [{'majority_assumption': 'The system assumes high accuracy is necessary'}]},
        {'final_answer': 'no', 'confidence': 0.65, 'converged': True, 'meta_depth': 2, 'iterations': [{'majority_assumption': 'Risk mitigation is prioritized'}]},
        {'final_answer': 'maybe', 'confidence': 0.50, 'converged': False, 'meta_depth': 3, 'iterations': []},
    ]
    
    ground_truth = ['yes', 'no', 'unknown']
    
    metrics = compute_all_metrics(mock_results, ground_truth)
    print(metrics.summary())
