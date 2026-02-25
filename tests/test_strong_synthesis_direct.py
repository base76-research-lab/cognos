#!/usr/bin/env python3
"""
test_strong_synthesis_direct.py — Direct testing of strong synthesis depth.

This test demonstrates the four-phase pipeline:
1. Assumptions + Geometry extraction
2. Concrete integration strategy generation
3. Dynamic meta-alternatives
4. Epistemic gain metrics

Each phase is tested separately to show the advancement beyond
weak heuristic synthesis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.strong_synthesis import (
    extract_assumptions_and_geometry,
    generate_integration_strategy,
    generate_meta_alternatives,
    compute_epistemic_gain,
    synthesize_strong,
)


def mock_llm_fn(system: str, prompt: str) -> str:
    """
    Mock LLM that returns realistically structured responses.
    """
    prompt_lower = prompt.lower()
    
    if "next meta-alternative" in prompt_lower or "next research question" in prompt_lower or "next" in prompt_lower:
        return """{
            "meta_question": "What is the optimal sequence for establishing falsifiability: theory first or empirical validation first?",
            "meta_alternatives": [
                {
                    "label": "Theoretical Rigor First",
                    "action": "Spend 2 weeks formalizing disconfirmation conditions mathematically before any empirical work",
                    "relevant_when": "If credibility in academic community is critical (publication requirements, peer review)",
                    "expected_clarity_gain": 0.7,
                    "effort": "medium"
                },
                {
                    "label": "Empirical Validation First",
                    "action": "Run 10×3×2 pilot immediately to gather data on measurement reliability, inform theory refinement",
                    "relevant_when": "If practical evidence is more compelling than theory; time-sensitive deployment",
                    "expected_clarity_gain": 0.8,
                    "effort": "medium"
                },
                {
                    "label": "Parallel Theory + Empirics",
                    "action": "Theory and validation teams work simultaneously; sync weekly on shared metrics",
                    "relevant_when": "If resources available (team > 3 people) and time is critical",
                    "expected_clarity_gain": 0.9,
                    "effort": "high"
                }
            ],
            "recommended_next_step": "Parallel Theory + Empirics — maximizes clarity and moves toward publication-ready status fastest"
        }"""
    
    elif "assumptions" in prompt_lower or "dimension" in prompt_lower:
        return """{
            "majority_assumption": "The hypothesis is falsifiable because specific, measurable criteria exist for testing each uncertainty component",
            "minority_assumption": "Falsifiability requires not only criteria but operational measurement protocols that are practically implementable",
            "divergence_axes": [
                {
                    "dimension": "Epistemological vs. Practical Threshold",
                    "majority_position": 0.85,
                    "minority_position": 0.25,
                    "interpretation": "Majority emphasizes theoretical clarity; Minority emphasizes empirical operationalization"
                },
                {
                    "dimension": "Complexity of Implementation",
                    "majority_position": 0.3,
                    "minority_position": 0.75,
                    "interpretation": "Majority: Simple criteria sufficient. Minority: Measurement is complex, requires detailed specs"
                }
            ],
            "common_ground": "Both perspectives agree the hypothesis must be empirically testable at some level",
            "core_tension": "Is theoretical falsifiability (Popperian) sufficient, or must we have practical operational measures?"
        }"""
    
    elif "integration" in prompt_lower or "strategy" in prompt_lower:
        return """{
            "primary_strategy": "Decompose falsifiability into two levels: theoretical criteria + operational validation protocol",
            "strategy_details": [
                {
                    "step": 1,
                    "action": "For each U type (U_model, U_prompt, U_problem), write: 'If [observation], then hypothesis is false'",
                    "expected_outcome": "Clear disconfirmation conditions for all three components",
                    "validation_criterion": "Two independent domain experts both find conditions sufficient for disconfirmation"
                },
                {
                    "step": 2,
                    "action": "Define measurement protocol: what instrument/method measures each U value? Resolution: 0.1 on [0,1] scale.",
                    "expected_outcome": "Full operational specification ready for empirical testing",
                    "validation_criterion": "Protocol produces reliable U values (test-retest correlation > 0.8)"
                },
                {
                    "step": 3,
                    "action": "Run pilot validation: 3×2 matrix (3 uncertainty types, 2 confidence levels) on 10 sample decisions",
                    "expected_outcome": "Empirical evidence that the three U types are independently measurable",
                    "validation_criterion": "Correlation between U_model and U_prompt < 0.5; similar for other pairs"
                }
            ],
            "remaining_uncertainty": "Whether empirical independence matches theoretical independence (requires field use)",
            "resource_requirement": "Theory work (1 week) + empirical validation (2-3 weeks) + data analysis (1 week)",
            "estimated_effort": "medium"
        }"""
    
    else:
        return """{
            "is_well_framed": true,
            "problem_type": "ok",
            "reframed_question": null,
            "reason": "Question is well-structured with clear alternatives"
        }"""


def test_phase_1_assumptions():
    """PHASE 1: Test assumption extraction + geometric mapping."""
    print("\n" + "=" * 90)
    print("PHASE 1: ASSUMPTION EXTRACTION + GEOMETRIC MAPPING")
    print("=" * 90)
    print("\nInput: Voting pattern (3 votes for B: 'partial', 2 votes for C: 'strong')")
    print("       on falsifiability question\n")
    
    result = extract_assumptions_and_geometry(
        question="How falsifiable is HYPOTHESIS_V02?",
        alternatives=[
            "A: Weakly falsifiable",
            "B: Partially falsifiable",
            "C: Strongly falsifiable"
        ],
        vote_distribution={"B": 3, "C": 2},
        context="Hypothesis about separable uncertainty types",
        llm_fn=mock_llm_fn,
    )
    
    print(f"✓ Majority Assumption:\n  {result['majority_assumption']}\n")
    print(f"✓ Minority Assumption:\n  {result['minority_assumption']}\n")
    print(f"✓ Core Tension:\n  {result['core_tension']}\n")
    
    if result['divergence_axes']:
        print(f"✓ Geometric Dimensions ({len(result['divergence_axes'])} axes):")
        for i, ax in enumerate(result['divergence_axes'], 1):
            print(f"  [{i}] {ax['dimension']}")
            print(f"      Majority: {ax['majority_position']:+.1f} | Minority: {ax['minority_position']:+.1f}")
            print(f"      → {ax['interpretation']}")
    
    print(f"\n✓ Common Ground:\n  {result['common_ground']}\n")
    
    return result


def test_phase_2_integration(phase1_result):
    """PHASE 2: Test integration strategy generation."""
    print("\n" + "=" * 90)
    print("PHASE 2: CONCRETE INTEGRATION STRATEGY")
    print("=" * 90)
    print("\nInput: Output from Phase 1 (assumptions + geometry)\n")
    
    result = generate_integration_strategy(
        question="How falsifiable is HYPOTHESIS_V02?",
        majority_assumption=phase1_result['majority_assumption'],
        minority_assumption=phase1_result['minority_assumption'],
        divergence_axes=phase1_result['divergence_axes'],
        common_ground=phase1_result['common_ground'],
        core_tension=phase1_result['core_tension'],
        llm_fn=mock_llm_fn,
    )
    
    print(f"✓ Primary Strategy:\n  {result['primary_strategy']}\n")
    
    if result['strategy_details']:
        print(f"✓ Concrete Action Steps ({len(result['strategy_details'])} steps):")
        for step in result['strategy_details']:
            print(f"\n  Step {step['step']}: {step['action'][:70]}...")
            print(f"    Expected: {step['expected_outcome'][:60]}...")
            print(f"    Validate: {step['validation_criterion'][:60]}...")
    
    print(f"\n✓ Resource Requirement: {result['resource_requirement']}")
    print(f"✓ Estimated Effort:    {result['estimated_effort'].upper()}")
    print(f"✓ Remaining Uncertainty:\n  {result['remaining_uncertainty']}\n")
    
    return result


def test_phase_3_meta_alternatives(phase2_result):
    """PHASE 3: Test dynamic meta-alternative generation."""
    print("\n" + "=" * 90)
    print("PHASE 3: DYNAMIC META-ALTERNATIVES (cost-weighted)")
    print("=" * 90)
    print("\nInput: Primary integration strategy from Phase 2\n")
    
    result = generate_meta_alternatives(
        question="How falsifiable is HYPOTHESIS_V02?",
        majority_assumption="Theoretical criteria sufficient",
        minority_assumption="Operational measures required",
        integration_strategy=phase2_result['primary_strategy'],
        confidence=0.65,
        llm_fn=mock_llm_fn,
    )
    
    print(f"✓ Meta-Question:\n  {result['meta_question']}\n")
    
    if result['meta_alternatives']:
        print(f"✓ Next Alternatives ({len(result['meta_alternatives'])} options):")
        for alt in result['meta_alternatives']:
            print(f"\n  • {alt['label']}")
            print(f"    Action: {alt['action'][:70]}...")
            print(f"    Relevant when: {alt['relevant_when'][:60]}...")
            print(f"    Clarity gain: {alt['expected_clarity_gain']:.1%} | Effort: {alt['effort']}")
    
    print(f"\n✓ Recommended Next:\n  {result['recommended_next_step']}\n")
    
    return result


def test_phase_4_epistemic_gain():
    """PHASE 4: Test epistemic gain metrics."""
    print("\n" + "=" * 90)
    print("PHASE 4: EPISTEMIC GAIN METRICS")
    print("=" * 90)
    
    clarity_before = "How falsifiable is HYPOTHESIS_V02?"
    clarity_after = """Decompose falsifiability into:
1. Theoretical criteria: Clear disconfirmation conditions for each U type
2. Operational measures: Measurement protocol with explicit resolution
3. Empirical validation: Pilot test independence of U_model, U_prompt, U_problem"""
    
    result = compute_epistemic_gain(
        confidence_before=0.65,
        confidence_after=0.82,
        clarity_before=clarity_before,
        clarity_after=clarity_after,
        entropy_reduction=0.15,
    )
    
    print(f"\n✓ Confidence Gain:     +{result['confidence_gain']:.3f}")
    print(f"✓ Clarity Improvement: {result['clarity_improvement']:.1%}")
    print(f"✓ Entropy Reduction:   {result['entropy_reduction']:.3f}")
    print(f"✓ Overall Epistemic Gain: {result['overall_epistemic_gain']:.3f}")
    print(f"\n✓ Interpretation:\n  {result['interpretation']}\n")
    
    return result


def test_full_pipeline():
    """Test complete synthesize_strong pipeline."""
    print("\n" + "=" * 90)
    print("FULL PIPELINE: synthesize_strong()")
    print("=" * 90)
    print("\nRunning all four phases in single call...\n")
    
    result = synthesize_strong(
        question="How falsifiable is HYPOTHESIS_V02?",
        alternatives=[
            "A: Weakly falsifiable",
            "B: Partially falsifiable",
            "C: Strongly falsifiable"
        ],
        vote_distribution={"B": 3, "C": 2},
        confidence=0.65,
        context="Hypothesis proposes three separable uncertainty types",
        llm_fn=mock_llm_fn,
    )
    
    print(f"Iteration Depth: {result['iteration_depth']}")
    print(f"\n✓ Analysis:")
    print(f"  - Core tension: {result['analysis']['core_tension'][:70]}...")
    print(f"  - Geometry: {len(result['analysis']['divergence_axes'])} dimensions")
    print(f"\n✓ Integration:")
    print(f"  - Primary: {result['integration']['primary_strategy'][:70]}...")
    print(f"  - Steps: {len(result['integration']['strategy_details'])}")
    print(f"  - Effort: {result['integration']['estimated_effort']}")
    print(f"\n✓ Meta-Loop:")
    print(f"  - Question: {result['meta_alternatives']['meta_question'][:70]}...")
    print(f"  - Alternatives: {len(result['meta_alternatives']['alternatives'])}")
    print(f"  - Recommended: {result['meta_alternatives']['recommended_next_step'][:50]}...")
    print(f"\n✓ Epistemic Gain:")
    print(f"  - Overall: {result['epistemic_gain']['overall_epistemic_gain']:.3f}")
    print(f"  - Status: {result['epistemic_gain']['interpretation']}")
    
    return result


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 30)
    print("COGNOS STRONG SYNTHESIS — COMPLETE DEMONSTRATION")
    print("🚀 " * 30)
    
    # Phase 1: Assumptions
    phase1 = test_phase_1_assumptions()
    
    # Phase 2: Integration
    phase2 = test_phase_2_integration(phase1)
    
    # Phase 3: Meta-alternatives
    phase3 = test_phase_3_meta_alternatives(phase2)
    
    # Phase 4: Epistemic gain
    phase4 = test_phase_4_epistemic_gain()
    
    # Full pipeline
    full = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 90)
    print("COMPARISON: WEAK vs STRONG SYNTHESIS")
    print("=" * 90)
    
    print("""
WEAK SYNTHESIS (Old):
  Input: vote_distribution, confidence
  Output: Simple narrative description (majorty: X, minority: Y)
  Integration: Generic text ("both valid", "need more info")
  Meta: Fixed alternatives (3 perspectives)
  Gain: No measurement

STRONG SYNTHESIS (New):
  Input: vote_distribution, confidence
  Output: Geometric mapping with explicit dimensions
  Integration: Concrete action steps with validation criteria
  Meta: Dynamic generation based on error cost + resource constraints
  Gain: Measured (entropy reduction, confidence improvement, clarity)
  
RESULT:
  ✓ Output is clear (before) + actionable (after)
  ✓ Integration strategy moves from narrative to concrete
  ✓ Meta-alternatives cost-weighted (high value alternatives first)
  ✓ Progress quantifiable through epistemic gain metrics
""")
    
    print("=" * 90)
    print("✅ STRONG SYNTHESIS DEMONSTRATION COMPLETE")
    print("=" * 90 + "\n")


if __name__ == '__main__':
    main()
