#!/usr/bin/env python3
"""
test_strong_synthesis.py — Test orchestrator with strong synthesis.

This test suite runs the full orchestrator against three bedömning
questions, evaluating:

1. Output clarity vs. baseline LLM
2. Integration strategy concreteness
3. Epistemic gain metrics
4. Meta-alternative quality

This is the validation of "conflict → assumptions → geometry → synthesis"
"""

import sys
from pathlib import Path

# Add parent and grandparent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos import CognOSOrchestrator


def mock_llm_fn(system: str, prompt: str) -> str:
    """
    Mock LLM that returns structured responses without API calls.
    
    For testing purposes, returns reasonably realistic answers.
    """
    # Simple fallback responses for testing
    if "assumptions" in prompt.lower() or "divergence" in prompt.lower():
        return """{
            "majority_assumption": "The hypothesis is falsifiable because clear criteria exist for disconfirmation",
            "minority_assumption": "Falsifiability requires not just criteria but practical operationalization",
            "divergence_axes": [
                {
                    "dimension": "Epistemological threshold",
                    "majority_position": 0.8,
                    "minority_position": 0.3,
                    "interpretation": "Majority: Theoretical criteria sufficient. Minority: Must have testable operations."
                }
            ],
            "common_ground": "Both agree the hypothesis should be testable in principle",
            "core_tension": "Definition of what counts as falsifiable (theoretical vs. practical)"
        }"""
    elif "integration" in prompt.lower() or "strategy" in prompt.lower():
        return """{
            "primary_strategy": "Define falsifiability operationally with specific threshold metrics",
            "strategy_details": [
                {
                    "step": 1,
                    "action": "Specify what observation would falsify each component of HYPOTHESIS_V02",
                    "expected_outcome": "Clear disconfirmation conditions for U_model, U_prompt, U_problem",
                    "validation_criterion": "Can hypothetical data be described that would reject each?"
                },
                {
                    "step": 2,
                    "action": "Define measurement protocol for U values (0-1 scale, based on what?)",
                    "expected_outcome": "Operational definitions ready for empirical testing",
                    "validation_criterion": "Two independent observers agree on U measurement within 0.1"
                }
            ],
            "remaining_uncertainty": "Whether the three U types are truly independent empirically",
            "resource_requirement": "Theoretical clarification + empirical validation study",
            "estimated_effort": "medium"
        }"""
    elif "meta" in prompt.lower() or "next" in prompt.lower():
        return """{
            "meta_question": "How should we operationalize and test the independence of U_model, U_prompt, U_problem?",
            "meta_alternatives": [
                {
                    "label": "Prioritize theoretical independence proof",
                    "action": "Show mathematically why U types must be independent",
                    "relevant_when": "If theoretical foundation is weak, this establishes credibility",
                    "expected_clarity_gain": 0.7,
                    "effort": "high"
                },
                {
                    "label": "Run empirical validation on real data",
                    "action": "10x3x3 experiment measuring all three U values simultaneously",
                    "relevant_when": "If theory seems sound, empirical evidence is strongest argument",
                    "expected_clarity_gain": 0.8,
                    "effort": "medium"
                }
            ],
            "recommended_next_step": "Run empirical validation; theory already provides sufficient foundation"
        }"""
    elif "frame" in prompt.lower() or "well-posed" in prompt.lower():
        return """{
            "is_well_framed": true,
            "problem_type": "ok",
            "reframed_question": null,
            "reason": "Question is well-defined with testable alternatives"
        }"""
    else:
        # Default choice response
        return """CHOICE: B
CONFIDENCE: 0.7
RATIONALE: Current formulation clear but operationalization needed"""


def test_orchestrator_with_strong_synthesis():
    """
    Test full orchestration pipeline with strong synthesis.
    
    Tests on three bedömning questions with clarity + gain metrics.
    """
    
    print("=" * 90)
    print("COGNOS STRONG SYNTHESIS TEST")
    print("=" * 90)
    print("\nTesting full epistemic orchestration with concrete integration strategies.\n")
    
    # Create orchestrator with mock LLM
    orchestrator = CognOSOrchestrator(
        llm_fn=mock_llm_fn,
        verbose=True,
        max_depth=3,
    )
    
    context = """CognOS — Epistemic Integrity Layer for AI Systems.

Formula: C = p × (1 - Ue - Ua)

Three separable uncertainty types:
1. U_model: Internal model/sampling uncertainty (confidence spread)
2. U_prompt: Format/representation uncertainty (alternative framings)
3. U_problem: Ill-posedness of question itself (definition gaps)

HYPOTHESIS_V02 proposes these are measurable and can be optimized independently.
"""
    
    questions = [
        {
            "title": "Falsifiability Assessment",
            "question": "How falsifiable is HYPOTHESIS_V02 in its current form?",
            "alternatives": [
                "A: Weakly falsifiable — theoretical criteria exist but operationalization unclear",
                "B: Partially falsifiable — some components testable, others require clarification",
                "C: Strongly falsifiable — clear disconfirmation conditions for all parts"
            ]
        },
        {
            "title": "Critical Weakness",
            "question": "What is HYPOTHESIS_V02's most critical weakness right now?",
            "alternatives": [
                "A: Empirical foundation too narrow — only tested on AI confidence decisions",
                "B: U_prompt and U_model overlap too much — not sufficiently independent",
                "C: Integration strategy underspecified — how to act on uncertainty values"
            ]
        },
        {
            "title": "Validation Strategy",
            "question": "Should we run the 10×3×3 validation experiment now?",
            "alternatives": [
                "A: Yes — definitions are sufficient, start collecting data",
                "B: Yes, but sharpen definitions first — 2 week preparation",
                "C: No — need more groundwork on operational metrics"
            ]
        }
    ]
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'#' * 90}")
        print(f"# QUESTION {i}/3: {q['title']}")
        print(f"{'#' * 90}\n")
        
        result = orchestrator.orchestrate(
            question=q['question'],
            alternatives=q['alternatives'],
            context=context,
            n_samples=3,
        )
        
        results.append({
            'title': q['title'],
            'question': q['question'],
            'result': result,
        })
        
        # Display summary
        print(f"\n{'=' * 90}")
        print(f"SUMMARY: {q['title']}")
        print(f"{'=' * 90}")
        print(f"Decision:              {result['decision'].upper()}")
        print(f"Confidence:            {result['confidence']:.3f} / 1.0")
        print(f"Iterations:            {result['iterations']}")
        print(f"Converged:             {'Yes ✓' if result['converged'] else 'No (more depth available)'}")
        
        if result.get('final_answer'):
            print(f"Final answer:          {result['final_answer']}")
        
        # Display divergence analysis if available
        if result['layers'] and 'divergence' in result['layers'][-1]:
            div = result['layers'][-1]['divergence']
            print(f"\n🧠 DIVERGENCE ANALYSIS (if synthesized):")
            
            if isinstance(div, dict) and 'analysis' in div:
                print(f"  Majority assumption: {div['analysis']['majority_assumption'][:70]}")
                print(f"  Minority assumption: {div['analysis']['minority_assumption'][:70]}")
                print(f"  Core tension:        {div['analysis']['core_tension'][:70]}")
                
                # Display integration strategy
                if 'integration' in div:
                    print(f"\n💪 INTEGRATION STRATEGY (concrete):")
                    print(f"  Primary: {div['integration']['primary_strategy'][:70]}")
                    if div['integration']['strategy_details']:
                        print(f"  Steps:   {len(div['integration']['strategy_details'])} actionable steps")
                    print(f"  Effort:  {div['integration']['estimated_effort']}")
                
                # Display epistemic gain
                if 'epistemic_gain' in div:
                    gain = div['epistemic_gain']
                    print(f"\n📈 EPISTEMIC GAIN:")
                    print(f"  Overall gain: {gain['overall_epistemic_gain']:.3f}")
                    print(f"  Interpretation: {gain['interpretation']}")
    
    # Final comparison
    print(f"\n\n{'=' * 90}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'=' * 90}\n")
    
    print("Question | Decision | Confidence | Iterations | Concreteness")
    print("---------|----------|------------|------------|-------------")
    
    for res in results:
        r = res['result']
        title_short = res['title'][:20]
        
        # Assess concreteness of output
        concreteness = "Low (baseline)"
        if r['layers'] and 'divergence' in r['layers'][-1]:
            div = r['layers'][-1]['divergence']
            if isinstance(div, dict) and 'integration' in div:
                if div['integration']['strategy_details']:
                    concreteness = "High (concrete steps ✓)"
                elif div['integration']['primary_strategy']:
                    concreteness = "Medium (strategy defined)"
        
        print(f"{title_short:<20}| {r['decision']:<8} | {r['confidence']:>10.3f} | {r['iterations']:>10} | {concreteness}")
    
    print(f"\n{'=' * 90}")
    print("✅ TEST COMPLETE")
    print(f"{'=' * 90}\n")
    
    # Validate that strong synthesis was used
    print("Key Metrics:")
    print(f"  • Strong synthesis available: {'✓ YES' if any(len(r['result']['layers']) > 0 for r in results) else '✗ NO'}")
    print(f"  • Average iterations per question: {sum(r['result']['iterations'] for r in results) / len(results):.1f}")
    print(f"  • Questions with concrete strategies: {sum(1 for r in results if r['result']['layers'] and 'divergence' in r['result']['layers'][-1])}/3")
    print()


if __name__ == '__main__':
    test_orchestrator_with_strong_synthesis()
