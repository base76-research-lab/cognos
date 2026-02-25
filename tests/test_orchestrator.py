#!/usr/bin/env python3
"""
test_orchestrator.py — Test CognOS orchestration pipeline.

Run the full epistemic orchestration on three bedömning questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos import CognOSOrchestrator


def mock_llm_provider(system: str, prompt: str, temperature: float = 0.7) -> str:
    """
    Mock LLM provider for testing (simulates votes and divergence).
    
    In production, replace with real Groq/OpenAI/GitHub Models call.
    """
    
    # Simulate realistic responses based on prompt content
    if "CHOICE:" in prompt and "CONFIDENCE:" in prompt:
        # Structured choice voting
        if "falsifiable" in prompt.lower():
            responses = [
                "CHOICE: B\nCONFIDENCE: 0.7\nRATIONALE: Testable but needs precision",
                "CHOICE: B\nCONFIDENCE: 0.6\nRATIONALE: Right direction unclear",
                "CHOICE: C\nCONFIDENCE: 0.8\nRATIONALE: Clear falsification criteria",
            ]
        elif "critical weakness" in prompt.lower():
            responses = [
                "CHOICE: B\nCONFIDENCE: 0.75\nRATIONALE: Definitions need sharpening",
                "CHOICE: C\nCONFIDENCE: 0.6\nRATIONALE: Actually underspecified",
                "CHOICE: B\nCONFIDENCE: 0.65\nRATIONALE: Threshold definitions lacking",
            ]
        elif "validation experiment" in prompt.lower() or "10×3×3" in prompt.lower():
            responses = [
                "CHOICE: B\nCONFIDENCE: 0.72\nRATIONALE: Sharpen first, then run",
                "CHOICE: A\nCONFIDENCE: 0.55\nRATIONALE: Ready to start now",
                "CHOICE: B\nCONFIDENCE: 0.70\nRATIONALE: One more iteration needed",
            ]
        else:
            responses = [
                "CHOICE: A\nCONFIDENCE: 0.7\nRATIONALE: Seems reasonable",
                "CHOICE: B\nCONFIDENCE: 0.5\nRATIONALE: Unclear, depends on context",
            ]
        
        # Rotate through responses
        import random
        return random.choice(responses)
    
    elif "underliggande antaganden" in prompt or "underlying assumption" in prompt.lower():
        # Divergence extraction
        return json.dumps({
            "majority_assumption": "The hypothesis needs stricter measurement criteria",
            "minority_assumption": "Current definitions are sufficient for testing",
            "divergence_source": "Different views on required rigor level",
            "integration_strategy": "Adopt stricter thresholds but recognize existing framework is solid",
            "meta_question": "What measurement rigor is actually needed for publication?",
            "is_resolvable": True
        })
    
    elif "välställd" in prompt or "well-framed" in prompt.lower():
        # Frame check
        return json.dumps({
            "is_well_framed": True,
            "problem_type": "ok",
            "reframed_question": None,
            "reason": "Question is operationalizable"
        })
    
    return "CHOICE: A\nCONFIDENCE: 0.5\nRATIONALE: Default response"


if __name__ == "__main__":
    import json
    
    print("\n" + "="*80)
    print("CognOS ORCHESTRATOR — FULL PIPELINE TEST")
    print("="*80)
    
    # Create orchestrator with mock LLM
    o = CognOSOrchestrator(llm_fn=mock_llm_provider, verbose=True)
    
    context = """CognOS is an epistemic integrity layer for AI.

Formula: C = p × (1 - Ue - Ua)

Uncertainty types:
- U_model: internal epistemic uncertainty (variance with temperature/seed)
- U_prompt: format-induced uncertainty (between narrativ/forced/structured)
- U_problem: question's intrinsic ill-posedness

HYPOTHESIS_V02 proposes these are separable and testable via 10×3×3 experiment."""
    
    questions = [
        {
            "title": "Falsifiability",
            "question": "How falsifiable is HYPOTHESIS_V02 in its current form?",
            "alternatives": [
                "A: Weakly falsifiable",
                "B: Partially falsifiable but requires stricter thresholds",
                "C: Strongly falsifiable with clear criteria"
            ]
        },
        {
            "title": "Critical weakness",
            "question": "What is HYPOTHESIS_V02's most critical weakness right now?",
            "alternatives": [
                "A: Empirical foundation too narrow",
                "B: U_prompt and U_model overlap too much",
                "C: Decision rule is underspecified"
            ]
        },
        {
            "title": "Run validation now?",
            "question": "Should we run the 10×3×3 validation experiment now?",
            "alternatives": [
                "A: Yes, definitions are sufficient",
                "B: Yes, but sharpen definitions iteratively",
                "C: No, need more preliminary work"
            ]
        }
    ]
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n\n{'#'*80}")
        print(f"# BEDÖMNING {i}/3: {q['title'].upper()}")
        print(f"{'#'*80}\n")
        
        result = o.orchestrate(
            question=q["question"],
            alternatives=q["alternatives"],
            context=context,
            n_samples=3,
        )
        
        results.append((q["title"], result))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("ORCHESTRATION RESULTS")
    print(f"{'='*80}\n")
    
    for title, result in results:
        print(f"📋 {title.upper()}")
        print(f"   Decision: {result['decision'].upper()}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Converged: {'✓' if result['converged'] else '✗'}")
        if result['final_answer']:
            print(f"   Answer: {result['final_answer'][:70]}")
        print()
    
    print(f"{'='*80}")
    print("✅ ORCHESTRATOR TEST COMPLETE")
    print(f"{'='*80}\n")
