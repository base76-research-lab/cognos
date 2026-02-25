#!/usr/bin/env python3
"""
test_reasoning_loop.py — Test the Meta-Recursive Reasoning Loop

Tests on concrete question types:
1. Falsifiability (science)
2. Empirical strength (evidence)
3. Go/No-Go (decision)

Plus:
4. Policy questions
5. Medical decisions
6. AI governance
7. Scientific hypotheses
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.reasoning_loop import ReasoningLoop


def get_llm_fn():
    """Get LLM function (Groq or mock for testing)."""
    try:
        from groq import Groq
        client = Groq()
        
        def ask_groq(system: str, prompt: str, temperature: float = 0.7) -> Optional[str]:
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️  Groq error: {e}", file=sys.stderr)
                return None
        
        return ask_groq
    except (ImportError, Exception):
        # Fallback: Mock LLM for testing the LOGIC
        def mock_llm(system: str, prompt: str, temperature: float = 0.7) -> Optional[str]:
            """Mock LLM for testing reasoning loop logic."""
            
            # Detect question type and return appropriate structured response
            if "CHOICE:" in prompt:
                # Voting question — return choice
                if "falsifiable" in prompt.lower():
                    return "CHOICE: C\nCONFIDENCE: 0.75\nRATIONALE: Clear criteria exist"
                elif "evidence" in prompt.lower():
                    return "CHOICE: B\nCONFIDENCE: 0.68\nRATIONALE: Multiple studies needed"
                else:
                    return "CHOICE: B\nCONFIDENCE: 0.72\nRATIONALE: Balanced approach"
            
            elif "divergence_source" in prompt.lower() or "underliggande antagandet" in prompt.lower():
                # Divergence analysis
                return """{
  "majority_assumption": "Rigorous thresholds are necessary for falsifiability",
  "minority_assumption": "Practical applicability matters more than theoretical purity",
  "divergence_source": "Different priorities: epistemic rigor vs practical utility",
  "divergence_type": "normative",
  "divergence_axes": [
    {"dimension": "rigor_vs_utility", "majority_position": 0.9, "minority_position": 0.4, "interpretation": "Rigor to utility tradeoff"}
  ],
  "integration_strategy": "Combine both: define rigor criteria AND practical thresholds",
  "integration_mode": "tradeoff",
  "meta_question": "How much rigor is necessary for practical deployment?",
  "meta_alternatives": ["Define minimum rigor for deployment", "Test on real data", "Establish review process"],
  "is_resolvable": true
}"""
            
            elif "well_framed" in prompt.lower() or "är denna fråga" in prompt.lower():
                # Frame check
                return """{
  "is_well_framed": true,
  "problem_type": "ok",
  "reframed_question": null,
  "specific_issues": [],
  "missing_specifications": [],
  "recommendation": "Question is well-posed"
}"""
            
            return None
        
        print("⚠️  Note: Using mock LLM for testing (Groq not available)")
        return mock_llm


def demo_falsifiability():
    """Test: Is a hypothesis falsifiable?"""
    
    print("\n" + "="*80)
    print("BENCHMARK 1: FALSIFIABILITY")
    print("="*80)
    
    llm_fn = get_llm_fn()
    if not llm_fn:
        print("❌ No LLM available")
        return None
    
    loop = ReasoningLoop(
        llm_fn=llm_fn,
        max_depth=3,
        convergence_threshold=0.05,
        verbose=True,
    )
    
    result = loop.run(
        question="Is HYPOTHESIS_V02 falsifiable in its current form?",
        alternatives=[
            "Weakly falsifiable — lacks clear thresholds for rejection",
            "Partially falsifiable but requires stricter operational criteria",
            "Strongly falsifiable with clear and testable predictions",
        ],
        context="""HYPOTHESIS_V02: Confidence = p × (1 - Ue - Ua)

Three uncertainty types:
- U_model: internal model epistemic uncertainty
- U_prompt: format-induced uncertainty
- U_problem: ill-posedness of the question itself

The hypothesis proposes these are separable and independently testable.""",
        n_samples=5,  # >= 4 required by compute_confidence
    )
    
    print("\n📊 ANALYSIS:")
    for i, it in enumerate(result['iterations'], 1):
        print(f"\n  Iteration {i} (L{it.depth}):")
        print(f"    Question: {it.question[:70]}...")
        print(f"    Confidence: {it.confidence:.3f}")
        print(f"    Decision: {it.decision}")
        if it.majority_assumption:
            print(f"    Majority: {it.majority_assumption[:60]}...")
        if it.minority_assumption:
            print(f"    Minority: {it.minority_assumption[:60]}...")
    
    return result


def demo_empirical_strength():
    """Test: What evidence strength justifies action?"""
    
    print("\n" + "="*80)
    print("BENCHMARK 2: EMPIRICAL STRENGTH")
    print("="*80)
    
    llm_fn = get_llm_fn()
    if not llm_fn:
        print("❌ No LLM available")
        return None
    
    loop = ReasoningLoop(
        llm_fn=llm_fn,
        max_depth=3,
        convergence_threshold=0.05,
        verbose=True,
    )
    
    result = loop.run(
        question="What level of empirical evidence should trigger a policy change?",
        alternatives=[
            "Single well-designed study with positive results",
            "Multiple studies with consistent effects across contexts",
            "Meta-analysis with high effect sizes and low heterogeneity",
        ],
        context="""Context: We're deciding whether to recommend a new medical treatment.

Trade-off: 
- Fast adoption helps patients but risks harm if evidence is weak
- Slow adoption ensures safety but delays benefit to patients

Question: What's the minimum evidence needed?""",
        n_samples=5,
    )
    
    print("\n📊 ANALYSIS:")
    print(f"Final confidence: {result['confidence']:.3f}")
    print(f"Meta-depth reached: {result['meta_depth']}")
    print(f"Converged: {result['converged']}")
    print(f"Reason: {result['convergence_reason']}")
    
    return result


def demo_go_no_go():
    """Test: Should we proceed with this initiative?"""
    
    print("\n" + "="*80)
    print("BENCHMARK 3: GO/NO-GO DECISION")
    print("="*80)
    
    llm_fn = get_llm_fn()
    if not llm_fn:
        print("❌ No LLM available")
        return None
    
    loop = ReasoningLoop(
        llm_fn=llm_fn,
        max_depth=2,
        convergence_threshold=0.05,
        verbose=True,
    )
    
    result = loop.run(
        question="Should we proceed with deploying this AI model in a customer-facing system?",
        alternatives=[
            "GO: Deploy immediately, monitor carefully",
            "HOLD: Deploy to limited user group first (beta)",
            "NO-GO: Do not deploy until additional safety testing",
        ],
        context="""Model readiness assessment:
- Accuracy: 94% (above threshold)
- Fairness: Balanced across demographics ✓
- Safety: 2 unresolved edge cases identified
- Regulatory: Awaiting final approval (2-3 weeks)

Cost of delay: $50k/week
Cost of failure: Reputational + legal damage""",
        n_samples=5,
    )
    
    print("\n📊 DECISION ANALYSIS:")
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Final answer: {result['final_answer']}")
    print(f"Meta-depth: {result['meta_depth']}")
    
    if result['iterations']:
        last_it = result['iterations'][-1]
        if last_it.divergence:
            div = last_it.divergence
            print(f"\n🔍 DIVERGENCE ANALYSIS:")
            print(f"  Type: {div.get('divergence_type')}")
            print(f"  Mode: {div.get('integration_mode')}")
            print(f"  Strategy: {div.get('integration_strategy', 'N/A')[:100]}...")
    
    return result


def benchmark_all():
    """Run all benchmarks."""
    
    print("\n" + "#"*80)
    print("# META-RECURSIVE REASONING ENGINE")
    print("# Full Benchmark Suite")
    print("#"*80)
    
    results = {}
    
    try:
        r1 = demo_falsifiability()
        if r1:
            results['falsifiability'] = r1
        
        r2 = demo_empirical_strength()
        if r2:
            results['empirical_strength'] = r2
        
        r3 = demo_go_no_go()
        if r3:
            results['go_no_go'] = r3
        
        # Summary
        if results:
            print("\n" + "="*80)
            print("📊 SUMMARY ACROSS BENCHMARKS")
            print("="*80)
            
            for name, result in results.items():
                if result:
                    print(f"\n{name.upper()}:")
                    print(f"  Decision: {result['decision']}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Meta-depth: {result['meta_depth']}")
                    print(f"  Converged: {result['converged']}")
        
        print("\n" + "="*80)
        print("✅ BENCHMARK SUITE COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    benchmark_all()
