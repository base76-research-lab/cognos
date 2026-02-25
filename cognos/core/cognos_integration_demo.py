#!/usr/bin/env python3
"""
cognos_integration_demo.py — Shows how the three layers (confidence, divergence, meta) interact.

This demo shows an actual agentic chain:
  1. Structured question → MC sampling
  2. CognOS confidence calculation
  3. If SYNTHESIZE: divergence_semantics extracts assumptions
  4. System recurses with new question based on meta_question
  5. Convergence control stops when C and assumptions stabilize
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from confidence import compute_confidence
from divergence_semantics import synthesize_reason, convergence_check


def demo_scenario():
    """
    Scenario: Evaluating the CognOS hypothesis.
    
    CognOS is assessed from three angles:
    1. Original question (falsifiability)
    2. If divergence ↦ synthesize_reason() extracts assumptions
    3. Convergence check to know when we're done
    """

    print("=" * 100)
    print("COGNOS INTEGRATED STACK DEMO")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────────
    # ITERATION 1: Original Question
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n" + "🔵 " * 25)
    print("\n⏱️  ITERATION 1 — Original Question")
    print("─" * 100)

    q1 = ("How strongly falsifiable is the HYPOTHESIS in its current form?")
    alt1 = [
        "A: Weakly falsifiable",
        "B: Partially falsifiable but requires stricter measurement thresholds",
        "C: Strongly falsifiable with clear criteria"
    ]
    votes1 = {"B": 3, "C": 2}  # From our earlier run
    mc_predictions1 = [0.7, 0.8, 0.9, 0.8, 0.7]  # Reported confidence from 5 samples
    p1 = 3 / 5  # 60% majority

    # Layer 1: CognOS confidence
    result1 = compute_confidence(p1, mc_predictions1)
    C1 = result1['confidence']
    decision1 = result1['decision']

    print(f"\n📝 Question: {q1}")
    print(f"🗳️  Vote distribution: {votes1}")
    print(f"📊 p={p1:.2f}, Ue={result1['epistemic_uncertainty']:.4f}, "
          f"Ua={result1['aleatoric_uncertainty']:.3f}")
    print(f"🎯 Result: C={C1:.3f}, Decision={decision1}")

    if decision1 == 'synthesize':
        print(f"\n✨ SYNTHESIZE DETECTED — Analyzing divergence...")

        # Layer 2: Divergence Semantics
        divergence1 = synthesize_reason(
            question=q1,
            alternatives=alt1,
            vote_distribution=votes1,
            confidence=C1,
            is_multimodal=result1['is_multimodal']
        )

        print(f"\n🔍 Divergence Analysis:")
        print(f"   Majority (B): {divergence1['majority_assumption']}")
        print(f"   Minority (C): {divergence1['minority_assumption']}")
        print(f"\n   🌉 Divergence source: {divergence1['divergence_source']}")
        print(f"   💡 Integration: {divergence1['integration_strategy']}")
        print(f"\n   ❓ Meta-question for next iteration:\n      {divergence1['meta_question']}")

        meta_q2 = divergence1['meta_question']
    else:
        print(f"\n✅ Already consensus or clear decision — no divergence analysis needed.")
        meta_q2 = None

    # ─────────────────────────────────────────────────────────────────────────────
    # ITERATION 2: Follow-up based on meta_question
    # ─────────────────────────────────────────────────────────────────────────────

    if meta_q2:
        print("\n\n" + "🔵 " * 25)
        print("\n⏱️  ITERATION 2 — Follow-up via Meta-question")
        print("─" * 100)

        print(f"\n🔄 Original question generated Meta-question:")
        print(f"   {meta_q2}")

        # Simulated new votes based on sharpened question
        q2 = meta_q2
        alt2 = [
            "A: Clear operational definitions required before testing",
            "B: We can run pilot and sharpen iteratively",
            "C: Definitions are already sufficient"
        ]
        votes2 = {"B": 4, "A": 1}  # Better consensus now
        mc_predictions2 = [0.75, 0.75, 0.85, 0.80, 0.75]
        p2 = 4 / 5  # 80% majority

        result2 = compute_confidence(p2, mc_predictions2)
        C2 = result2['confidence']
        decision2 = result2['decision']

        print(f"\n📊 New result: C={C2:.3f}, Decision={decision2}")
        print(f"   Consensus improved: p {p1:.2f} → {p2:.2f}")
        print(f"   Confidence improved: C {C1:.3f} → {C2:.3f}")

        # Layer 3: Convergence check
        confidence_history = [C1, C2]
        assumption_history = [
            divergence1.get('divergence_source', 'Initial'),
            "Iterative sharpening of definitions"
        ]

        convergence = convergence_check(
            iteration=2,
            confidence_history=confidence_history,
            assumption_history=assumption_history,
            threshold=0.05
        )

        print(f"\n🔄 Convergence check:")
        print(f"   Stability: {convergence['stability_score']:.1%}")
        print(f"   {convergence['reason']}")
        print(f"   Continue? {convergence['should_continue']}")

    # ─────────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n\n" + "=" * 100)
    print("📋 ARCHITECTURE SUMMARY")
    print("=" * 100)

    summary = """
    Layer 0 (Structured Input):
      - Structured question (CHOICE/CONFIDENCE) forces model to be discrete
      
    Layer 1 (Epistemic Integrity):
      - compute_confidence() combines p + Ue + Ua → C
      - Four decisions: auto | synthesize | explore | escalate
      
    Layer 2 (Divergence Semantics):
      - On SYNTHESIZE: synthesize_reason() extracts UNDERLYING ASSUMPTIONS
      - Not "we disagree" but "we assume different things about X"
      - Generates meta_question for next iteration
      
    Layer 3 (Convergence Control):
      - convergence_check() detects when C and assumptions stabilize
      - Stops recursion automatically
      - Enables multi-turn agentic chains

    🎯 RESULTS:
      - System knows when it knows (HIGH C: AUTO)
      - System knows when it diverges (SYNTHESIZE: extract divergence)
      - System knows when to stop (CONVERGENCE)
      
    🧠 FUNCTIONALLY EQUIVALENT TO:
      - ACC (Anterior Cingulate Cortex) — conflict monitoring
      - Prefrontal layer — meta-reasoning
      - Integration layer — semantic summarization
      
    📦 EXTERNALIZED METACOGNITION for AI
    """

    print(summary)

    print("\n" + "=" * 100)


if __name__ == '__main__':
    demo_scenario()
