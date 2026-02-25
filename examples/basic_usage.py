"""
CognOS — Reasoning Layer: Basic Usage
======================================

The Reasoning Layer sits between LLM output and agent action.
It answers the question every autonomous agent needs to answer:
    "Should I act on this — or stop and ask?"

Run:
    pip install cognos-ai
    python examples/basic_usage.py
"""

from cognos import compute_confidence


def show(label: str, result: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Scenario: {label}")
    print(f"{'─' * 50}")
    print(f"  Decision:   {result['decision'].upper()}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Ue (epistemic): {result['epistemic_uncertainty']:.3f}  — how much the model disagrees with itself")
    print(f"  Ua (aleatoric): {result['aleatoric_uncertainty']:.3f}  — how risky the action is")


# ── Scenario 1: Routine query, high agreement ───────────────────────────────
# Agent samples the LLM 5 times. Answers cluster tightly → low Ue.
# Action is low-risk → low Ua. CognOS says: go ahead.

result_auto = compute_confidence(
    prediction=0.88,
    mc_predictions=[0.87, 0.89, 0.88, 0.86, 0.90],
    ambiguity=0.05,
    irreversibility=0.10,
    blast_radius=0.05,
)
show("Routine query — consistent, low-stakes", result_auto)


# ── Scenario 2: Medical diagnosis — model splits 60/40 ──────────────────────
# Two clusters of answers (bimodal Ue) → genuine uncertainty.
# High irreversibility → Ua is high. CognOS says: stop, get a human.

result_escalate = compute_confidence(
    prediction=0.62,
    mc_predictions=[0.80, 0.82, 0.40, 0.38, 0.65],
    ambiguity=0.40,
    irreversibility=0.85,
    blast_radius=0.70,
)
show("Medical diagnosis — split model, high stakes", result_escalate)


# ── Scenario 3: Strategic question — coherent conflict ───────────────────────
# Two plausible opposing views (bimodal). Not random noise — genuine tradeoff.
# CognOS says: synthesize the perspectives rather than pick one.

result_synthesize = compute_confidence(
    prediction=0.55,
    mc_predictions=[0.70, 0.72, 0.35, 0.33, 0.55],
    ambiguity=0.50,
    irreversibility=0.30,
    blast_radius=0.25,
)
show("Strategic choice — two coherent perspectives", result_synthesize)


print(f"\n{'─' * 50}")
print("  Decision map:")
print("    auto       → act now")
print("    synthesize → combine the perspectives")
print("    explore    → gather more data")
print("    escalate   → defer to human")
print(f"{'─' * 50}\n")
