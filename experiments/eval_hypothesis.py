#!/usr/bin/env python3
"""
eval_hypothesis.py — Run CognOS on the HYPOTHESIS.md.

Loads INSIGHTS.md as system context (otherwise model responds with generic best practice).
Asks four structured questions about hypothesis quality.

Note: Requires an LLM provider function. By default uses environment variables.
If you want to use a specific provider, modify the ask_structur_choice_llm() function.
"""

import sys
from pathlib import Path
import json

# Add research directory to path
RESEARCH_DIR = Path(__file__).parent.parent / "research"
sys.path.insert(0, str(RESEARCH_DIR))

INSIGHTS = (RESEARCH_DIR / "INSIGHTS.md").read_text(encoding="utf-8")
HYPOTHESIS = (RESEARCH_DIR / "HYPOTHESIS.md").read_text(encoding="utf-8")

SYSTEM = f"""You are a scientific methodology advisor for the CognOS project.

=== PROJECT CONTEXT ===
{INSIGHTS}

=== HYPOTHESIS UNDER EVALUATION ===
{HYPOTHESIS}

Answer ONLY in the specified format. Base your evaluation on the project's own findings, not general AI research."""

QUESTIONS = [
    {
        "question": "How falsifiable is the hypothesis in its current formulation?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Yes — the falsification criteria are operational and testable",
            "Partially — the core is testable but the taxonomy requires more precision",
            "No — too vague to be refuted",
        ],
    },
    {
        "question": "What is the hypothesis's most critical weakness right now?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Empirical foundation too narrow — one experiment, one model",
            "U_prompt and U_model overlap too much — edge cases not handled",
            "Decision rule at end is underspecified — needs thresholds",
        ],
    },
    {
        "question": "Are the three uncertainty types (U_model/U_prompt/U_problem) distinct enough to be operationally separable?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Yes — definitions sufficient for 10×3×3 experiment",
            "Partially — definitions hold but edge cases (e.g., high U_model + high U_prompt) lack handling",
            "No — categories too coarse, finer-grained taxonomy needed",
        ],
    },
    {
        "question": "What is most important to validate empirically for the hypothesis to hold?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "That Ue actually varies systematically with prompt format (U_prompt exists)",
            "That majority choice is unstable across formats — not just Ue value",
            "That U_problem questions show persistent low-confidence outcomes in all three formats",
        ],
    },
]


def ask_structured_choice_simple(question: str, alternatives: list[str], 
                                 system: str = "", n_samples: int = 1) -> dict:
    """
    Minimal mock implementation. In production, integrate with your LLM provider:
    - groq.Groq()
    - anthropic.Anthropic()
    - openai.OpenAI()
    - None (use as stub for local testing)
    
    Returns a mock result with reasonable defaults.
    """
    # Stub implementation — replace with real LLM call
    print(f"[Mock LLM] Question: {question}")
    print(f"[Mock LLM] Alternatives: {alternatives}")
    
    # Return a reasonable mock result
    labels = [chr(65 + i) for i in range(len(alternatives))]
    return {
        "decision": "explore",
        "confidence": 0.65,
        "epistemic_ue": 0.12,
        "is_multimodal": False,
        "majority_choice": labels[1],
        "majority_label": alternatives[1],
        "vote_distribution": {labels[0]: 1, labels[1]: 2},
    }


def run():
    print("\n\033[1mCognOS — Evaluation of HYPOTHESIS\033[0m")
    print("=" * 60)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n\033[90mQuestion {i}/{len(QUESTIONS)}...\033[0m")
        result = ask_structured_choice_simple(
            question=q["question"],
            alternatives=q["alternatives"],
            system=SYSTEM,
            n_samples=5,
        )
        results.append((q, result))

    print("\n" + "=" * 60)
    print("\033[1mResults\033[0m\n")

    for q, r in results:
        decision = r["decision"].upper()
        C = r["confidence"]
        Ue = r.get("epistemic_ue", 0.0)
        mm = "⊕" if r.get("is_multimodal") else " "
        winner = r.get("majority_choice", "?")
        votes = r.get("vote_distribution", {})
        vote_str = "  ".join(f"{k}:{v}" for k, v in sorted(votes.items()))

        # winner is "A"/"B"/"C" → index 0/1/2
        labels = q["labels"]
        alts = q["alternatives"]
        try:
            winner_text = alts[labels.index(winner)]
        except (ValueError, IndexError):
            winner_text = "?"

        color = {
            "AUTO": "\033[32m",
            "SYNTHESIZE": "\033[34m",
            "EXPLORE": "\033[33m",
            "ESCALATE": "\033[31m",
        }.get(decision, "")

        print(f"\033[1m{q['question']}\033[0m")
        print(f"  → {winner}: {winner_text}")
        print(f"  {color}[C:{C:.3f} Ue:{Ue:.3f}{mm} → {decision}]\033[0m  votes: {vote_str}")
        print()


if __name__ == "__main__":
    run()
