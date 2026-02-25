#!/usr/bin/env python3
"""
eval_hypothesis.py — Kör CognOS på HYPOTHESIS_V02.md.

Laddar cognos_insights.md som systemkontext (annars svarar modellen med best practice).
Ställer fyra strukturerade frågor om hypotesens kvalitet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Jasper"))
from jasper_cognos import ask_jasper_structured_choice

COGNOS_DIR = Path(__file__).parent
INSIGHTS = (COGNOS_DIR / "cognos_insights.md").read_text(encoding="utf-8")
HYPOTHESIS = (COGNOS_DIR / "HYPOTHESIS_V02.md").read_text(encoding="utf-8")

SYSTEM = f"""Du är vetenskaplig metodrådgivare för CognOS-projektet.

=== PROJEKTKONTEXT ===
{INSIGHTS}

=== HYPOTES UNDER BEDÖMNING ===
{HYPOTHESIS}

Svara ENBART i angivet format. Basera bedömningen på projektets egna fynd, inte på generell AI-forskning."""

QUESTIONS = [
    {
        "question": "Är hypotesen falsifierbar som den är formulerad?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Ja — falsifieringskriterierna är operationella och testbara",
            "Delvis — kärnan är testbar men taxonomin kräver mer precision",
            "Nej — för vag för att kunna motbevisas",
        ],
    },
    {
        "question": "Vad är hypotesens mest kritiska svaghet just nu?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Empirisk grund för smal — ett experiment, en modell",
            "U_prompt och U_model överlappar för mycket — gränsfall hanteras inte",
            "Beslutsregeln i slutet är underspecificerad — behöver trösklar",
        ],
    },
    {
        "question": "Är de tre osäkerhetstyperna (U_model/U_prompt/U_problem) tillräckligt distinkt definierade för att vara operationaliserbart separerbara?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Ja — definitionerna är tillräckliga för 10x3x3-experimentet",
            "Delvis — definitionerna håller men gränsfall (t.ex. hög U_model + hög U_prompt) saknar hantering",
            "Nej — kategorierna är för grova, mer finkornig taxonomi krävs",
        ],
    },
    {
        "question": "Vilket av dessa är det viktigaste att validera empiriskt för att hypotesen ska hålla?",
        "labels": ["A", "B", "C"],
        "alternatives": [
            "Att Ue faktiskt varierar systematiskt med promptformat (U_prompt existerar)",
            "Att majoritetsval är instabilt över format — inte bara Ue-värdet",
            "Att U_problem-frågor visar persistenta lågkonfidensutfall i alla tre format",
        ],
    },
]


def run():
    print("\n\033[1mCognOS — Bedömning av HYPOTHESIS_V02\033[0m")
    print("=" * 60)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n\033[90mFråga {i}/{len(QUESTIONS)}...\033[0m")
        result = ask_jasper_structured_choice(
            question=q["question"],
            alternatives=q["alternatives"],  # lista
            system=SYSTEM,
            n_samples=5,
        )
        results.append((q, result))

    print("\n" + "=" * 60)
    print("\033[1mResultat\033[0m\n")

    for q, r in results:
        decision = r["decision"].upper()
        C = r["confidence"]
        Ue = r["epistemic_ue"]
        mm = "⊕" if r.get("is_multimodal") else " "
        winner = r.get("majority_choice", "?")
        votes = r.get("vote_distribution", {})
        vote_str = "  ".join(f"{k}:{v}" for k, v in sorted(votes.items()))

        # winner är "A"/"B"/"C" → index 0/1/2
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
        print(f"  {color}[C:{C:.3f} Ue:{Ue:.3f}{mm} → {decision}]\033[0m  röster: {vote_str}")
        print()


if __name__ == "__main__":
    run()
