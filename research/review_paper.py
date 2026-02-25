#!/usr/bin/env python3
"""
review_paper.py — Kör paper-utkastet genom CognOS och få en epistemisk bedömning.

CognOS utvärderar paprets claims, evidens och publicerbarhet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from cognos.orchestrator import CognOSOrchestrator
from llm_backend import auto_backend

PAPER_PATH = Path("/media/bjorn/iic/paper/pågående/epistemic-noise/draft_v0.1.md")

def load_paper_abstract(path: Path) -> str:
    text = path.read_text()
    # Extrahera Abstract + Introduction som kontext
    lines = text.split("\n")
    context_lines = []
    in_section = False
    for line in lines:
        if line.startswith("## Abstract") or line.startswith("## 1."):
            in_section = True
        if in_section and line.startswith("## 2."):
            break
        if in_section:
            context_lines.append(line)
    return "\n".join(context_lines[:80])  # Max 80 rader

QUESTIONS = [
    {
        "question": "Is the core claim scientifically sound: that RLHF alignment suppresses epistemic variance, making frontier models unsuitable substrates for external metacognitive architectures?",
        "alternatives": [
            "A: Strongly supported — the claim is precise and experimentally grounded",
            "B: Partially supported — plausible but needs stronger empirical backing",
            "C: Insufficient evidence — correlation shown but causation unclear",
            "D: Flawed premise — alignment and epistemic variance are not causally linked",
        ]
    },
    {
        "question": "Does the experimental design (4 model classes, Exp 001 divergence activation) provide sufficient evidence for the paper's main hypothesis?",
        "alternatives": [
            "A: Yes — sufficient for an arxiv preprint on this topic",
            "B: Preliminary but promising — needs temperature sweep and semantic divergence",
            "C: Insufficient — too few questions (12) and models (4) for robust claims",
            "D: No — the ground truth labels in config.yaml are arbitrary, invalidating comparisons",
        ]
    },
    {
        "question": "Is the central architectural claim justified: that metacognition is a property of architectures, not models?",
        "alternatives": [
            "A: Well justified — CognOS demonstrates this empirically",
            "B: Conceptually sound but overclaimed without quality-gain experiment (Exp 002)",
            "C: Needs clarification — 'metacognition' used loosely vs cognitive science definition",
            "D: Fundamentally flawed — this conflates meta-level processing with metacognition",
        ]
    },
    {
        "question": "What is the primary weakness that would draw reviewer criticism?",
        "alternatives": [
            "A: Limited model coverage — 4 models is not enough for a scaling claim",
            "B: No semantic divergence measurement — confidence variance ≠ semantic variance",
            "C: Conflation of alignment-induced confidence with genuine epistemic overconfidence",
            "D: Missing related work — ReAct, Reflexion, chain-of-thought address similar problems",
        ]
    },
    {
        "question": "In current form (v0.1 draft), is this paper publishable?",
        "alternatives": [
            "A: ArXiv ready after minor editing",
            "B: Needs Exp 002 (quality gain) and temperature sweep before submission",
            "C: Major revision — needs semantic divergence, more models, stronger related work",
            "D: Not yet — the core insight is strong but evidence base is too preliminary",
        ]
    },
]

def main():
    print("=" * 70)
    print("COGNOS PAPER REVIEW")
    print("Paper: Epistemic Noise as Signal (draft v0.1)")
    print("=" * 70)

    llm = auto_backend()
    orchestrator = CognOSOrchestrator(llm_fn=llm, max_depth=2)

    context = load_paper_abstract(PAPER_PATH)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] {q['question'][:70]}...")
        result = orchestrator.orchestrate(
            question=q["question"],
            alternatives=q["alternatives"],
            context=context,
            n_samples=5,
        )
        results.append({
            "question": q["question"],
            "answer": result.get("final_answer", "N/A"),
            "confidence": result.get("confidence", 0),
            "converged": result.get("converged", False),
            "meta_depth": len(result.get("layers", [])),
        })
        print(f"  → {result.get('final_answer', 'N/A')}")
        print(f"     confidence={result.get('confidence', 0):.3f}  depth={len(result.get('layers', []))}")

    print("\n" + "=" * 70)
    print("SAMMANFATTNING")
    print("=" * 70)
    for r in results:
        conf = r['confidence']
        flag = "⚠" if conf < 0.7 else "✓"
        print(f"{flag} [{conf:.2f}] {r['answer'][:80]}")

    print("\nLåg confidence = CognOS identifierar genuine osäkerhet i bedömningen.")

if __name__ == "__main__":
    main()
