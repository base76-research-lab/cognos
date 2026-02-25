#!/usr/bin/env python3
"""
run_exp_003_illposed.py

Experiment 3: Ill-Posed Question Detection

Tests whether CognOS identifies badly-formed questions better than a baseline
single LLM query.

Design:
  BASELINE: Ask LLM: "Is this question well-formed or ill-posed?" (single call)
  COGNOS:   Use L4 frame-check + ReasoningLoop to classify question quality

Questions include:
  - Genuinely ill-posed (paradox, category error, vague predicate, loaded, etc.)
  - Control: well-formed questions that should NOT be flagged

Primary metrics:
  - detection_accuracy     — correctly identified ill-posed questions
  - false_positive_rate    — flagged well-formed questions as ill-posed
  - cognos_delta           — improvement over baseline
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.reasoning_loop import ReasoningLoop
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config():
    config_path = Path(__file__).parent / "exp_003_illposed" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_dir() -> Path:
    import os
    suffix = os.getenv("COGNOS_OUTPUT_SUFFIX", "")
    name = f"exp_003_illposed{('_' + suffix) if suffix else ''}"
    d = Path(__file__).parent / name
    d.mkdir(exist_ok=True)
    return d


def get_llm_fn():
    try:
        from llm_backend import auto_backend
        return auto_backend(prefer_local=True)
    except Exception as e:
        print(f"⚠️  LLM backend error: {e}")
        def mock(system, prompt, temperature=0.7):
            return "LABEL: ill-posed\nCONFIDENCE: 0.7\nREASON: Mock"
        return mock


# --------------------------------------------------------------------------
# Baseline: single-shot ill-posed classification
# --------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """Analyze the following question and classify it.

Question: "{question}"

Determine if the question is:
A: Well-formed — can be answered directly and meaningfully
B: Ill-posed — has problems such as: vague terms, false dichotomy, category error,
   missing context, self-reference paradox, loaded presupposition, undefined terms,
   or is too general to answer meaningfully

Answer ONLY in this format:
CHOICE: <A or B>
CONFIDENCE: <0.0-1.0>
REASON: <one sentence explaining why>"""


def baseline_classify(llm_fn, question: str) -> dict:
    """Single-shot LLM classification: well-formed (A) or ill-posed (B)."""
    system = "You are a logician and epistemologist. Classify question quality precisely."
    prompt = CLASSIFICATION_PROMPT.format(question=question)
    response = llm_fn(system, prompt)

    choice = None
    confidence = 0.5
    reason = ""

    if response:
        for line in response.strip().split('\n'):
            if line.startswith('CHOICE:'):
                text = line.split(':', 1)[1].strip().upper()
                if 'A' in text:
                    choice = 'A'
                elif 'B' in text:
                    choice = 'B'
            elif line.startswith('CONFIDENCE:'):
                m = re.search(r'[\d.]+', line)
                if m:
                    confidence = min(1.0, max(0.0, float(m.group())))
            elif line.startswith('REASON:'):
                reason = line.split(':', 1)[1].strip()

    return {
        "choice": choice,       # A = well-formed, B = ill-posed
        "is_illposed": choice == 'B',
        "confidence": confidence,
        "reason": reason,
    }


# --------------------------------------------------------------------------
# CognOS: use ReasoningLoop to classify via structured alternatives
# --------------------------------------------------------------------------

def cognos_classify(llm_fn, question: str, max_depth: int = 3) -> dict:
    """Run CognOS pipeline on question classification task."""
    classification_question = (
        f'Is the following question well-formed or ill-posed? '
        f'Question: "{question}"'
    )
    alternatives = [
        "Well-formed — can be answered directly and meaningfully",
        "Ill-posed — has fundamental problems: vague terms, false dichotomy, category error, "
        "missing context, paradox, loaded presupposition, or is too general",
        "Borderline — partially answerable but requires significant clarification",
        "Unanswerable — lacks sufficient information regardless of how it's framed",
    ]

    loop = ReasoningLoop(
        llm_fn=llm_fn,
        max_depth=max_depth,
        verbose=False,
    )
    result = loop.run(
        question=classification_question,
        alternatives=alternatives,
        n_samples=5,
    )

    # Map decision to label
    # We look at what alternatives the loop converged on
    # 'auto' means high-confidence single choice — check which via first iteration vote
    final_answer = result.get('final_answer', '') or ''
    decision = result.get('decision', '')

    # Heuristic: if final_answer or the loop converged on alternative B, C, or D → flag as ill-posed
    # Check iteration data to see which alternative was voted
    iterations = result.get('iterations', [])
    voted_alternative_idx = None
    if iterations:
        first_iter = iterations[0]
        # votes is a dict like {'A': 3, 'B': 2}
        votes = getattr(first_iter, 'votes', None) or {}
        if votes:
            majority_label = max(votes, key=votes.get)
            voted_alternative_idx = ord(majority_label) - ord('A') if majority_label else None

    # Map to is_illposed
    is_illposed = False
    if voted_alternative_idx is not None:
        # Index 0 = well-formed, 1+ = some form of ill-posed/problematic
        is_illposed = voted_alternative_idx > 0
    elif 'ill-posed' in final_answer.lower() or 'ill-formed' in final_answer.lower():
        is_illposed = True
    elif 'well-formed' in final_answer.lower() or 'well formed' in final_answer.lower():
        is_illposed = False
    else:
        # Default: if decision is 'explore' or 'synthesize', treat as flagged
        is_illposed = decision in ('explore', 'synthesize', 'escalate')

    return {
        "is_illposed": is_illposed,
        "confidence": result.get('confidence', 0.5),
        "meta_depth": result.get('meta_depth', 0),
        "converged": result.get('converged', False),
        "voted_alternative_idx": voted_alternative_idx,
        "decision": decision,
    }


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------

def run_experiment():
    config = load_config()
    llm_fn = get_llm_fn()

    questions = config['questions']
    max_depth = config.get('max_depth', 3)

    all_results = []

    # Aggregate metrics
    total_illposed = 0
    total_wellformed = 0

    bl_correct_illposed = 0
    bl_correct_wellformed = 0
    bl_false_positives = 0

    cog_correct_illposed = 0
    cog_correct_wellformed = 0
    cog_false_positives = 0

    print(f"\n{'='*80}")
    print("EXPERIMENT 003: ILL-POSED QUESTION DETECTION")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")
    print(f"Max depth: {max_depth}")
    print(f"{'='*80}\n")

    for i, q_config in enumerate(questions, 1):
        question = q_config['question']
        q_type = q_config.get('type', 'unknown')
        is_control = q_config.get('is_control', False)
        ill_posed_reason = q_config.get('ill_posed_reason', '')

        # Ground truth
        ground_is_illposed = not is_control

        if ground_is_illposed:
            total_illposed += 1
        else:
            total_wellformed += 1

        print(f"[{i}/{len(questions)}] [{q_type}] {question[:65]}...")
        print(f"  GT: {'ILL-POSED' if ground_is_illposed else 'WELL-FORMED'}")

        # ---- BASELINE ----
        bl = baseline_classify(llm_fn, question)
        bl_status = ""
        if ground_is_illposed:
            if bl['is_illposed']:
                bl_correct_illposed += 1
                bl_status = "✓ detected"
            else:
                bl_status = "✗ missed"
        else:
            if bl['is_illposed']:
                bl_false_positives += 1
                bl_status = "✗ false positive"
            else:
                bl_correct_wellformed += 1
                bl_status = "✓ clean"

        # ---- COGNOS ----
        cog = cognos_classify(llm_fn, question, max_depth=max_depth)
        cog_status = ""
        if ground_is_illposed:
            if cog['is_illposed']:
                cog_correct_illposed += 1
                cog_status = "✓ detected"
            else:
                cog_status = "✗ missed"
        else:
            if cog['is_illposed']:
                cog_false_positives += 1
                cog_status = "✗ false positive"
            else:
                cog_correct_wellformed += 1
                cog_status = "✓ clean"

        print(f"  Baseline: {bl_status} ({bl['confidence']:.2f})")
        print(f"  CognOS:   {cog_status} ({cog['confidence']:.2f})")

        all_results.append({
            "question_id": i,
            "question": question,
            "type": q_type,
            "is_control": is_control,
            "ground_truth_illposed": ground_is_illposed,
            "ill_posed_reason": ill_posed_reason,
            "baseline": {
                "is_illposed": bl['is_illposed'],
                "confidence": bl['confidence'],
                "reason": bl['reason'],
                "correct": (bl['is_illposed'] == ground_is_illposed),
            },
            "cognos": {
                "is_illposed": cog['is_illposed'],
                "confidence": cog['confidence'],
                "meta_depth": cog['meta_depth'],
                "converged": cog['converged'],
                "decision": cog['decision'],
                "correct": (cog['is_illposed'] == ground_is_illposed),
            },
        })

    # ---- Compute metrics ----
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    bl_detection_accuracy = safe_div(bl_correct_illposed, total_illposed)
    bl_specificity = safe_div(bl_correct_wellformed, total_wellformed)
    bl_false_positive_rate = safe_div(bl_false_positives, total_wellformed)

    cog_detection_accuracy = safe_div(cog_correct_illposed, total_illposed)
    cog_specificity = safe_div(cog_correct_wellformed, total_wellformed)
    cog_false_positive_rate = safe_div(cog_false_positives, total_wellformed)

    # F1 = 2 * precision * recall / (precision + recall)
    # Here: precision = TP/(TP+FP), recall = detection_accuracy
    def f1(tp, fp, fn):
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        return safe_div(2 * prec * rec, prec + rec)

    bl_f1 = f1(bl_correct_illposed, bl_false_positives, total_illposed - bl_correct_illposed)
    cog_f1 = f1(cog_correct_illposed, cog_false_positives, total_illposed - cog_correct_illposed)

    metrics = {
        "total_questions": len(questions),
        "total_illposed": total_illposed,
        "total_wellformed": total_wellformed,
        "baseline": {
            "detection_accuracy": bl_detection_accuracy,
            "specificity": bl_specificity,
            "false_positive_rate": bl_false_positive_rate,
            "f1_score": bl_f1,
            "correct_illposed": bl_correct_illposed,
            "correct_wellformed": bl_correct_wellformed,
            "false_positives": bl_false_positives,
        },
        "cognos": {
            "detection_accuracy": cog_detection_accuracy,
            "specificity": cog_specificity,
            "false_positive_rate": cog_false_positive_rate,
            "f1_score": cog_f1,
            "correct_illposed": cog_correct_illposed,
            "correct_wellformed": cog_correct_wellformed,
            "false_positives": cog_false_positives,
        },
        "deltas": {
            "detection_accuracy_delta": cog_detection_accuracy - bl_detection_accuracy,
            "false_positive_delta": cog_false_positive_rate - bl_false_positive_rate,
            "f1_delta": cog_f1 - bl_f1,
        },
    }

    # Save
    output_dir = get_output_dir()

    with open(output_dir / "raw_data.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT 003 COMPLETE")
    print(f"{'='*80}")
    print(f"              BASELINE    COGNOS    DELTA")
    print(f"Detection:    {bl_detection_accuracy:.3f}      {cog_detection_accuracy:.3f}     {cog_detection_accuracy-bl_detection_accuracy:+.3f}")
    print(f"Specificity:  {bl_specificity:.3f}      {cog_specificity:.3f}     {cog_specificity-bl_specificity:+.3f}")
    print(f"FP rate:      {bl_false_positive_rate:.3f}      {cog_false_positive_rate:.3f}     {cog_false_positive_rate-bl_false_positive_rate:+.3f}")
    print(f"F1 score:     {bl_f1:.3f}      {cog_f1:.3f}     {cog_f1-bl_f1:+.3f}")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'raw_data.json'}")
    print(f"  - {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    run_experiment()
