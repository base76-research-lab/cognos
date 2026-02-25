#!/usr/bin/env python3
"""
run_exp_002_epistemic_gain.py

Experiment 2: CognOS vs Baseline Epistemic Gain

Measures:
- Does CognOS pick the ground-truth answer more often than a naive baseline?
- Is CognOS's confidence better calibrated (confident when right, uncertain when wrong)?
- Does CognOS add explanatory value vs single-shot query?

Design:
  BASELINE: Single direct LLM call with multiple-choice format
  COGNOS:   Full ReasoningLoop pipeline (L0–L5)

Primary metric: accuracy_delta = cognos_accuracy - baseline_accuracy
Secondary:      confidence_calibration_error (ECE)
                answer_consistency across samples
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.reasoning_loop import ReasoningLoop
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config():
    config_path = Path(__file__).parent / "exp_002_epistemic_gain" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_dir() -> Path:
    import os
    suffix = os.getenv("COGNOS_OUTPUT_SUFFIX", "")
    name = f"exp_002_epistemic_gain{('_' + suffix) if suffix else ''}"
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
            return "CHOICE: A\nCONFIDENCE: 0.7\nRATIONALE: Mock"
        return mock


# --------------------------------------------------------------------------
# Baseline: single direct LLM query
# --------------------------------------------------------------------------

def baseline_answer(llm_fn, question: str, alternatives: List[str], context: str = "") -> dict:
    """Run a simple single-shot LLM query (no CognOS pipeline)."""
    labels = [chr(65 + i) for i in range(len(alternatives))]
    alt_text = "\n".join(f"{la}: {alt}" for la, alt in zip(labels, alternatives))

    ctx_block = f"\nContext: {context}" if context else ""
    prompt = f"""Question: {question}{ctx_block}

Alternatives:
{alt_text}

Answer ONLY in this format:
CHOICE: <{'/'.join(labels)}>
CONFIDENCE: <0.0-1.0>
RATIONALE: <one sentence>"""

    system = "You are a precise decision analyst. Answer in exactly the specified format."
    response = llm_fn(system, prompt)

    chosen_label = None
    confidence = 0.5
    rationale = ""

    if response:
        for line in response.strip().split('\n'):
            if line.startswith('CHOICE:'):
                text = line.split(':', 1)[1].strip().upper()
                for la in labels:
                    if la in text:
                        chosen_label = la
                        break
            elif line.startswith('CONFIDENCE:'):
                m = re.search(r'[\d.]+', line)
                if m:
                    confidence = min(1.0, max(0.0, float(m.group())))
            elif line.startswith('RATIONALE:'):
                rationale = line.split(':', 1)[1].strip()

    chosen_idx = labels.index(chosen_label) if chosen_label in labels else 0
    chosen_text = alternatives[chosen_idx] if chosen_idx < len(alternatives) else ""

    return {
        "chosen_label": chosen_label,
        "chosen_text": chosen_text,
        "confidence": confidence,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# Ground-truth matching
# --------------------------------------------------------------------------

def matches_ground_truth(answer_text: str, ground_truth: str) -> bool:
    """Check if answer text matches ground truth (fuzzy, case-insensitive)."""
    if not answer_text or not ground_truth:
        return False
    ans = answer_text.lower().strip()
    gt = ground_truth.lower().strip()
    # Exact substring match
    if gt in ans or ans in gt:
        return True
    # Word overlap heuristic (>50% of gt words in ans)
    gt_words = set(gt.split())
    ans_words = set(ans.split())
    overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
    return overlap >= 0.5


def cognos_answer_text(result: dict, alternatives: List[str]) -> str:
    """Extract final answer text from CognOS result."""
    final = result.get('final_answer') or result.get('decision') or ''
    # Try to map to alternatives if final answer is 'auto' or label
    if final in ['auto', 'explore', 'escalate', 'synthesize']:
        # Use majority vote: highest confidence alternative
        # We don't have direct alt mapping — use meta iteration decision
        # Fall back to first alternative if unclear
        final = alternatives[0] if alternatives else final
    return final


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------

def run_experiment():
    config = load_config()
    llm_fn = get_llm_fn()

    questions = config['questions']
    n_samples_cognos = config.get('cognos', {}).get('n_samples', 5)
    max_depth = config.get('max_depth', 3)

    all_results = []

    # Aggregated metrics
    baseline_correct = 0
    cognos_correct = 0
    total = 0

    # Calibration data
    baseline_calib = []   # (confidence, is_correct)
    cognos_calib = []

    # Consistency data
    cognos_decisions = {}  # question_id -> list of choices

    print(f"\n{'='*80}")
    print("EXPERIMENT 002: EPISTEMIC GAIN vs BASELINE")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")
    print(f"CognOS n_samples: {n_samples_cognos}")
    print(f"{'='*80}\n")

    for i, q_config in enumerate(questions, 1):
        question = q_config['question']
        alternatives = q_config.get('alternatives', [])
        ground_truth = q_config.get('ground_truth', '')
        context = q_config.get('context', '')
        q_type = q_config.get('type', 'unknown')

        print(f"[{i}/{len(questions)}] [{q_type}] {question[:65]}...")

        total += 1

        # ---- BASELINE ----
        bl = baseline_answer(llm_fn, question, alternatives, context)
        bl_correct = matches_ground_truth(bl['chosen_text'], ground_truth)
        if bl_correct:
            baseline_correct += 1
        baseline_calib.append((bl['confidence'], bl_correct))

        # ---- COGNOS ----
        loop = ReasoningLoop(
            llm_fn=llm_fn,
            max_depth=max_depth,
            verbose=False,
        )
        cog_result = loop.run(
            question=question,
            alternatives=alternatives,
            n_samples=n_samples_cognos,
        )

        cog_answer = cognos_answer_text(cog_result, alternatives)
        cog_correct = matches_ground_truth(cog_answer, ground_truth)
        if cog_correct:
            cognos_correct += 1
        cognos_calib.append((cog_result['confidence'], cog_correct))

        # Track consistency (what label does majority vote pick?)
        cog_decision = cog_result.get('decision', 'unknown')
        if i not in cognos_decisions:
            cognos_decisions[i] = []
        cognos_decisions[i].append(cog_decision)

        # Print comparison
        status = "✓" if cog_correct else "✗"
        bl_status = "✓" if bl_correct else "✗"
        print(f"  Baseline {bl_status} ({bl['confidence']:.2f}) | CognOS {status} ({cog_result['confidence']:.2f})")
        print(f"  GT: '{ground_truth[:50]}'")
        print(f"  BL: '{bl['chosen_text'][:50]}'")
        print(f"  CG: '{cog_answer[:50]}'")

        all_results.append({
            "question_id": i,
            "question": question,
            "type": q_type,
            "ground_truth": ground_truth,
            "baseline": {
                "chosen_label": bl['chosen_label'],
                "chosen_text": bl['chosen_text'],
                "confidence": bl['confidence'],
                "rationale": bl['rationale'],
                "correct": bl_correct,
            },
            "cognos": {
                "final_answer": cog_answer,
                "confidence": cog_result['confidence'],
                "meta_depth": cog_result.get('meta_depth', 0),
                "converged": cog_result.get('converged', False),
                "convergence_reason": cog_result.get('convergence_reason', ''),
                "correct": cog_correct,
            },
        })

    # ---- Metrics ----
    baseline_accuracy = baseline_correct / total if total > 0 else 0
    cognos_accuracy = cognos_correct / total if total > 0 else 0
    accuracy_delta = cognos_accuracy - baseline_accuracy

    # Expected Calibration Error (simplified)
    def ece(calib_data):
        if not calib_data:
            return 0.0
        bins = [(i/10, (i+1)/10) for i in range(10)]
        total_error = 0
        n = len(calib_data)
        for lo, hi in bins:
            in_bin = [(c, correct) for c, correct in calib_data if lo <= c < hi]
            if in_bin:
                avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
                frac_correct = sum(1 for _, correct in in_bin if correct) / len(in_bin)
                total_error += (len(in_bin) / n) * abs(avg_conf - frac_correct)
        return total_error

    baseline_ece = ece(baseline_calib)
    cognos_ece = ece(cognos_calib)

    metrics = {
        "total_questions": total,
        "baseline_accuracy": baseline_accuracy,
        "cognos_accuracy": cognos_accuracy,
        "accuracy_delta": accuracy_delta,
        "baseline_ece": baseline_ece,
        "cognos_ece": cognos_ece,
        "calibration_delta": baseline_ece - cognos_ece,  # positive = CognOS better calibrated
        "baseline_correct": baseline_correct,
        "cognos_correct": cognos_correct,
    }

    # Save
    output_dir = get_output_dir()

    with open(output_dir / "raw_data.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    delta_sign = "+" if accuracy_delta >= 0 else ""
    print(f"\n{'='*80}")
    print("EXPERIMENT 002 COMPLETE")
    print(f"{'='*80}")
    print(f"Baseline accuracy:   {baseline_accuracy:.3f} ({baseline_correct}/{total})")
    print(f"CognOS accuracy:     {cognos_accuracy:.3f} ({cognos_correct}/{total})")
    print(f"Accuracy delta:      {delta_sign}{accuracy_delta:.3f}")
    print(f"Baseline ECE:        {baseline_ece:.4f}")
    print(f"CognOS ECE:          {cognos_ece:.4f}")
    print(f"Calibration delta:   {delta_sign}{cognos_ece - baseline_ece:.4f} (neg = CognOS better)")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'raw_data.json'}")
    print(f"  - {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    run_experiment()
