#!/usr/bin/env python3
"""
cognos_deep.py — Recursive epistemic analysis stack.

Five layers in order:
  -1. check_context_anchor  — is the answer anchored in context, or does the model generalize?
   0. validate_frame        — is the question operationalizable? (U_problem filter)
   1. structured_choice     — object level, MC sampling
   2. analyze_divergence    — on SYNTHESIZE: what drives disagreement?
   3. meta-recursion        — iterate until C stabilizes

Model-agnostic: requires ask_fn(system: str, prompt: str) -> str | None.
Pip-package-ready: no Jasper dependencies.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from confidence import compute_confidence

# ── MC sampling ──────────────────────────────────────────────────────────────

def _mc_samples(system: str, prompt: str, ask_fn, n: int = 5) -> list[str]:
    """n calls to ask_fn — temperature variation handled by provider."""
    out = []
    for _ in range(n):
        r = ask_fn(system, prompt)
        if r:
            out.append(r)
    return out


def _parse_choice(text: str, labels: list[str]) -> tuple[str | None, float]:
    """Parse CHOICE + CONFIDENCE from structured choice response."""
    choice, conf = None, 0.5
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith("CHOICE:"):
            val = l.split(":", 1)[1].strip().upper()
            for label in labels:
                if label in val:
                    choice = label
                    break
        elif l.upper().startswith("CONFIDENCE:"):
            m = re.search(r"[\d.]+", l)
            if m:
                conf = min(1.0, max(0.0, float(m.group())))
    return choice, conf


def _run_choice(question: str, alternatives: list[str], context: str,
                ask_fn, n_samples: int = 5) -> dict:
    """
    MC sampling with structured choice format.
    Core for all layers — returns result dict.
    """
    labels = [chr(65 + i) for i in range(len(alternatives))]
    alt_text = "\n".join(f"{la}: {alt}" for la, alt in zip(labels, alternatives))
    prompt = (
        f"Question: {question}\n\nAlternatives:\n{alt_text}\n\n"
        f"Answer ONLY in this format:\n"
        f"CHOICE: <{'/'.join(labels)}>\nCONFIDENCE: <0.0-1.0>\nRATIONALE: <max 20 words>"
    )
    system = (
        f"You are a decision analyst. Always respond in exactly the format specified.\n\n"
        f"=== CONTEXT ===\n{context}"
        if context else
        "You are a decision analyst. Always respond in exactly the format specified."
    )

    samples = _mc_samples(system, prompt, ask_fn, n_samples)
    choices, confs = [], []
    for s in samples:
        c, k = _parse_choice(s, labels)
        if c is not None:
            choices.append(c)
            confs.append(k)

    if not choices:
        return {
            'decision': 'escalate', 'confidence': 0.0, 'epistemic_ue': 1.0,
            'aleatoric_ua': 0.0, 'is_multimodal': False,
            'majority_choice': None, 'majority_label': None,
            'vote_distribution': {}, 'samples_parsed': 0,
        }

    votes = {la: choices.count(la) for la in labels if la in choices}
    majority = max(votes, key=votes.get)
    p = votes[majority] / len(choices)
    mc_predictions = [k if c == majority else 1.0 - k for c, k in zip(choices, confs)]

    # compute_confidence requires ≥4 samples — pad if necessary
    while len(mc_predictions) < 4:
        mc_predictions.append(p)

    cr = compute_confidence(p, mc_predictions)

    return {
        'decision': cr['decision'],
        'confidence': cr['confidence'],
        'epistemic_ue': cr['epistemic_uncertainty'],
        'aleatoric_ua': cr['aleatoric_uncertainty'],
        'is_multimodal': cr['is_multimodal'],
        'majority_choice': majority,
        'majority_label': alternatives[labels.index(majority)],
        'vote_distribution': votes,
        'raw_confidences': confs,
        'samples_parsed': len(choices),
    }


# ── Layer -1: Context Anchor ─────────────────────────────────────────────────

def check_context_anchor(question: str, context: str, ask_fn,
                          n_samples: int = 3) -> dict:
    """
    Detects U_prompt risk: will the answer be context-anchored,
    or does the model generalize away from given context?
    """
    r = _run_choice(
        question=(
            f"The question is: '{question}'\n"
            "Is it likely that the answer will be based on the given context "
            "rather than general training data?"
        ),
        alternatives=[
            "Yes — the question is specific enough that context drives the answer",
            "Partially — the question can be answered with or without context",
            "No — the question is too general, the model will generalize",
        ],
        context=context,
        ask_fn=ask_fn,
        n_samples=n_samples,
    )
    return {
        'anchored': r['majority_choice'] == 'A',
        'partial': r['majority_choice'] == 'B',
        'confidence': r['confidence'],
        'decision': r['decision'],
        'issue': None if r['majority_choice'] == 'A' else r.get('majority_label'),
    }


# ── Layer 0: Frame Validator ──────────────────────────────────────────────────

def validate_frame(question: str, context: str, ask_fn,
                   n_samples: int = 3) -> dict:
    """
    Detects U_problem before Layer 1 runs.
    If question is ill-posed → escalate directly, save MC cost.
    """
    r = _run_choice(
        question=(
            f"The question is: '{question}'\n"
            "Is the question well-formulated and answerable with a discrete choice "
            "based on available information?"
        ),
        alternatives=[
            "Yes — well-formulated and answerable",
            "Partially — answerable but requires clarification",
            "No — ill-posed, too vague or fundamentally ambiguous (U_problem)",
        ],
        context=context,
        ask_fn=ask_fn,
        n_samples=n_samples,
    )
    return {
        'valid': r['majority_choice'] in ('A', 'B'),
        'uncertainty_type': 'U_problem' if r['majority_choice'] == 'C' else None,
        'confidence': r['confidence'],
        'decision': r['decision'],
    }


# ── Layer 2: Divergence Analysis ─────────────────────────────────────────────

def analyze_divergence(question: str, votes: dict, context: str,
                       ask_fn, n_samples: int = 3) -> dict:
    """
    On SYNTHESIZE: identifies what type of assumption drives the disagreement.
    Output is structured — used as input to next recursive level.
    """
    vote_str = ", ".join(f"{k}:{v}" for k, v in sorted(votes.items()))
    r = _run_choice(
        question=(
            f"The question '{question}' yielded disagreement ({vote_str}). "
            "What is the most likely source of disagreement?"
        ),
        alternatives=[
            "FACTUAL BASE — perspectives rest on different empirical assumptions",
            "VALUES — perspectives prioritize different criteria or goals",
            "INTERPRETATION — perspectives interpret the same information differently",
            "FRAME — perspectives operate in fundamentally different conceptual frames",
        ],
        context=context,
        ask_fn=ask_fn,
        n_samples=n_samples,
    )
    dtype = r.get('majority_label', 'unknown divergence')
    return {
        'divergence_type': dtype,
        'confidence': r['confidence'],
        'decision': r['decision'],
        'votes': votes,
        # Next recursive question — structured, not free text
        'meta_question': (
            f"Given that disagreement stems from {dtype.split(' — ')[0].lower()}: "
            f"which perspective is better grounded in available evidence for '{question}'?"
        ),
    }


# ── Recursive Pipeline ─────────────────────────────────────────────────────────

def cognos_deep(
    question: str,
    context: str,
    alternatives: list[str],
    ask_fn,
    max_depth: int = 4,
    tol: float = 0.05,
    n_samples: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Complete recursive epistemic analysis stack.

    Args:
        question:     The question to analyze
        context:      Project context as system message
        alternatives: Alternatives for Layer 1 (object level)
        ask_fn:       Model call: (system: str, prompt: str) -> str | None
        max_depth:    Max recursion depth (default 4)
        tol:          Stop criterion — |ΔC| < tol → converged (default 0.05)
        n_samples:    MC samples per layer (default 5)
        verbose:      Write progress

    Returns:
        {decision, confidence, layers, context_anchor, frame, converged}
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"\033[90m[CognOS] {msg}\033[0m")

    # Truncate context for all layers — reduces token consumption per call
    # Meta layers: 1200 chars enough to understand project character
    # Layer 1+: 3000 chars provide sufficient substance without hitting rate limits
    MAX_META = 1200
    MAX_LAYER = 3000
    meta_context = context[:MAX_META] + "\n[...truncated...]" if len(context) > MAX_META else context
    layer_context = context[:MAX_LAYER] + "\n[...truncated...]" if len(context) > MAX_LAYER else context

    out: dict = {'question': question, 'layers': [], 'converged': False}

    # ── Layer -1: Context anchor ──────────────────────────────────────────────
    log("Checking context anchor...")
    anchor = check_context_anchor(question, meta_context, ask_fn, n_samples=3)
    out['context_anchor'] = anchor
    if not anchor['anchored']:
        log(f"  ⚠ U_prompt risk: {anchor['issue']}")

    # ── Layer 0: Frame validation ─────────────────────────────────────────────
    log("Validating frame...")
    frame = validate_frame(question, meta_context, ask_fn, n_samples=3)
    out['frame'] = frame
    if not frame['valid']:
        log("  ✗ U_problem — escalating without MC sampling")
        out.update({'decision': 'escalate', 'confidence': 0.0,
                    'reason': 'U_problem — question is ill-posed'})
        return out

    # ── Layer 1+: Recursive structured choice ──────────────────────────────────
    cur_q = question
    cur_alts = alternatives
    prev_C: float | None = None

    for depth in range(max_depth):
        log(f"Layer {depth + 1}: '{cur_q[:70]}...'")
        layer = _run_choice(cur_q, cur_alts, layer_context, ask_fn, n_samples)
        C = layer['confidence']

        out['layers'].append({
            'depth': depth,
            'question': cur_q,
            'decision': layer['decision'],
            'confidence': C,
            'epistemic_ue': layer['epistemic_ue'],
            'is_multimodal': layer['is_multimodal'],
            'votes': layer['vote_distribution'],
            'majority': layer.get('majority_label'),
        })
        log(f"  C:{C:.3f} Ue:{layer['epistemic_ue']:.3f} → {layer['decision'].upper()}")

        # Stop criterion: confidence stabilizes
        if prev_C is not None and abs(C - prev_C) < tol:
            log(f"  Converged (|ΔC|={abs(C - prev_C):.3f} < {tol})")
            out['converged'] = True
            break

        # Stop criterion: not SYNTHESIZE
        if layer['decision'] != 'synthesize':
            break

        # SYNTHESIZE → divergence analysis → reformulated question
        log("  SYNTHESIZE → analyzing divergence...")
        div = analyze_divergence(cur_q, layer['vote_distribution'], meta_context,
                                 ask_fn, n_samples=3)
        out['layers'][-1]['divergence'] = div
        log(f"  Divergence type: {div['divergence_type']}")

        cur_q = div['meta_question']
        cur_alts = [
            "Perspective A is better grounded in available evidence",
            "Perspective B is better grounded in available evidence",
            "No clear priority — perspectives are equivalent",
        ]
        prev_C = C

    # Final result from last layer
    last = out['layers'][-1] if out['layers'] else {}
    out['decision'] = last.get('decision', 'escalate')
    out['confidence'] = last.get('confidence', 0.0)
    out['majority'] = last.get('majority')
    return out
