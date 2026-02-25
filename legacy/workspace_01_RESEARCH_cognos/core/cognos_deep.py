#!/usr/bin/env python3
"""
cognos_deep.py — Rekursiv epistemisk analysstack.

Fem lager i ordning:
  -1. check_context_anchor  — förankras svaret i kontexten, eller generaliserar modellen?
   0. validate_frame        — är frågan operationaliserbar? (U_problem-filter)
   1. structured_choice     — objektnivå, MC sampling
   2. analyze_divergence    — vid SYNTHESIZE: vad driver oenigheten?
   3. meta-rekursion        — kör om tills C stabiliseras

Modell-agnostisk: kräver ask_fn(system: str, prompt: str) -> str | None.
Pip-paket-redo: inga Jasper-beroenden.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from confidence import compute_confidence

# ── MC sampling ──────────────────────────────────────────────────────────────

def _mc_samples(system: str, prompt: str, ask_fn, n: int = 5) -> list[str]:
    """n anrop till ask_fn — temperaturvariation hanteras av providern."""
    out = []
    for _ in range(n):
        r = ask_fn(system, prompt)
        if r:
            out.append(r)
    return out


def _parse_choice(text: str, labels: list[str]) -> tuple[str | None, float]:
    """Parsa VAL + KONFIDENS från structured choice-svar."""
    choice, conf = None, 0.5
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith("VAL:"):
            val = l.split(":", 1)[1].strip().upper()
            for label in labels:
                if label in val:
                    choice = label
                    break
        elif l.upper().startswith("KONFIDENS:"):
            m = re.search(r"[\d.]+", l)
            if m:
                conf = min(1.0, max(0.0, float(m.group())))
    return choice, conf


def _run_choice(question: str, alternatives: list[str], context: str,
                ask_fn, n_samples: int = 5) -> dict:
    """
    MC sampling med structured choice-format.
    Kärnan för alla lager — returnerar result-dict.
    """
    labels = [chr(65 + i) for i in range(len(alternatives))]
    alt_text = "\n".join(f"{la}: {alt}" for la, alt in zip(labels, alternatives))
    prompt = (
        f"Fråga: {question}\n\nAlternativ:\n{alt_text}\n\n"
        f"Svara ENBART i detta format:\n"
        f"VAL: <{'/'.join(labels)}>\nKONFIDENS: <0.0-1.0>\nMOTIVERING: <max 20 ord>"
    )
    system = (
        f"Du är en beslutsanalytiker. Svara alltid i exakt det format som anges.\n\n"
        f"=== KONTEXT ===\n{context}"
        if context else
        "Du är en beslutsanalytiker. Svara alltid i exakt det format som anges."
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

    # compute_confidence kräver ≥4 samples — padda om nödvändigt
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


# ── Lager -1: Context Anchor ─────────────────────────────────────────────────

def check_context_anchor(question: str, context: str, ask_fn,
                          n_samples: int = 3) -> dict:
    """
    Detekterar U_prompt-risk: kommer svaret vara kontextförankrat,
    eller generaliserar modellen bort från given kontext?
    """
    r = _run_choice(
        question=(
            f"Frågan är: '{question}'\n"
            "Är det troligt att svaret baseras på given kontext snarare än generell träningsdata?"
        ),
        alternatives=[
            "Ja — frågan är specifik nog att kontexten styr svaret",
            "Delvis — frågan kan besvaras med eller utan kontext",
            "Nej — frågan är för generell, modellen kommer att generalisera",
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


# ── Lager 0: Frame Validator ──────────────────────────────────────────────────

def validate_frame(question: str, context: str, ask_fn,
                   n_samples: int = 3) -> dict:
    """
    Detekterar U_problem innan Layer 1 körs.
    Om frågan är ill-posed → escalate direkt, spara MC-kostnad.
    """
    r = _run_choice(
        question=(
            f"Frågan är: '{question}'\n"
            "Är frågan välformulerad och möjlig att besvara med ett diskret val "
            "baserat på tillgänglig information?"
        ),
        alternatives=[
            "Ja — välformulerad och besvarbar",
            "Delvis — besvarbar men kräver precisering",
            "Nej — ill-posed, för vag eller fundamentalt tvetydig (U_problem)",
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


# ── Lager 2: Divergence Analysis ─────────────────────────────────────────────

def analyze_divergence(question: str, votes: dict, context: str,
                       ask_fn, n_samples: int = 3) -> dict:
    """
    Vid SYNTHESIZE: identifierar vilken typ av antagning som driver oenigheten.
    Output är strukturerat — används som input till nästa rekursiva nivå.
    """
    vote_str = ", ".join(f"{k}:{v}" for k, v in sorted(votes.items()))
    r = _run_choice(
        question=(
            f"Frågan '{question}' gav oeniga svar ({vote_str}). "
            "Vad är den mest sannolika orsaken till oenigheten?"
        ),
        alternatives=[
            "FAKTABAS — perspektiven bygger på olika empiriska antagningar",
            "VÄRDERING — perspektiven prioriterar olika kriterier eller mål",
            "TOLKNING — perspektiven tolkar samma information på olika sätt",
            "RAM — perspektiven opererar i fundamentalt olika begreppsramar",
        ],
        context=context,
        ask_fn=ask_fn,
        n_samples=n_samples,
    )
    dtype = r.get('majority_label', 'okänd divergens')
    return {
        'divergence_type': dtype,
        'confidence': r['confidence'],
        'decision': r['decision'],
        'votes': votes,
        # Nästa rekursiva fråga — strukturerat, inte fri text
        'meta_question': (
            f"Givet att oenigheten beror på {dtype.split(' — ')[0].lower()}: "
            f"vilket perspektiv är bättre grundat i tillgänglig evidens för '{question}'?"
        ),
    }


# ── Rekursiv Pipeline ─────────────────────────────────────────────────────────

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
    Fullständig rekursiv epistemisk analysstack.

    Args:
        question:     Frågan att analysera
        context:      Projektkontext som systemmeddelande
        alternatives: Alternativ för Layer 1 (objektnivå)
        ask_fn:       Modell-anrop: (system: str, prompt: str) -> str | None
        max_depth:    Max rekursionsdjup (default 4)
        tol:          Stoppkriterie — |ΔC| < tol → konvergerat (default 0.05)
        n_samples:    MC-samples per lager (default 5)
        verbose:      Skriv ut progress

    Returns:
        {decision, confidence, layers, context_anchor, frame, converged}
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"\033[90m[CognOS] {msg}\033[0m")

    # Trunkera kontext för alla lager — minskar token-förbrukning per anrop
    # Meta-lager: 1200 tecken räcker för att förstå projektets karaktär
    # Layer 1+: 3000 tecken ger tillräcklig substans utan att springa på rate limit
    MAX_META = 1200
    MAX_LAYER = 3000
    meta_context = context[:MAX_META] + "\n[...trunkerat...]" if len(context) > MAX_META else context
    layer_context = context[:MAX_LAYER] + "\n[...trunkerat...]" if len(context) > MAX_LAYER else context

    out: dict = {'question': question, 'layers': [], 'converged': False}

    # ── Lager -1: Context anchor ──────────────────────────────────────────────
    log("Kontextförankring...")
    anchor = check_context_anchor(question, meta_context, ask_fn, n_samples=3)
    out['context_anchor'] = anchor
    if not anchor['anchored']:
        log(f"  ⚠ U_prompt-risk: {anchor['issue']}")

    # ── Lager 0: Frame validation ─────────────────────────────────────────────
    log("Frame-validering...")
    frame = validate_frame(question, meta_context, ask_fn, n_samples=3)
    out['frame'] = frame
    if not frame['valid']:
        log("  ✗ U_problem — eskalerar utan att köra MC sampling")
        out.update({'decision': 'escalate', 'confidence': 0.0,
                    'reason': 'U_problem — frågan är ill-posed'})
        return out

    # ── Lager 1+: Rekursiv structured choice ──────────────────────────────────
    cur_q = question
    cur_alts = alternatives
    prev_C: float | None = None

    for depth in range(max_depth):
        log(f"Lager {depth + 1}: '{cur_q[:70]}...'")
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

        # Stoppkriterie: konfidens stabiliseras
        if prev_C is not None and abs(C - prev_C) < tol:
            log(f"  Konvergerat (|ΔC|={abs(C - prev_C):.3f} < {tol})")
            out['converged'] = True
            break

        # Stoppkriterie: inte SYNTHESIZE
        if layer['decision'] != 'synthesize':
            break

        # SYNTHESIZE → divergensanalys → reformulerad fråga
        log("  SYNTHESIZE → divergensanalys...")
        div = analyze_divergence(cur_q, layer['vote_distribution'], meta_context,
                                 ask_fn, n_samples=3)
        out['layers'][-1]['divergence'] = div
        log(f"  Divergenstyp: {div['divergence_type']}")

        cur_q = div['meta_question']
        cur_alts = [
            "Perspektiv A är bättre grundat i tillgänglig evidens",
            "Perspektiv B är bättre grundat i tillgänglig evidens",
            "Ingen klar prioritet — perspektiven är likvärdiga",
        ]
        prev_C = C

    # Finalt resultat från sista lagret
    last = out['layers'][-1] if out['layers'] else {}
    out['decision'] = last.get('decision', 'escalate')
    out['confidence'] = last.get('confidence', 0.0)
    out['majority'] = last.get('majority')
    return out
