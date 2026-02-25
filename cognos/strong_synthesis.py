#!/usr/bin/env python3
"""
strong_synthesis.py — Robust LLM-driven epistemic synthesis.

Core innovation:

    conflict → assumptions → geometry → integration

Not:

    conflict → description

This module replaces the weak heuristic synthesis with a structured,
LLM-powered meta-reasoning pipeline that produces:

1. Assumption extraction (what differs between perspectives)
2. Geometric mapping (on which dimensions do they diverge?)
3. Concrete integration strategy (how to resolve in practice)
4. Dynamic meta-alternatives (what should we ask next?)
5. Epistemic gain metrics (how much did we clarify?)

Author: Jasper + Björn
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Any

# Try to import Jasper's ask functions
try:
    sys.path.append('/media/bjorn/iic/Jasper')
    from jasper_brain import ask_groq, ask_github_models
except ImportError:
    ask_groq = None
    ask_github_models = None


def _call_llm(system: str, prompt: str) -> Optional[str]:
    """Try GitHub Models first, then Groq, then return None."""
    if ask_github_models:
        try:
            result = ask_github_models(system, prompt)
            if result:
                return result
        except Exception:
            pass
    if ask_groq:
        try:
            result = ask_groq(system, prompt)
            if result:
                return result
        except Exception:
            pass
    return None


def extract_assumptions_and_geometry(
    question: str,
    alternatives: list[str],
    vote_distribution: dict,
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    PHASE 1: Deep assumption extraction + geometric mapping.

    Returns:
    {
        'majority_assumption': str,      # What the majority assumes
        'minority_assumption': str,      # What the minority assumes
        'divergence_axes': [             # Explicit dimensions of difference
            {
                'dimension': str,        # e.g., "Epistemological certainty"
                'majority_position': float,  # -1 to +1 (e.g., 0.8 = high certainty)
                'minority_position': float,
                'interpretation': str,       # Why this matters
            },
            ...
        ],
        'common_ground': str,            # What they implicitly agree on
        'core_tension': str,             # The irreducible difference
    }
    """
    sorted_votes = sorted(vote_distribution.items(), key=lambda x: x[1], reverse=True)
    majority_choice = sorted_votes[0][0]
    minority_choice = sorted_votes[1][0] if len(sorted_votes) > 1 else None

    # Map choices to alternatives
    choice_to_alt = {}
    for i, alt in enumerate(alternatives):
        label = chr(65 + i)  # A, B, C, ...
        choice_to_alt[label] = alt

    majority_alt = choice_to_alt.get(majority_choice, f"Alternative {majority_choice}")
    minority_alt = choice_to_alt.get(minority_choice, f"Alternative {minority_choice}") if minority_choice else None

    prompt = f"""You are an epistemologist specializing in assumption mapping.

QUESTION: {question}

ALTERNATIVES:
{chr(10).join(f"  {label}: {choice_to_alt.get(label, '?')}" for label in sorted(choice_to_alt.keys()))}

VOTING PATTERN:
  Majority: {majority_choice} — {majority_alt}
  Minority: {minority_choice} — {minority_alt}

{f"CONTEXT: {context}" if context else ""}

YOUR TASK:
1. Identify the CORE ASSUMPTIONS driving majority vs. minority positions.
   Not: "A is better" but "A assumes X is true, B assumes Y is true"

2. Map the divergence onto 2-3 explicit DIMENSIONS (geometric space).
   Examples of dimensions:
   - "Epistemological certainty" (-1=skeptical, +1=confident)
   - "Practical conservatism" (-1=risky, +1=cautious)
   - "Empirical feedback weight" (-1=theory-driven, +1=data-driven)

3. Describe the COMMON GROUND — what both sides implicitly agree on.

4. Articulate the CORE TENSION — the irreducible difference.

RESPOND WITH ONLY VALID JSON (no markdown, no extra text):
{{
  "majority_assumption": "What the majority assumes",
  "minority_assumption": "What the minority assumes",
  "divergence_axes": [
    {{
      "dimension": "Name of the axis",
      "majority_position": 0.7,
      "minority_position": -0.5,
      "interpretation": "Why this dimension matters"
    }}
  ],
  "common_ground": "What both sides implicitly agree on",
  "core_tension": "The key irreducible difference"
}}
"""

    system = "You are an epistemologist. Respond ONLY with valid JSON, no markdown, no explanation."

    llm_to_use = llm_fn if llm_fn else _call_llm
    response_text = llm_to_use(system, prompt)

    if not response_text:
        # Fallback
        return {
            'majority_assumption': f"Majority prefers {majority_choice}",
            'minority_assumption': f"Minority prefers {minority_choice}",
            'divergence_axes': [],
            'common_ground': "Unknown (LLM unavailable)",
            'core_tension': "Could not analyze",
        }

    # Parse JSON
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {
        'majority_assumption': data.get('majority_assumption', f"Majority prefers {majority_choice}"),
        'minority_assumption': data.get('minority_assumption', f"Minority prefers {minority_choice}"),
        'divergence_axes': data.get('divergence_axes', []),
        'common_ground': data.get('common_ground', "Unknown"),
        'core_tension': data.get('core_tension', "Could not determine"),
    }


def generate_integration_strategy(
    question: str,
    majority_assumption: str,
    minority_assumption: str,
    divergence_axes: list,
    common_ground: str,
    core_tension: str,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    PHASE 2: Concrete integration strategy (not just narrative).

    Returns:
    {
        'primary_strategy': str,         # Main way to resolve
        'strategy_details': [            # Concrete steps
            {
                'step': int,
                'action': str,
                'expected_outcome': str,
                'validation_criterion': str,
            },
            ...
        ],
        'remaining_uncertainty': str,    # What still can't be resolved
        'resource_requirement': str,     # What's needed to resolve
        'estimated_effort': str,         # How hard is it? (low/medium/high)
    }
    """

    axes_text = "\n".join([
        f"  - {ax['dimension']}: Majority={ax['majority_position']:.1f}, "
        f"Minority={ax['minority_position']:.1f} ({ax['interpretation'][:60]}...)"
        for ax in divergence_axes[:3]  # Top 3 only
    ])

    prompt = f"""You are a philosophers specializing in conflict resolution through clarification.

SITUATION:
  Question: {question}
  Majority assumes: {majority_assumption}
  Minority assumes: {minority_assumption}
  Common ground: {common_ground}
  Core tension: {core_tension}

DIVERGENCE GEOMETRY:
{axes_text if axes_text else "  (Single-dimensional)"}

YOUR TASK:
Produce a CONCRETE INTEGRATION STRATEGY — not narrative, but actionable.

This means:
1. Primary strategy: The main mechanism to resolve this (e.g., "clarify definition of X")
2. Specific steps: Ordered list of what to DO, not what to THINK
3. Success criteria: How to validate resolution worked
4. Resource requirement: What's needed (data, clarification, clarification level)
5. Effort estimate: How difficult is this? (low/medium/high)

Format: ONLY VALID JSON
{{
  "primary_strategy": "One-sentence strategy (e.g., 'Distinguish epistemological from practical certainty')",
  "strategy_details": [
    {{
      "step": 1,
      "action": "Concrete action (e.g., 'Request definition of falsifiability in context of X')",
      "expected_outcome": "What we expect to learn",
      "validation_criterion": "How to verify it worked"
    }},
    {{
      "step": 2,
      "action": "...",
      "expected_outcome": "...",
      "validation_criterion": "..."
    }}
  ],
  "remaining_uncertainty": "What can't be resolved even with this strategy",
  "resource_requirement": "Theoretical clarity? Empirical data? Expert judgment?",
  "estimated_effort": "low|medium|high"
}}
"""

    system = "You are a philosopher. Respond ONLY with valid JSON."

    llm_to_use = llm_fn if llm_fn else _call_llm
    response_text = llm_to_use(system, prompt)

    if not response_text:
        return {
            'primary_strategy': "Could not generate strategy (LLM unavailable)",
            'strategy_details': [],
            'remaining_uncertainty': "Unknown",
            'resource_requirement': "Unknown",
            'estimated_effort': "high",
        }

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {
        'primary_strategy': data.get('primary_strategy', "Could not determine"),
        'strategy_details': data.get('strategy_details', []),
        'remaining_uncertainty': data.get('remaining_uncertainty', "Unknown"),
        'resource_requirement': data.get('resource_requirement', "Unknown"),
        'estimated_effort': data.get('estimated_effort', 'high'),
    }


def generate_meta_alternatives(
    question: str,
    majority_assumption: str,
    minority_assumption: str,
    integration_strategy: str,
    confidence: float,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    PHASE 3: Generate next meta-alternatives (dynamic, cost-weighted).

    Returns:
    {
        'meta_question': str,            # What should we ask next?
        'meta_alternatives': [           # Next round options
            {
                'label': str,            # e.g., "Prioritize conceptual clarity"
                'action': str,           # What to do
                'relevant_when': str,    # Condition: "If cost of error dominates..."
                'expected_clarity_gain': float,  # 0-1, estimated improvement
                'effort': str,           # low/medium/high
            },
            ...
        ],
        'recommended_next_step': str,    # Which alternative to pursue first
    }
    """

    prompt = f"""You are a strategic researcher designing the next research question.

CURRENT STATE:
  Question: {question}
  Majority assumption: {majority_assumption}
  Minority assumption: {minority_assumption}
  Integration strategy: {integration_strategy}
  Current confidence: {confidence:.3f}

YOUR TASK:
Generate 3-4 NEXT META-ALTERNATIVES — i.e., what should we investigate next to clarify this?

These should be cost-weighted — considering:
- Cost of error (what's at stake if we get this wrong?)
- Practical consequences
- Empirical precision required
- Theoretical vs empirical resolution

For each:
1. Action: What to DO (not think)
2. Relevant when: Under what condition is this the right next step?
3. Clarity gain: Rough estimate of improvement in understanding (0-1)
4. Effort: How hard? (low/medium/high)

Format: ONLY VALID JSON
{{
  "meta_question": "What should we clarify next?",
  "meta_alternatives": [
    {{
      "label": "Alternative name",
      "action": "Concrete next step",
      "relevant_when": "If cost of error dominates... or if precision matters...",
      "expected_clarity_gain": 0.6,
      "effort": "medium"
    }}
  ],
  "recommended_next_step": "Which alternative is most useful now?"
}}
"""

    system = "You are a strategic researcher. Respond ONLY with valid JSON."

    llm_to_use = llm_fn if llm_fn else _call_llm
    response_text = llm_to_use(system, prompt)

    if not response_text:
        return {
            'meta_question': "Could not generate (LLM unavailable)",
            'meta_alternatives': [],
            'recommended_next_step': "Retry with LLM",
        }

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {
        'meta_question': data.get('meta_question', "Unknown"),
        'meta_alternatives': data.get('meta_alternatives', []),
        'recommended_next_step': data.get('recommended_next_step', "No recommendation"),
    }


def compute_epistemic_gain(
    confidence_before: float,
    confidence_after: float,
    clarity_before: str,
    clarity_after: str,
    entropy_reduction: Optional[float] = None,
) -> dict:
    """
    PHASE 4: Measure epistemic improvement.

    Returns:
    {
        'confidence_gain': float,        # Δ confidence
        'clarity_improvement': float,    # Heuristic (0-1)
        'entropy_reduction': float,      # If provided
        'overall_epistemic_gain': float, # Weighted combination
        'interpretation': str,           # What this means
    }
    """

    # Confidence gain
    conf_gain = max(0, confidence_after - confidence_before)

    # Clarity improvement (heuristic: length of clarity_after vs before)
    clarity_before_len = len(clarity_before) if clarity_before else 0
    clarity_after_len = len(clarity_after) if clarity_after else 0
    clarity_imp = max(0, min(1.0, (clarity_after_len - clarity_before_len) / 500))

    # Entropy reduction
    entropy_red = entropy_reduction if entropy_reduction is not None else 0.0

    # Weighted combination: 0.4 * conf_gain + 0.3 * clarity_imp + 0.3 * entropy_red
    overall_gain = 0.4 * conf_gain + 0.3 * clarity_imp + 0.3 * entropy_red

    interpretation = ""
    if overall_gain > 0.2:
        interpretation = "Significant epistemic gain — understanding substantially improved"
    elif overall_gain > 0.1:
        interpretation = "Moderate epistemic gain — useful clarification achieved"
    elif overall_gain > 0.05:
        interpretation = "Modest epistemic gain — incremental improvement"
    else:
        interpretation = "Minimal epistemic gain — little clarification; continue synthesis"

    return {
        'confidence_gain': conf_gain,
        'clarity_improvement': clarity_imp,
        'entropy_reduction': entropy_red,
        'overall_epistemic_gain': overall_gain,
        'interpretation': interpretation,
    }


def synthesize_strong(
    question: str,
    alternatives: list[str],
    vote_distribution: dict,
    confidence: float,
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    Full strong synthesis pipeline:

    conflict → assumptions → geometry → integration → meta → gain

    Returns comprehensive synthesis with:
    - Assumption analysis
    - Geometric mapping of divergence
    - Concrete integration strategy (with action steps)
    - Dynamic meta-alternatives for next round
    - Epistemic gain metrics
    """

    # PHASE 1: Assumptions + Geometry
    phase1 = extract_assumptions_and_geometry(
        question=question,
        alternatives=alternatives,
        vote_distribution=vote_distribution,
        context=context,
        llm_fn=llm_fn,
    )

    # PHASE 2: Integration Strategy
    phase2 = generate_integration_strategy(
        question=question,
        majority_assumption=phase1['majority_assumption'],
        minority_assumption=phase1['minority_assumption'],
        divergence_axes=phase1['divergence_axes'],
        common_ground=phase1['common_ground'],
        core_tension=phase1['core_tension'],
        llm_fn=llm_fn,
    )

    # PHASE 3: Meta-alternatives
    phase3 = generate_meta_alternatives(
        question=question,
        majority_assumption=phase1['majority_assumption'],
        minority_assumption=phase1['minority_assumption'],
        integration_strategy=phase2['primary_strategy'],
        confidence=confidence,
        llm_fn=llm_fn,
    )

    # PHASE 4: Epistemic Gain (post-analysis)
    # clarity_before = question, clarity_after = integration strategy
    phase4 = compute_epistemic_gain(
        confidence_before=confidence,
        confidence_after=min(0.95, confidence + 0.1),  # Placeholder
        clarity_before=question,
        clarity_after=phase2['primary_strategy'],
    )

    # Assemble result
    return {
        'question': question,
        'analysis': {
            'majority_assumption': phase1['majority_assumption'],
            'minority_assumption': phase1['minority_assumption'],
            'divergence_axes': phase1['divergence_axes'],
            'common_ground': phase1['common_ground'],
            'core_tension': phase1['core_tension'],
        },
        'integration': {
            'primary_strategy': phase2['primary_strategy'],
            'strategy_details': phase2['strategy_details'],
            'remaining_uncertainty': phase2['remaining_uncertainty'],
            'resource_requirement': phase2['resource_requirement'],
            'estimated_effort': phase2['estimated_effort'],
        },
        'meta_alternatives': {
            'meta_question': phase3['meta_question'],
            'alternatives': phase3['meta_alternatives'],
            'recommended_next_step': phase3['recommended_next_step'],
        },
        'epistemic_gain': phase4,
        'iteration_depth': 'meta²',  # For recursion tracking
    }


if __name__ == '__main__':
    # Demo
    print("🚀 STRONG SYNTHESIS DEMO")
    print("=" * 80)

    result = synthesize_strong(
        question="Is the hypothesis falsifiable?",
        alternatives=[
            "A: Weakly falsifiable",
            "B: Partially falsifiable with stricter thresholds",
            "C: Strongly falsifiable with clear criteria",
        ],
        vote_distribution={"B": 3, "C": 2},
        confidence=0.309,
    )

    print(f"\n📊 Question: {result['question']}")
    print(f"\n🧠 ASSUMPTIONS & GEOMETRY:")
    print(f"  Majority: {result['analysis']['majority_assumption'][:60]}...")
    print(f"  Minority: {result['analysis']['minority_assumption'][:60]}...")
    print(f"  Common ground: {result['analysis']['common_ground'][:60]}...")
    print(f"  Core tension: {result['analysis']['core_tension'][:60]}...")

    if result['analysis']['divergence_axes']:
        print(f"\n📐 DIVERGENCE GEOMETRY:")
        for ax in result['analysis']['divergence_axes'][:2]:
            print(f"  - {ax['dimension']}: Majority={ax['majority_position']:.1f}, "
                  f"Minority={ax['minority_position']:.1f}")

    print(f"\n🤝 INTEGRATION STRATEGY:")
    print(f"  Primary: {result['integration']['primary_strategy'][:80]}...")
    if result['integration']['strategy_details']:
        print(f"  Steps: {len(result['integration']['strategy_details'])} concrete actions")

    print(f"\n❓ NEXT META-ALTERNATIVES:")
    print(f"  Question: {result['meta_alternatives']['meta_question'][:60]}...")
    print(f"  Recommended: {result['meta_alternatives']['recommended_next_step'][:60]}...")

    print(f"\n📈 EPISTEMIC GAIN:")
    print(f"  Confidence gain: +{result['epistemic_gain']['confidence_gain']:.3f}")
    print(f"  Overall gain: {result['epistemic_gain']['overall_epistemic_gain']:.3f}")
    print(f"  Interpretation: {result['epistemic_gain']['interpretation']}")

    print("\n" + "=" * 80)
