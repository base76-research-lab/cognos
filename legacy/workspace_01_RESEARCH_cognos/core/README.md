# CognOS — Core

The epistemic integrity layer for agentic AI pipelines.

**Two files. One idea:** agents should know when they know — and when they're guessing.

---

## Files

| File | What it does |
| ---- | ------------ |
| `confidence.py` | The CognOS formula: `C = p × (1 - Ue - Ua)` |
| `cognos_deep.py` | Recursive epistemic analysis stack — five layers |

---

## confidence.py

Computes decision confidence from MC sampling results.

### Formula

```
C = p × (1 - Ue - Ua)

p   = prediction probability (majority fraction) [0, 1]
Ue  = epistemic uncertainty (variance of MC samples) [0, 1]
Ua  = aleatoric/semantic risk (ambiguity + irreversibility + blast_radius) / 3
C   = decision confidence [0, 1]
```

### Four decisions

| Decision | Condition | Meaning |
| -------- | --------- | ------- |
| `auto` | C ≥ threshold | High confidence — act autonomously |
| `synthesize` | C low, bimodal Ue | Perspective conflict detected — combine views |
| `explore` | C low, unimodal Ue | Noise — gather more information |
| `escalate` | High irreversibility AND low C | Too risky — human judgment required |

The `synthesize` vs `explore` distinction is key: both have low confidence, but for different reasons. Bimodal uncertainty means two valid perspectives exist and should be combined. Unimodal uncertainty means the signal is weak and more data is needed.

### Usage

```python
from confidence import compute_confidence

p = 0.8                          # majority fraction (4/5 samples agreed)
mc_predictions = [0.9, 0.85, 0.9, 0.8, 0.2]  # per-sample confidence scores

result = compute_confidence(p, mc_predictions)

print(result['decision'])          # 'auto' / 'synthesize' / 'explore' / 'escalate'
print(result['confidence'])        # C value [0, 1]
print(result['epistemic_uncertainty'])  # Ue
print(result['aleatoric_uncertainty'])  # Ua
print(result['is_multimodal'])     # True if bimodal Ue (SYNTHESIZE signal)
```

### Parameters

`compute_confidence(p, mc_predictions, threshold=0.7, irreversibility=0.0, ambiguity=0.0, blast_radius=0.0, synthesis_separation=0.20)`

- `threshold`: Minimum C for AUTO decision (default 0.7)
- `irreversibility`: How hard to undo [0, 1] — raises Ua
- `ambiguity`: How ambiguous the question is [0, 1] — raises Ua
- `blast_radius`: How many systems affected [0, 1] — raises Ua
- `synthesis_separation`: Minimum peak separation to detect bimodal Ue (default 0.20)

---

## cognos_deep.py

Recursive epistemic analysis stack. Runs five layers in sequence, each examining the previous one.

### Architecture

```
Question + Context
  │
  ├─ Layer -1: check_context_anchor()
  │    Will the answer be grounded in context, or will the model generalize?
  │    Detects U_prompt risk before spending API budget.
  │
  ├─ Layer  0: validate_frame()
  │    Is the question well-formed and answerable?
  │    Detects U_problem — exits early if question is ill-posed.
  │    (Saves 5 MC samples that would return noise anyway.)
  │
  ├─ Layer  1: structured choice (object level)
  │    MC sampling with forced VAL/CONFIDENCE/MOTIVATION format.
  │    Returns: decision + C + vote distribution.
  │
  ├─ Layer  2: analyze_divergence()   [only if SYNTHESIZE triggered]
  │    Why did models disagree? What assumption drives the split?
  │    Output: structured divergence type + reformulated meta-question.
  │
  └─ Layer  3+: recursive meta-confidence
       Runs Layer 1 on the divergence meta-question.
       Stops when |C(n) - C(n-1)| < tol (convergence), not after fixed depth.
```

### Why this order matters

Frame validation (Layer 0) runs **before** Layer 1, not after. If the question is U_problem — fundamentally ill-posed — running 5 MC samples produces noise. Early detection is cheaper and more informative.

Context anchor (Layer -1) runs even earlier. It detects whether the model will likely use the provided context or default to training data. This is the empirical observation from the CognOS research: context access does not guarantee context-grounded responses.

### Usage

```python
from cognos_deep import cognos_deep

def ask_fn(system: str, prompt: str) -> str | None:
    # Any LLM provider — model-agnostic
    return your_llm_api(system=system, prompt=prompt)

result = cognos_deep(
    question="Should we use agentic AI or classical ML for this task?",
    context=your_project_context,   # str — relevant background
    alternatives=[
        "Agentic AI — dynamic reasoning required",
        "Classical ML — well-defined input/output mapping",
        "Hybrid — use both where appropriate",
    ],
    ask_fn=ask_fn,
    max_depth=4,      # max recursive layers (default 4)
    tol=0.05,         # convergence tolerance: stop when |ΔC| < tol
    n_samples=5,      # MC samples per layer
    verbose=True,     # print progress
)

print(result['decision'])              # final decision
print(result['confidence'])            # final C
print(result['context_anchor'])        # Layer -1 result
print(result['frame'])                 # Layer 0 result
for layer in result['layers']:
    print(layer['depth'], layer['decision'], layer['confidence'])
    if 'divergence' in layer:
        print("  Divergence type:", layer['divergence']['divergence_type'])
```

### Return value

```python
{
    'question': str,
    'decision': 'auto' | 'synthesize' | 'explore' | 'escalate',
    'confidence': float,
    'majority': str,          # winning alternative text
    'converged': bool,        # True if |ΔC| < tol
    'context_anchor': {
        'anchored': bool,
        'partial': bool,
        'confidence': float,
        'issue': str | None,
    },
    'frame': {
        'valid': bool,
        'uncertainty_type': 'U_problem' | None,
        'confidence': float,
    },
    'layers': [
        {
            'depth': int,
            'question': str,
            'decision': str,
            'confidence': float,
            'epistemic_ue': float,
            'is_multimodal': bool,
            'votes': dict,
            'majority': str,
            'divergence': {           # only if SYNTHESIZE triggered
                'divergence_type': str,
                'meta_question': str,
            }
        },
        ...
    ]
}
```

### Model-agnostic design

`cognos_deep` does not import any LLM provider. The `ask_fn` parameter accepts any callable with signature `(system: str, prompt: str) -> str | None`. This means it works with OpenAI, Anthropic, Groq, Ollama, or any other provider.

```python
# OpenAI example
import openai
def ask_fn(system, prompt):
    r = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content

# Anthropic example
import anthropic
client = anthropic.Anthropic()
def ask_fn(system, prompt):
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text
```

---

## Three uncertainty types

CognOS distinguishes three sources of uncertainty that standard MC sampling conflates:

| Type | Name | Source | Hidden? |
| ---- | ---- | ------ | ------- |
| `U_model` | Model uncertainty | Varies with model training, stable across prompt formats | Partially |
| `U_prompt` | Prompt uncertainty | Varies with prompt format for identical questions | Yes |
| `U_problem` | Problem uncertainty | High across all prompt formats — question is ill-posed | Yes |

**Diagnostic rule:**
- Ue varies with format but not model → `U_prompt`
- Ue high across all formats → `U_problem`
- Ue varies within format (temperature sampling) → `U_model`

Standard MC sampling measures **format-conditioned variance**, not belief variance. This is the core finding that motivates CognOS.

---

## Key finding

> *An LLM with access to context does not guarantee context-grounded responses — it requires a verification layer to detect when the model defaults to statistical best practice despite available context.*

CognOS is that verification layer.

---

## Install (planned)

```bash
pip install cognos
```

Currently in research phase. GitHub: `Applied-Ai-Philosophy/cognos`
