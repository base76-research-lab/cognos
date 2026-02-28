# exp_017 — LLM Compress + Embedding Validate

**Date:** 2026-02-28
**Status:** Complete
**Depends on:** exp_015, exp_016
**Author:** Björn Wikström, Base76 Research Lab

## Architecture

```
User prompt
    ↓
tokens < 80? → skip (raw, not worth pipeline cost)
    ↓
LLM (llama3.2:1b, local) — compress, preserve conditionality
    ↓
Embedding (nomic-embed-text) — cosine_similarity >= 0.90?
    ├── YES → send compressed
    └── NO  → send raw fallback
```

LLM = compressor (handles conditionality)
Embedding = validator (cheap safety net)
They are complementary, not competing.

## Results

| Test | Mode | Coverage | Tokens | Saving |
|------|------|----------|--------|--------|
| FNC abstract (207t) | COMPRESSED | 92.0% | 207→78 | **62%** |
| Nyansrik kondition (23t) | SKIPPED | — | 23→23 | 0% |
| Svenska MCP-intent (30t) | SKIPPED | — | 30→30 | 0% |
| Original session-prompt (64t) | SKIPPED | — | 64→64 | 0% |
| Kort prompt (4t) | SKIPPED | — | 4→4 | 0% |
| **TOTALT** | | | **328→199** | **39%** |

## Key finding

The LLM correctly compresses academic text (62%) while the embedding
validator confirms semantic integrity (92% > 90% threshold).

Short prompts (<80 tokens) skip the pipeline entirely — no cost,
no risk.

## Comparison across experiments

| Experiment | Method | FNC abstract coverage | Saving |
|------------|--------|-----------------------|--------|
| exp_015 | Regex | 43.5% → fallback | 0% |
| exp_016 | Embedding only | 98.3% → compressed | 32% |
| exp_017 | LLM + Embedding | 92.0% → compressed | **62%** |

## Pipeline cost

| Step | Cost | Who |
|------|------|-----|
| LLM compression | local Ollama | llama3.2:1b (free) |
| Embedding validation | local Ollama | nomic-embed-text (free) |
| Claude API saving | -129 tokens | net positive >300t input |

Break-even: prompts >80 tokens.
At 207 tokens (FNC abstract): saves 129 tokens per call.

## Models

- Compress: `llama3.2:1b` (local, fast, no API cost)
- Validate: `nomic-embed-text` (local, ~0ms overhead)
