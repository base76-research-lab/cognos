# exp_015 — Semantic Compression with Epistemic Threshold

**Date:** 2026-02-28
**Status:** Proof-of-concept
**Author:** Björn Wikström, Base76 Research Lab

## Hypothesis

A CognOS pre-processor can reduce Claude Code prompt token usage by 30–65%
while preserving semantic integrity — provided it refuses to compress when
coverage confidence falls below a defined threshold.

## Core insight

Silent loss of nuance is more dangerous than token overhead.

If the compressor is uncertain about semantic coverage, it must fall back
to raw text. This is the same epistemic principle as in exp_014 and the
Epistemic Variance Collapse preprint:

> *"A system that cannot represent ignorance cannot safely modify its own
> knowledge state."*

## Threshold

`THRESHOLD = 0.90`

Below 90% coverage confidence → raw fallback, no compression.

## Architecture

```
User prompt
    ↓
SemanticCompressor.process()
    ↓
Filler removal (safe patterns)
    ↓
High-signal extraction
    ↓
Coverage calculation (weighted)
    ↓
coverage >= 0.90?
    ├── YES → send compressed (~30-65% fewer tokens)
    └── NO  → send raw (safe fallback)
```

## Usage

```bash
python experiments/exp_015_semantic_compression/semantic_compressor.py
```

## Next steps

- Train coverage model on Swedish semantic vectors
- Integrate as MCP tool in Claude Code pipeline
- Add logging to measure real-world compression rates
- Connect to b76_context_processor.rb for unified token stack
