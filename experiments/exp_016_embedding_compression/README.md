# exp_016 — Embedding-Based Semantic Compression

**Date:** 2026-02-28
**Status:** Planned — next session
**Depends on:** exp_015 (semantic compression with epistemic threshold)
**Author:** Björn Wikström, Base76 Research Lab

## Motivation

exp_015 demonstrated that regex-based compression fails on academic text:
- Coverage: 43% on FNC abstract (correct fallback behavior)
- Manual semantic destillation achieves ~84% token reduction
- Gap: 283 tokens = compression potential unreachable by regex

The hypothesis: embedding similarity can replace word overlap as coverage
metric, enabling safe compression of semantically dense academic text.

## Hypothesis

An embedding-based compressor trained on Base76 research papers can achieve
80%+ token reduction on academic text while maintaining >90% semantic coverage
as measured by cosine similarity between original and compressed embeddings.

## Measured baseline (from exp_015)

| Input | Tokens | Target compressed | Target saving |
|-------|--------|-------------------|---------------|
| FNC abstract (169 words) | ~335 | ~52 | 84% |

## Approach

### Phase 1 — Corpus preparation
Collect Base76 papers as training/evaluation corpus:
- From Frequency to Field (FNC + Frequency Model)
- Epistemic Variance Collapse (preprint, zenodo)
- The Shared Mind (Wikström 2025, zenodo.17467745)
- Additional publications from FORSKNING/02_PUBLICATIONS/

### Phase 2 — Embedding model
Use local Ollama (llama3.2:1b or nomic-embed-text) for embeddings.
No external API calls — local-first principle.

```python
# Target interface
compressor = EmbeddingCompressor(
    model="nomic-embed-text",   # via Ollama
    threshold=0.90,             # cosine similarity floor
    max_tokens=60,              # hard cap on output
)
result = compressor.process(abstract)
# result.coverage = cosine_similarity(embed(abstract), embed(result.output))
```

### Phase 3 — Evaluation
Measure:
1. Token reduction (%)
2. Semantic coverage (cosine similarity, target >0.90)
3. Fallback rate (how often raw text is safer)
4. Claude Code response quality: compressed vs raw prompt

## Key constraint (from exp_015)

**Safety threshold is non-negotiable.**
If cosine similarity < 0.90, return raw text.
Silent semantic loss is worse than token overhead.

## Dataset location

```
/media/bjorn/iic/workspace/Base76_Research_Lab/Papers/
/home/bjorn/Hämtningar/From Frequency to Field.pdf
```

## Expected outcome

Regex compressor (exp_015): works on structured technical prompts only
Embedding compressor (exp_016): works on academic text + natural language

Together: full-spectrum token efficiency for Claude Code sessions.
