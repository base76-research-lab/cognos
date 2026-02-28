# exp_016 — Embedding-Based Semantic Compression

**Date:** 2026-02-28
**Status:** Complete — principled limitation identified
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

## Results (2026-02-28)

| Test | exp_015 coverage | exp_016 coverage | Besparing |
|------|-----------------|-----------------|-----------|
| FNC abstract | 43.5% → fallback | 98.3% → compressed | 32% |
| Svenska MCP-intent | 40.0% → fallback | 100% → compressed | – |
| Nyansrik kondition | 88.9% → fallback | 100% → compressed | – |
| Teknisk hög-signal | 100% → compressed | 100% → compressed | = |
| Original session-prompt | 29.6% → fallback | 99.3% → compressed | 16% |

exp_016 solves the exp_015 failure on academic and natural language text.

## Principled limitation

Embeddings measure semantic similarity between candidate and original.
They cannot detect loss of conditionality.

Example: *"reversibelt men bara under vissa förutsättningar när confidence är låg"*
scores 100% (candidate = original sentence) — technically correct,
but the system cannot know whether a shorter candidate would preserve
the conditional structure.

This is not an implementation problem. It is a structural one.

**Conclusion:** exp_016 is complete as a tool. It does what it can do.

## Architecture consequence

A complete solution requires two components working together:

```
User prompt
    ↓
LLM (Claude) — compress, preserving conditionality
    ↓
Embedding validator — verify coverage >= 0.90
    ↓
coverage OK? → send compressed
coverage fail? → raw fallback
```

**LLM = compressor. Embedding = validator. They are complementary, not competitors.**

This is the architecture for exp_017.
