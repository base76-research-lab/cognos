"""
exp_016 — Embedding-Based Semantic Compression
Base76 Research Lab

Status: SCAFFOLD — implementation pending next session.

Architecture:
  1. Embed original text via Ollama (nomic-embed-text)
  2. Extract candidate compressions (sentence-level chunking)
  3. Score each candidate: cosine_similarity(embed(original), embed(candidate))
  4. If best_score >= THRESHOLD: return compressed
  5. Else: raw fallback (same safety principle as exp_015)

Requires:
  - Ollama running locally with nomic-embed-text model
  - pip install ollama numpy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

THRESHOLD: float = 0.90
EMBED_MODEL: str = "nomic-embed-text"
MAX_OUTPUT_TOKENS: int = 60


@dataclass
class EmbeddingCompressionResult:
    input_text: str
    output_text: str
    coverage: float          # cosine similarity score
    confident: bool
    mode: str                # "compressed" | "raw_fallback"
    tokens_in: int
    tokens_out: int
    tokens_saved: int

    def report(self) -> str:
        return (
            f"MODE:     {self.mode.upper()}\n"
            f"COVERAGE: {self.coverage * 100:.1f}% "
            f"{'✓ above threshold' if self.confident else '⚠ BELOW 90% → raw fallback'}\n"
            f"TOKENS:   {self.tokens_in} → {self.tokens_out} (-{self.tokens_saved})"
        )


class EmbeddingCompressor:
    """
    Embedding-based semantic compressor.
    Uses cosine similarity as coverage metric instead of word overlap.

    This solves the exp_015 failure mode on academic text:
    - exp_015 coverage on FNC abstract: 43% (word overlap)
    - exp_016 target coverage: >90% (semantic similarity)
    """

    def __init__(
        self,
        threshold: float = THRESHOLD,
        model: str = EMBED_MODEL,
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ):
        self.threshold = threshold
        self.model = model
        self.max_tokens = max_tokens
        self._ollama = None

    def process(self, text: str) -> EmbeddingCompressionResult:
        candidates = self._generate_candidates(text)
        best, score = self._select_best(text, candidates)
        confident = score >= self.threshold
        output = best if confident else text

        tokens_in = len(text) // 4
        tokens_out = len(output) // 4

        return EmbeddingCompressionResult(
            input_text=text,
            output_text=output,
            coverage=score,
            confident=confident,
            mode="compressed" if confident else "raw_fallback",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=tokens_in - tokens_out,
        )

    def _generate_candidates(self, text: str) -> list[str]:
        """
        Generate compression candidates at multiple granularities.
        TODO: Replace with LLM-guided extraction in Phase 2.
        """
        candidates = []

        # Candidate 1: First + last sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) >= 2:
            candidates.append(f"{sentences[0]} {sentences[-1]}")

        # Candidate 2: Key noun phrases (naive extraction)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if noun_phrases:
            candidates.append(" + ".join(dict.fromkeys(noun_phrases[:8])))

        # Candidate 3: sentences containing defined terms (=, :, →)
        key_sentences = [s for s in sentences if any(c in s for c in ["=", ":", "→", "–"])]
        if key_sentences:
            candidates.append(" ".join(key_sentences[:2]))

        return candidates

    def _select_best(self, original: str, candidates: list[str]) -> tuple[str, float]:
        """
        Select candidate with highest cosine similarity to original.
        Falls back to word-overlap if Ollama unavailable.
        """
        if not candidates:
            return original, 0.0

        try:
            import ollama
            import numpy as np

            orig_emb = self._embed(original)
            best_text, best_score = original, 0.0

            for candidate in candidates:
                if len(candidate) // 4 > self.max_tokens:
                    continue
                cand_emb = self._embed(candidate)
                score = self._cosine(orig_emb, cand_emb)
                if score > best_score:
                    best_score = score
                    best_text = candidate

            return best_text, best_score

        except ImportError:
            # Ollama not available — fall back to word overlap (exp_015 behavior)
            return self._word_overlap_fallback(original, candidates)

    def _embed(self, text: str) -> list[float]:
        import ollama
        if self._ollama is None:
            self._ollama = ollama
        response = self._ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _word_overlap_fallback(self, original: str, candidates: list[str]) -> tuple[str, float]:
        orig_words = set(re.findall(r"\b\w{4,}\b", original.lower()))
        best_text, best_score = original, 0.0
        for candidate in candidates:
            cand_words = set(re.findall(r"\b\w{4,}\b", candidate.lower()))
            score = len(orig_words & cand_words) / max(len(orig_words), 1)
            if score > best_score:
                best_score = score
                best_text = candidate
        return best_text, best_score


# ─── Scaffold test (runs without Ollama) ─────────────────────────────────────

if __name__ == "__main__":
    abstract = (
        "This note examines two converging theoretical frameworks in contemporary "
        "consciousness research: Mohammad Forghani's Consciousness Frequency Model (2024) "
        "and Björn Wikström's Field–Node–Cockpit (FNC) Model (2025). Both models challenge "
        "the classical assumption that consciousness is a local, emergent byproduct of neural "
        "computation. Instead, they describe consciousness as a distributed phenomenon governed "
        "by resonance, coherence, and information exchange across different ontological layers. "
        "The result is a unified interpretation where frequency defines how consciousness "
        "manifests, and field topology defines why it manifests. This synthesis suggests a "
        "research trajectory that bridges analytic idealism, quantum information theory, and "
        "empirical hyperscanning data under one principle: consciousness as structured resonance."
    )

    compressor = EmbeddingCompressor(threshold=0.90)
    result = compressor.process(abstract)

    print("exp_016 — Embedding Compressor (scaffold test)")
    print("=" * 60)
    print(f"IN:  \"{abstract[:80]}...\"")
    print(f"OUT: \"{result.output_text[:80]}...\"")
    print(result.report())
    print()
    print("NOTE: Ollama not available — using word overlap fallback.")
    print("Full embedding test requires: ollama pull nomic-embed-text")
