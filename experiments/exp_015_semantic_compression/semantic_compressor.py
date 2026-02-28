"""
exp_015 — Semantic Compression with Epistemic Threshold
Base76 Research Lab

Proof-of-concept: CognOS as prompt pre-processor for Claude Code.
Compresses prompts to reduce token usage while preserving semantic integrity.

Core principle: if coverage confidence < THRESHOLD, fall back to raw text.
This mirrors the epistemic variance collapse finding — a system that cannot
represent its own uncertainty must not silently modify its inputs.

Threshold: 0.90 (configurable)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


THRESHOLD: float = 0.90

# High-signal domain terms (Claude Code / CognOS context)
HIGH_VALUE_TERMS: list[str] = [
    # Architecture
    "mcp", "server", "cognos", "claude", "gateway", "pipeline",
    "function", "module", "component", "layer", "endpoint",
    # Operations
    "validate", "validation", "compress", "compression", "weight",
    "semantic", "token", "efficiency", "threshold", "confidence",
    "epistemic", "uncertainty", "inference", "reasoning",
    # Actions
    "create", "build", "implement", "refactor", "generate",
    "analyze", "evaluate", "measure", "detect",
]

# Filler patterns — safe to remove regardless of language
FILLER_PATTERNS: list[str] = [
    r"^jag tänker att\s*",
    r"^vi skulle behöva\s*",
    r"^idén är att\s*",
    r"^detta bör\s*",
    r"^man skulle kunna\s*",
    r"^kanske\s*",
    r"^i think\s*",
    r"^we should\s*",
    r"^the idea is\s*",
    r"^perhaps\s*",
    r"\bsom gör X\b",
    r"\bför förståelse\b",
]


@dataclass
class CompressionResult:
    input_text: str
    output_text: str
    compressed: str
    removed_fillers: list[str]
    coverage: float
    confident: bool
    mode: str  # "compressed" | "raw_fallback"
    tokens_in: int
    tokens_out: int
    tokens_saved: int

    def report(self) -> str:
        lines = [
            f"MODE:     {self.mode.upper()}",
            f"COVERAGE: {self.coverage * 100:.1f}% "
            f"{'✓ above threshold' if self.confident else '⚠ BELOW 90% → raw fallback'}",
            f"TOKENS:   {self.tokens_in} → {self.tokens_out} "
            f"({'+' if self.tokens_saved < 0 else '-'}{abs(self.tokens_saved)})",
        ]
        if self.removed_fillers:
            lines.append(f"REMOVED:  {self.removed_fillers}")
        return "\n".join(lines)


class SemanticCompressor:
    """
    Compresses prompts using signal/filler weighting.

    If semantic coverage of the compressed form falls below THRESHOLD,
    returns raw input unchanged. This is the epistemic safety mechanism —
    silent loss of nuance is more dangerous than token overhead.
    """

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold

    def process(self, text: str) -> CompressionResult:
        compressed, removed = self._compress(text)
        coverage = self._calculate_coverage(compressed, text)
        confident = coverage >= self.threshold
        output = compressed if confident else text

        tokens_in = self._estimate_tokens(text)
        tokens_out = self._estimate_tokens(output)

        return CompressionResult(
            input_text=text,
            output_text=output,
            compressed=compressed,
            removed_fillers=removed,
            coverage=coverage,
            confident=confident,
            mode="compressed" if confident else "raw_fallback",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=tokens_in - tokens_out,
        )

    def _compress(self, text: str) -> tuple[str, list[str]]:
        cleaned = text
        removed = []

        for pattern in FILLER_PATTERNS:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                removed.append(match.group(0).strip())
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Extract high-signal tokens
        words = cleaned.lower().split()
        high_signal = [
            w for w in words
            if any(kw in w for kw in HIGH_VALUE_TERMS)
        ]

        # If compressed form is too short, keep cleaned text
        compressed = " ".join(dict.fromkeys(high_signal))  # deduplicated
        if len(compressed) < 15:
            compressed = cleaned

        return compressed, removed

    def _calculate_coverage(self, compressed: str, original: str) -> float:
        """
        Coverage = fraction of original semantic keywords preserved.
        High-value terms are weighted 2x.
        """
        orig_words = set(re.findall(r"\b\w{4,}\b", original.lower()))
        comp_words = set(re.findall(r"\b\w{4,}\b", compressed.lower()))

        if not orig_words:
            return 1.0

        covered = sum(
            2 if any(kw in w for kw in HIGH_VALUE_TERMS) else 1
            for w in orig_words
            if w in comp_words or any(kw in w for kw in HIGH_VALUE_TERMS)
        )
        total = sum(
            2 if any(kw in w for kw in HIGH_VALUE_TERMS) else 1
            for w in orig_words
        )

        return round(covered / total, 3)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# ─── Test suite ───────────────────────────────────────────────────────────────

TEST_CASES: list[dict] = [
    {
        "label": "Clear MCP intent (sv)",
        "input": "jag tänker att vi skulle behöva skapa en mcp server med cognos som kan validera semantiskt",
    },
    {
        "label": "Nuanced conditional — must NOT compress",
        "input": "idén är att det bör vara reversibelt men bara under vissa förutsättningar när confidence är låg",
    },
    {
        "label": "Original prompt from session",
        "input": (
            "jag tänker att vi skulle behöva skapa en funktion som gör X inom strukturen "
            "för claude code. Idén är att ha en mcp server med cognos som kan vallidera "
            "och väga värdet semantiskt för förståelse. Detta bör då kunna skapa en bättre "
            "och mindre tokens användning?"
        ),
    },
    {
        "label": "Already compressed — minimal gain expected",
        "input": "build cognos mcp server with semantic validation threshold",
    },
    {
        "label": "High-signal technical prompt",
        "input": "implement epistemic confidence threshold in cognos reasoning layer for token compression",
    },
]


if __name__ == "__main__":
    compressor = SemanticCompressor(threshold=THRESHOLD)

    print(f"CognOS Semantic Compressor — exp_015")
    print(f"Threshold: {THRESHOLD * 100:.0f}%")
    print("=" * 60)

    total_saved = 0
    fallbacks = 0

    for i, case in enumerate(TEST_CASES, 1):
        result = compressor.process(case["input"])
        total_saved += result.tokens_saved
        if not result.confident:
            fallbacks += 1

        print(f"\nTEST {i}: {case['label']}")
        print(f"  IN:  \"{result.input_text[:80]}{'...' if len(result.input_text) > 80 else ''}\"")
        print(f"  OUT: \"{result.output_text[:80]}{'...' if len(result.output_text) > 80 else ''}\"")
        for line in result.report().split("\n"):
            print(f"  {line}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(TEST_CASES)} tests")
    print(f"  Fallbacks (safety):  {fallbacks}/{len(TEST_CASES)}")
    print(f"  Total tokens saved:  {total_saved}")
    print(f"  Avg per prompt:      {total_saved / len(TEST_CASES):.1f}")
