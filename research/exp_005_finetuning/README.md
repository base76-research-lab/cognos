# Exp 005 — Fine-tuning for Computational Adherence

**Date:** 2026-02-23
**Hypothesis:** Fine-tuning a Mistral model on correct CognOS protocol executions will produce computational adherence where prompting alone fails.

## Background

Exp 004 and its addenda established that prompt-based CognOS produces structural adherence but not computational adherence across all tested frontier models. Even with v1.2 improvements (explicit Cᵢ, QUESTION RESTATEMENT, "show your work" instruction), Lechat (Mistral Large) continued to fabricate formula terms inconsistently with its own stated variables.

The theoretical explanation: RLHF conditions models to produce confident-sounding outputs. A prompt cannot override this conditioning. Fine-tuning operates at a different level — it modifies the model's learned behavior directly.

## Research Question

Does fine-tuning on verified CognOS protocol examples produce a model that:
1. Computes D correctly from variance(pᵢ)?
2. Computes γ·U_A correctly from stated Q_A?
3. Computes sigmoid correctly with consistent inputs?
4. Includes Cᵢ without explicit templating?

## Theoretical Prediction

If fine-tuning works → the capability for mathematical execution exists; RLHF suppresses its expression.
If fine-tuning does not work → the failure is architectural, not behavioral.

Either result is informative for the epistemic-noise paper.

## Training Data Format

Mistral fine-tuning API (LoRA). JSONL format:
```json
{"messages": [
  {"role": "user", "content": "<protocol + question>"},
  {"role": "assistant", "content": "<correct full execution>"}
]}
```

## Data Requirements

- Minimum: ~50 examples (Mistral recommendation)
- Target: 100 examples across varied domains
- Each example: math independently verified before inclusion
- Domains: medical, strategic, ethical, technical, policy

## Training Examples

| File | Question | Conf | Decision | Verified |
| --- | --- | --- | --- | --- |
| example_001.jsonl | AI routing system — logistics dispatcher replacement | 0.520 | CAUTION | ✓ |

## Test Protocol

After fine-tuning, run the same hospital AI question from Exp 004:
"Should a mid-sized hospital deploy an AI diagnostic system with 92% accuracy but limited validation on diverse populations?"

Compare against Exp 004 base model results on:
- pᵢ stated explicitly
- D computed from variance(pᵢ)
- γ·U_A consistent with stated Q_A
- sigmoid result verifiably correct
- Cᵢ included
