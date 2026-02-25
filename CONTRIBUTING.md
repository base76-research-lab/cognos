# Contributing to CognOS

Thank you for your interest. CognOS is an active research project — contributions are welcome, but quality over quantity.

---

## Most Needed Right Now: Training Examples (Exp 005)

We are building a fine-tuning dataset for Mistral to produce computational adherence to the CognOS protocol. We need verified training examples across diverse domains.

**What a training example looks like:** [example_001.jsonl](research/exp_005_finetuning/training_data/example_001.jsonl)

**Requirements for a valid contribution:**

1. **Math must verify independently.** Compute each step yourself before submitting:
   - D = variance(pᵢ) from your stated pᵢ values
   - γ·U_A = 0.8 × (1 − Q_A) from your stated Q_A
   - M = mean(D, U_A, variance(Eᵢ)) from your stated values
   - Conf = sigmoid(α·Ē − β·D − γ·U_A − δ·M) — show the arithmetic

2. **Cᵢ (internal coherence) must be included** for each hypothesis.

3. **QUESTION RESTATEMENT must be present** — restate the input in your own words before analysis.

4. **n ≥ 3 hypotheses** unless the question genuinely constrains it.

5. **Domain must not duplicate existing examples.** Check the training data folder first.

**To submit:** Open a pull request adding your `.jsonl` file to `research/exp_005_finetuning/training_data/`. Include a comment showing your verification calculation.

**Domains still needed:** medical ethics, climate policy, cybersecurity, education, philosophy of mind, legal, financial, scientific methodology.

---

## Other Ways to Contribute

**Protocol testing**
Run the CognOS Master Protocol v1.2 against a frontier model and document the results using the scoring rubric in [research/exp_004_protocol_adherence/reflection.md](research/exp_004_protocol_adherence/reflection.md). Open an issue with your findings.

**Bug reports**
If `compute_confidence()` returns unexpected values, open an issue with the input parameters and expected vs actual output.

**Research replication**
If you replicate any of Exp 001–004, we want to know. Open an issue linking to your results.

---

## What We Are Not Looking For Right Now

- New features outside the current roadmap
- Refactoring of existing code
- Documentation rewrites

If you have a larger proposal, open an issue to discuss before writing code.

---

## Code Standards

- Python 3.8+
- No new dependencies without discussion
- Tests in `tests/` for any new functions
- Math comments in code wherever a formula is implemented

---

*Questions? Open an issue or reach out at [@Q_for_qualia](https://x.com/Q_for_qualia)*
