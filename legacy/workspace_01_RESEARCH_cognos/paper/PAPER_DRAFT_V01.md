# CognOS: Confidence-Driven Decision Gating for Autonomous Systems Under Cost Constraints

**Authors:** Björn Wikström  
**Affiliation:** Independent AI Researcher  
**ORCID:** 0009-0000-4015-2357  
**Date:** February 2026  
**Status:** DRAFT v0.1

---

## Abstract

Autonomous AI systems often require human review for uncertain predictions, but review capacity is fundamentally limited in practice. Existing approaches use simple probability thresholds to decide when to escalate decisions, but this fails to capture the underlying uncertainty structure of model predictions. We present **CognOS**, a confidence-driven arbitration framework that combines epistemic and aleatoric uncertainty to identify high-risk predictions more effectively than baseline methods.

Testing on two domains (medical diagnosis and text classification), we demonstrate that CognOS achieves **40-100% higher safety gains** than probability thresholds at matched escalation rates of 40-55%. However, at escalation rates exceeding 70%, all methods converge due to ceiling effects, revealing a critical **operating regime dependency**.

Our findings suggest CognOS is optimally suited for **cost-constrained deployment scenarios** where escalation budgets limit review capacity—a common constraint in content moderation, medical triage, fraud detection, and customer support systems. We provide implementation guidelines, document failure modes on synthetic data, and discuss the specific conditions under which uncertainty-aware arbitration provides measurable value.

**Keywords:** confidence estimation, model arbitration, epistemic uncertainty, aleatoric uncertainty, human-in-the-loop, cost-constrained AI

---

## 1. Introduction

### 1.1 The Cost-Constrained Escalation Problem

Autonomous AI systems increasingly make high-stakes decisions: medical diagnoses, content moderation, fraud detection, and customer support routing. When models are uncertain, the standard solution is **human escalation**—routing predictions to expert review. However, human review capacity is **fundamentally limited**:

- Content moderation platforms can review 20-40% of flagged content
- Medical triage systems have constrained specialist availability
- Fraud detection teams can investigate 10-30% of alerts
- Customer support can escalate 30-50% of queries

The critical question shifts from *"should we escalate uncertain predictions?"* to **"which predictions should we escalate when we can only review half?"**

### 1.2 Limitations of Probability Thresholds

Current practice uses simple **probability thresholds**: escalate if $p < \tau$. This approach has three fundamental limitations:

1. **Ignores epistemic uncertainty** — High agreement among wrong models appears confident
2. **Ignores aleatoric uncertainty** — Predictions near decision boundaries (p ≈ 0.5) are inherently risky
3. **Conflates prediction with confidence** — $p = 0.75$ from an uncertain model is different from $p = 0.75$ from a certain model

### 1.3 Prior Work

**Uncertainty quantification:**
- Bayesian deep learning (Gal & Ghahramani, 2016)
- Ensemble methods (Lakshminarayanan et al., 2017)
- Conformal prediction (Vovk et al., 2005)

**Human-in-the-loop:**
- Active learning (Settles, 2009)
- Rejection learning (Cortes et al., 2016)
- Confidence calibration (Guo et al., 2017)

**Gap:** Existing work focuses on **accuracy improvement** or **calibration**, not **cost-aware safety optimization**. No prior work evaluates uncertainty methods under **matched escalation budgets**.

### 1.4 Contributions

1. **CognOS formula:** Combined epistemic-aleatoric confidence metric for decision gating
2. **Matched escalation methodology:** Rigorous comparison at equal escalation costs
3. **Operating regime discovery:** Value at 40-55% escalation, convergence at >70%
4. **Cross-domain validation:** Medical (UCI Breast Cancer) + Text (20 Newsgroups)
5. **Failure documentation:** Analysis of synthetic data failure and root cause (Ue too low)

---

## 2. Method

### 2.1 Problem Formulation

**Given:**
- Model prediction: $p \in [0, 1]$ (probability of positive class)
- MC predictions: $\{p_1, ..., p_N\}$ from $N$ forward passes or ensemble members
- Ground truth: $y \in \{0, 1\}$
- Escalation budget: $\beta \in [0, 1]$ (fraction of predictions that can be escalated)

**Goal:** Select which predictions to escalate to maximize **safety gain**—the proportion of overconfident errors blocked.

**Overconfident Error (OE):** A prediction where:
- Model is wrong: $\text{argmax}(p) \neq y$
- Model is confident: $\max(p, 1-p) \geq \tau_{OE}$ (typically 0.6-0.7)

**Metrics:**
- **BOE (Blocked OE):** Overconfident errors that were escalated
- **MOE (Missed OE):** Overconfident errors that went to auto
- **Safety Gain:** $\frac{\text{BOE}}{\text{BOE} + \text{MOE}} \times 100\%$

### 2.2 CognOS Confidence Formula

We define confidence as:

$$C = p \times (1 - U_e - U_a)$$

where:

**Epistemic Uncertainty ($U_e$):** Variance of MC predictions
$$U_e = \text{Var}(\{p_1, ..., p_N\})$$

**Aleatoric Uncertainty ($U_a$):** Proximity to decision boundary
$$U_a = 2 \times p \times (1 - p)$$

**Intuition:**
- $U_e$ captures **model disagreement** (epistemic)
- $U_a$ captures **inherent class ambiguity** (aleatoric)
- $C$ penalizes both types of uncertainty

**Decision Rule:**
$$\text{decision} = \begin{cases} 
\text{auto} & \text{if } C \geq \tau \\
\text{escalate} & \text{if } C < \tau
\end{cases}$$

### 2.3 Baseline Comparisons

We compare three methods:

**Method A (Baseline):** Simple probability threshold
$$\text{escalate if } p < \tau$$

**Method B (v1):** Epistemic only
$$C = p \times (1 - U_e)$$

**Method C (v1.5, CognOS):** Epistemic + Aleatoric
$$C = p \times (1 - U_e - U_a)$$

### 2.4 Matched Escalation Evaluation

**Critical Methodological Contribution:** We do **not** compare methods at a fixed threshold $\tau$. Instead:

1. Fix target escalation rate (e.g., 50%)
2. Find $\tau$ for each method that achieves this rate
3. Compare **safety gain at matched escalation**

This prevents the artifact: "Method A catches more errors because it escalates more, not because it's better."

**Example:**
- Method A at $\tau=0.75$: 50% escalation → 40% safety gain
- Method C at $\tau=0.68$: 50% escalation → 80% safety gain
- **Conclusion:** Method C is 2x better at choosing *which* predictions to escalate

---

## 3. Experiments

### 3.1 Datasets

#### Domain 1: Medical (UCI Breast Cancer)
- **Task:** Binary classification (malignant vs benign)
- **Samples:** 569 (284 train, 285 test)
- **Features:** 30 clinical measurements
- **Model:** RandomForest (30 trees, depth=5)
- **Accuracy:** 93.3%
- **Errors:** 19 (6.7%)
- **Overconfident Errors (p ≥ 0.7):** 5

**Epistemic Uncertainty:**
- Mean $U_e$: 0.036
- $U_e > 0.05$: 24.6% of predictions
- $U_e > 0.10$: 16.1% of predictions

#### Domain 2: Text (20 Newsgroups)
- **Task:** Binary classification (comp.graphics vs sci.space)
- **Samples:** 1960 (1177 train, 783 test)
- **Features:** TF-IDF (2000 terms)
- **Model:** RandomForest (20 trees, depth=4)
- **Accuracy:** 89.0%
- **Errors:** 86 (11.0%)
- **Overconfident Errors (p ≥ 0.55):** 5

**Epistemic Uncertainty:**
- Mean $U_e$: 0.041
- $U_e > 0.05$: 32.8% of predictions
- $U_e > 0.10$: 16.1% of predictions

### 3.2 Synthetic Data Failure (Documented)

We initially tested on synthetic data (100 samples, 4 scenarios, beta distributions). **Result: NO VALUE**—all methods achieved identical safety gain.

**Root Cause Analysis:**
- Mean $U_e = 0.015$ (10x lower than real models)
- Beta distributions have low inherent variance
- MC predictions drawn from same distribution → low $U_e$
- Formula had no opportunity to differentiate

**Conclusion:** Synthetic data must have **realistic epistemic uncertainty** ($U_e \geq 0.03$) or results are meaningless.

### 3.3 Implementation

**MC Predictions:** Generated using individual tree predictions from RandomForest ensembles (analogous to MC Dropout for neural networks).

**Thresholds Tested:** $\tau \in [0.45, 0.75]$ (50 values) to enable fine-grained matched escalation.

**Operating Systems:** Ubuntu 24.04, Python 3.12, scikit-learn 1.8.0

**Code:** Available at [github.com/Applied-AI-Philosophy/cognos](https://github.com/Applied-AI-Philosophy/cognos)

---

## 4. Results

### 4.1 Medical Domain (UCI Breast Cancer)

**Low Escalation (40-52%):**

| Target Esc% | Baseline Safety | CognOS Safety | Δ Safety | Winner |
|-------------|-----------------|---------------|----------|---------|
| 40% | 0% | **60%** | +60% | ✅ CognOS |
| 42% | 40% | **60%** | +20% | ✅ CognOS |
| 48% | 40% | **80%** | +40% | ✅ CognOS |
| 50% | 40% | **80%** | +40% | ✅ CognOS |
| 52% | 40% | **80%** | +40% | ✅ CognOS |

**CognOS wins:** 5/7 comparisons at 40-52% escalation

**High Escalation (>70%):** All methods converge to 80% safety gain (ceiling effect)

### 4.2 Text Domain (20 Newsgroups)

**Low Escalation (40-55%):**

| Target Esc% | Baseline Safety | CognOS Safety | Δ Safety | Winner |
|-------------|-----------------|---------------|----------|---------|
| 40% | 0% | **100%** | +100% | ✅ CognOS |
| 45% | 0% | **100%** | +100% | ✅ CognOS |
| 50% | 0% | **100%** | +100% | ✅ CognOS |
| 55% | 0% | **100%** | +100% | ✅ CognOS |

**CognOS wins:** 4/4 comparisons at 40-55% escalation

### 4.3 Cross-Domain Summary

| Domain | CognOS Wins | Mean $U_e$ | OE Count | Escalation Range |
|--------|-------------|------------|----------|------------------|
| Medical | 5/7 | 0.036 | 5 | 40-52% |
| Text | 4/4 | 0.041 | 5 | 40-55% |

**Conclusion:** CognOS provides **consistent value across domains** at low-to-medium escalation rates.

### 4.4 Operating Regime Discovery

**Key Finding:** CognOS value is **escalation-rate dependent**:

- **40-55% escalation:** CognOS provides 40-100% safety gain over baseline
- **>70% escalation:** All methods converge (ceiling effect)

**Explanation:**
- At high escalation, baseline escalates almost everything → catches all OE
- At low escalation, CognOS **selects smarter** → catches more OE with same budget

**Implication:** CognOS is **not universal**—it targets cost-constrained scenarios.

---

## 5. Discussion

### 5.1 When CognOS Helps

**Optimal Scenarios:**
1. **Cost-constrained escalation** (40-60% review budget)
2. **High epistemic uncertainty** ($U_e > 0.03$)
3. **Overconfident error risk** (model sometimes confidently wrong)

**Example Domains:**
- Content moderation (limited reviewer capacity)
- Medical triage (constrained specialists)
- Fraud detection (expensive investigations)
- Customer support (budget-limited escalation)

### 5.2 When CognOS Doesn't Help

**Scenarios Where Simple Thresholds Suffice:**
1. **High escalation budgets** (>70% review)
2. **Well-calibrated models** (no overconfident errors)
3. **Low epistemic uncertainty** ($U_e < 0.02$)

**Explanation:** Ceiling effect—when you can escalate most predictions, method doesn't matter.

### 5.3 Limitations

#### Small Test Sets
- Only 5 OE per domain (statistical power limited)
- Need validation on 100+ OE datasets (MIMIC-III, production data)

#### Aleatoric Heuristic
- $U_a = 2p(1-p)$ is approximation
- True aleatoric uncertainty requires proper quantification (softmax entropy, data uncertainty)

#### Single Model Type
- Only tested RandomForest ensembles
- Neural networks with MC Dropout may show different behavior

#### Domain Coverage
- Medical + Text validated
- Need: Vision (dermatology), Tabular (fraud detection), LLM routing

### 5.4 Comparison to Related Work

**vs Conformal Prediction:**
- CognOS: Decision gating (which to escalate)
- Conformal: Uncertainty sets (quantify uncertainty)
- **Complementary:** Can combine CognOS confidence with conformal sets

**vs Active Learning:**
- CognOS: Maximize safety within fixed budget
- Active: Maximize learning from labeled samples
- **Different objectives:** Safety vs accuracy improvement

**vs Rejection Learning:**
- CognOS: Epistemic + aleatoric combined
- Rejection: Often prediction-only
- **Methodological:** We use matched escalation evaluation

### 5.5 Future Work

1. **Meta-learned confidence:** Train $C = f(p, U_e, \text{entropy}, \text{features})$
2. **Cost-aware optimization:** Minimize $\text{Risk} + \lambda \cdot \text{EscalationCost}$
3. **Multi-model routing:** Small → Medium → Large model arbitration
4. **Online calibration:** Adapt $\tau$ based on realized outcomes
5. **Production deployment:** Real systems with telemetry feedback

---

## 6. Conclusion

We presented **CognOS**, a confidence-driven decision gating framework that combines epistemic and aleatoric uncertainty for cost-aware model arbitration. Through rigorous **matched escalation evaluation** across two domains, we demonstrated that CognOS achieves **40-100% higher safety gains** than probability thresholds at low-to-medium escalation rates (40-55%).

Critically, we discovered an **operating regime dependency**: CognOS provides value only in cost-constrained scenarios. At high escalation rates (>70%), all methods converge due to ceiling effects. This finding has important implications for deployment—CognOS is **not a universal solution**, but a **practical tool for systems with limited review capacity**.

We documented synthetic data failure (root cause: unrealistic $U_e$), validated cross-domain generalizability (medical + text), and provided implementation guidelines. Our work suggests that uncertainty-aware arbitration is valuable **specifically for cost-constrained autonomous systems**—a common constraint in content moderation, medical triage, fraud detection, and customer support.

**Key Takeaway:** The question is not *whether* to use uncertainty, but **when uncertainty-based methods provide measurable value over simpler alternatives**. For AI systems operating under escalation budgets, CognOS offers a practical answer.

---

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
2. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation. *NeurIPS*.
3. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.
4. Settles, B. (2009). Active learning literature survey. *Computer Sciences Technical Report*.
5. Cortes, C., DeSalvo, G., & Mohri, M. (2016). Learning with rejection. *ALT*.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.

---

## Appendix A: Synthetic Data Analysis

**Setup:** 100 synthetic datapoints, 4 scenarios (correct_confident, correct_uncertain, wrong_confident, wrong_uncertain).

**Failure:**
- Baseline: 0% safety gain
- v1: 0% safety gain  
- v1.5: 66.7% safety gain (but artifact—higher escalation)

**Matched Escalation (98%):** All methods identical (7/7 BOE, 100% safety).

**Root Cause:** Mean $U_e = 0.015$ → Formula $C = p(1 - 0.015) \approx p$ → v1 ≈ baseline

**Lesson:** Synthetic uncertainty must match real model distributions ($U_e \geq 0.03$).

---

## Appendix B: Implementation Details

**CognOS Python Implementation:**

```python
import numpy as np

def compute_confidence(prediction: float, mc_predictions: np.ndarray) -> float:
    """
    Compute CognOS confidence score.
    
    Args:
        prediction: Mean model prediction (ensemble mean or MC mean)
        mc_predictions: Array of predictions from MC forward passes or ensemble members
        
    Returns:
        confidence: C ∈ [0, 1]
    """
    # Epistemic uncertainty (model disagreement)
    Ue = float(np.var(mc_predictions))
    
    # Aleatoric uncertainty (decision boundary proximity)
    Ua = 2 * prediction * (1 - prediction)
    
    # Combined confidence
    C = max(0.0, prediction * (1 - Ue - Ua))
    
    return C

def should_escalate(confidence: float, threshold: float = 0.8) -> bool:
    """
    Decision rule: escalate if confidence below threshold.
    """
    return confidence < threshold
```

**Threshold Selection:**
- Find $\tau$ that matches desired escalation rate on validation set
- Typical range: $\tau \in [0.6, 0.9]$
- Lower $\tau$ = more escalation, higher safety

---

## Appendix C: Figures

*(To be generated)*

**Figure 1:** Pareto curves (Safety Gain vs Escalation Rate) for all methods across both domains

**Figure 2:** Operating regime plot showing convergence at >70% escalation

**Figure 3:** Epistemic uncertainty distributions (synthetic vs real data)

**Figure 4:** BOE/MOE stacked bars at matched escalation points

---

**Draft Status:** v0.1 (Structure complete, needs figures + revision)  
**Word Count:** ~2800 (target: 4000-5000 for full paper)  
**Next Steps:** Generate figures, expand discussion, add related work details
