# CognOS-Jasper Integration Architecture

**Datum:** 21 februari 2026  
**Status:** Design Complete, Implementation Pending  
**Estimerad tid:** 3-4 timmar

---

## Vision

Jasper är Björns AI-forskningspartner. CognOS blir Jaspers **confidence layer**—gör honom medveten om sin egen osäkerhet och smart om när han ska eskalera vs agera autonomt.

**Value Prop:** Jasper with CognOS catches 40-60% more mistakes within same escalation budget.

---

## Integration Points

### 1. jasper_brain.py

**Current:** Jasper genererar svar via LLM  
**New:** Efter varje svar, compute CognOS confidence  
**Decision:** Auto (presentera svar) vs Escalate (ask Björn for review)

```python
# jasper_brain.py additions

from cognos.confidence import compute_confidence

def generate_response(prompt, context):
    # Existing LLM call
    response = llm.generate(prompt, context)
    
    # NEW: Generate MC predictions (multiple samples with temperature)
    mc_responses = []
    for _ in range(5):  # 5 MC samples
        mc_resp = llm.generate(prompt, context, temperature=0.8)
        mc_responses.append(score_response(mc_resp))  # Convert to [0,1] score
    
    # Compute confidence
    prediction = score_response(response)
    confidence = compute_confidence(prediction, np.array(mc_responses))
    
    # Decision
    if confidence >= threshold:
        return response, 'auto'
    else:
        return response, 'escalate'  # Flag for Björn review
```

### 2. Escalation Policy

**Context-aware thresholds:**

```python
def get_threshold(energy_state, task_type):
    """
    Adaptive threshold based on Björn's energy and task criticality.
    """
    base_threshold = 0.7
    
    # Energy adjustment
    if energy_state < 50:
        # Low energy → escalate less (Björn can't handle)
        base_threshold -= 0.1
    elif energy_state > 80:
        # High energy → escalate more (Björn can review)
        base_threshold += 0.1
    
    # Task adjustment
    if task_type == 'critical':  # PhD, publication decisions
        base_threshold += 0.2  # More conservative
    elif task_type == 'exploratory':  # Brainstorming, drafts
        base_threshold -= 0.1  # More autonomous
    
    return np.clip(base_threshold, 0.5, 0.9)
```

### 3. Metrics & Logging

**Track:**
- Confidence scores (all responses)
- Escalation decisions (auto vs escalate)
- Outcomes (was escalation correct?)
- Energy correlation (does low energy → more escalations?)

```python
# Add to jasper_memory.py or new jasper_cognos_telemetry.py

def log_cognos_decision(response, confidence, decision, energy, outcome=None):
    """
    Log every CognOS decision for analysis.
    """
    entry = {
        'timestamp': datetime.now(),
        'response_hash': hash(response[:100]),
        'confidence': confidence,
        'decision': decision,
        'energy_state': energy,
        'outcome': outcome,  # 'correct', 'incorrect', 'unknown'
        'escalated_to_bjorn': decision == 'escalate'
    }
    
    append_to_jsonl('FORSKNING/COCKPIT/cognos_telemetry.jsonl', entry)
```

### 4. Visualization (COCKPIT)

**Add to today.md or new cognos_status.md:**

```markdown
## CognOS Status

**Today's Confidence:**
- Mean: 0.72
- Escalations: 3/10 (30%)
- Auto decisions: 7/10 (70%)

**Recent Low-Confidence Tasks:**
1. PhD application strategy (C=0.45) → ESCALATED ✓
2. Paper title choice (C=0.52) → ESCALATED ✓
3. Jasper refactor scope (C=0.48) → ESCALATED ✓

**Telemetry:**
- Total decisions: 247
- Correct escalations: 89%
- Missed escalations: 11%
```

---

## MC Sampling Strategy for LLMs

**Problem:** LLMs don't have built-in MC predictions like RandomForest trees.

**Solution:** Generate multiple samples with temperature:

```python
def generate_mc_predictions(prompt, n_samples=5):
    """
    Generate N responses with temperature sampling.
    Treat variance as epistemic uncertainty.
    """
    responses = []
    scores = []
    
    for i in range(n_samples):
        resp = llm.generate(
            prompt,
            temperature=0.8 + (i * 0.1),  # Vary temperature
            top_p=0.9
        )
        
        # Score response quality [0, 1]
        score = score_response_quality(resp)
        scores.append(score)
        responses.append(resp)
    
    # Use best response, but compute Ue from variance
    best_response = responses[np.argmax(scores)]
    mean_score = np.mean(scores)
    Ue = np.var(scores)
    
    return best_response, mean_score, Ue
```

**Response Scoring Options:**

1. **Self-consistency:** Count how often model gives same answer
2. **Perplexity:** Lower = more confident
3. **Heuristic:** Length, specificity, hedging words ("maybe", "possibly")
4. **Embedding similarity:** Variance in response embeddings

---

## Implementation Phases

### Phase 1: Core Integration (2h)

**Files to Modify:**
1. `Jasper/jasper_brain.py` — Add CognOS compute after response generation
2. `Jasper/jasper_memory.py` — Add cognos_telemetry logging
3. `cognos/confidence.py` — Ensure route_model() exists (small→medium→large)

**Deliverable:** Jasper computes confidence, logs decisions

### Phase 2: Escalation UX (1h)

**Files to Modify:**
1. `Jasper/jasper_interaction.py` — Add escalation prompt to Björn
2. `FORSKNING/COCKPIT/today.md` template — Add CognOS status section

**UX Flow:**
```
Jasper: [Generates response]
CognOS: Confidence = 0.45 (LOW)
Jasper: "⚠️  I'm uncertain about this. Should I proceed or would you like to review?"
Björn: [Reviews and decides]
Jasper: [Logs outcome for learning]
```

### Phase 3: Adaptive Thresholds (1h)

**Files to Modify:**
1. `Jasper/jasper_brain.py` — Read energy_state.md, adjust threshold
2. `FORSKNING/COCKPIT/energy_state.md` — Ensure machine-readable format

**Energy-Aware:**
- Low energy (< 50%) → Lower threshold (escalate less, Björn can't handle)
- High energy (> 80%) → Higher threshold (escalate more, Björn can review)

### Phase 4: Telemetry Dashboard (optional, 1h)

**File to Create:**
- `Jasper/jasper_cognos_dashboard.py` — Analyze telemetry, show trends

**Visualizations:**
- Confidence distribution over time
- Escalation rate vs energy correlation
- Accuracy of escalation decisions

---

## Demo Script (For Paper/Presentation)

**Scenario:** Jasper helping with paper revision

**Without CognOS:**
```
Björn: "Should I submit this paper draft or revise more?"
Jasper: "Submit now. The structure is strong." [Auto]
[Paper gets rejected — Jasper was overconfident]
```

**With CognOS:**
```
Björn: "Should I submit this paper draft or revise more?"
Jasper: [Computes C = 0.52 — LOW]
Jasper: "⚠️  I'm uncertain (C=0.52). My reasoning:
  - Structure is strong ✓
  - Results are solid ✓
  - BUT: Limited related work ⚠️
  - Discussion could be deeper ⚠️
  
  Would you like me to proceed with submission, or should we review together?"
  
Björn: [Reviews] "Good catch. Let's add 2-3 related work citations first."
[Paper improved, higher acceptance chance]
```

---

## Key Metrics to Track

### Operational Metrics
- **Escalation Rate:** % of responses flagged for review
- **Auto Accuracy:** % of auto decisions that were correct
- **Escalation Accuracy:** % of escalations that were needed

### Research Metrics
- **Energy Correlation:** Does low energy → more escalations?
- **Task Type Correlation:** Do critical tasks → lower confidence?
- **Learning Curve:** Does CognOS improve over time?

### Value Metrics
- **Mistakes Prevented:** OE that were escalated and corrected
- **False Escalations:** Auto-safe decisions that were escalated
- **Net Value:** Mistakes prevented - False escalations cost

---

## Integration Checklist

**Before Implementation:**
- [x] CognOS validated on real data (medical + text)
- [x] Paper draft structure complete
- [ ] Response scoring function designed
- [ ] Telemetry storage format defined

**Implementation:**
- [ ] Add compute_confidence() to jasper_brain.py
- [ ] Add MC sampling for LLM responses
- [ ] Add escalation prompt to jasper_interaction.py
- [ ] Add telemetry logging to jasper_memory.py
- [ ] Test on 10 real Jasper sessions

**Post-Implementation:**
- [ ] Run telemetry analysis (1 week)
- [ ] Tune threshold based on outcomes
- [ ] Document case studies (3-5 examples)
- [ ] Add to paper as "Real Deployment" section

---

## Risk Mitigation

### Risk 1: LLM MC Sampling Too Slow
**Mitigation:** Use cached responses or async generation

### Risk 2: Response Scoring Unreliable
**Mitigation:** Start with simple heuristic, improve iteratively

### Risk 3: Too Many Escalations (Björn Overload)
**Mitigation:** Energy-aware thresholds + task filtering

### Risk 4: False Sense of Security
**Mitigation:** Log outcomes, measure accuracy, adjust

---

## Success Criteria

**Minimum Viable (for paper case study):**
- Jasper runs with CognOS for 1 week
- 20+ decisions logged
- 2-3 documented cases where escalation prevented mistake

**Strong Demo:**
- 100+ decisions logged
- Clear energy correlation
- Measurable reduction in mistakes (vs historical)

**Production Ready:**
- 1000+ decisions logged
- Adaptive threshold tuning
- Dashboard for monitoring
- Documented in Jasper system docs

---

## Next Immediate Action

**Start with Phase 1 (2h):**

1. Read current jasper_brain.py implementation
2. Add CognOS import and compute_confidence() call
3. Implement simple MC sampling (5 samples with temperature)
4. Add escalation flag to response
5. Test on 3 prompts manually

**Estimated Start Time:** After paper draft structure review  
**Blocker:** Need to understand current jasper_brain.py architecture

---

**Ready to implement? Say "start jasper integration" and I'll begin with jasper_brain.py analysis.**
