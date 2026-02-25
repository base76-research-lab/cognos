# CognOS Research Experiments

**Monte Carlo epistemic sampling** for evaluating CognOS architecture.

📖 **See [TEST_PROJECT.md](TEST_PROJECT.md) for full research design.**  
🔧 **See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for local Ollama setup.**

---

## Quick Start

**Rekommenderad setup (avoids venv issues on mounted filesystem):**

```bash
# Setup environment in home directory
cd /media/bjorn/iic/cognos-standalone/research
./setup_home_env.sh

# Run experiments from ~/tests
cd ~/tests/cognos-research
./run_exp_001.sh
```

**Se [~/tests/cognos-research/QUICKSTART.md](~/tests/cognos-research/QUICKSTART.md) för detaljer.**

---

## Structure

```
research/
├── TEST_PROJECT.md                     # 🎯 Complete research design
├── ENVIRONMENT_SETUP.md                # 🔧 Ollama + venv setup
├── llm_backend.py                      # Unified LLM interface
├── requirements.txt                    # Dependencies
├── setup_research_env.sh               # Quick setup script
├── run_exp_001_divergence.py           # Experiment 1 runner
├── metrics.py                          # Metric implementations
├── experiment_runner.py                # N-iteration runner
├── exp_001_divergence/                 # Experiment 1: Activation Rate
│   ├── config.yaml                     
│   └── reflection.md                  
├── exp_002_epistemic_gain/             # Experiment 2: vs Baseline
│   ├── config.yaml                     
│   └── reflection.md                  
└── exp_003_illposed/                   # Experiment 3: Bad Question Detection
    ├── config.yaml                     
    └── reflection.md                  
```

---

## The Research Questions

### 1️⃣ Divergence Activation Rate
**Hur ofta aktiveras synthesis när LLM röstar?**

### 2️⃣ Epistemic Gain vs Baseline
**Ger CognOS mätbar förbättring jämfört med direct LLM query?**

### 3️⃣ Ill-Posed Detection
**Kan CognOS identifiera dåliga frågor?**

---

## 🔥 Starkaste Forskningsbidraget

**Inte confidence-formeln.**

**Utan:**

> **Conflict → Assumptions → Geometry → Integration → Meta-loop**

Detta är originellt. Detta är vad papers ska handla om.

---

## 3 Core Experiments

### 1️⃣ Divergence Activation Rate (`exp_001_divergence`)
**Fråga:** Hur ofta aktiveras synthesis?

**Metrics:**
- divergence_detected_rate
- synthesis_success_rate  
- convergence_depth

**Why publishable:** Bevisar att arkitekturen **faktiskt fungerar**, inte bara är teori.

### 2️⃣ Epistemic Gain vs Baseline (`exp_002_epistemic_gain`)
**Fråga:** Är CognOS bättre än direct LLM query?

**Metrics:**
- clarity_score (1-5)
- actionability_score (1-5)
- hallucination_detection

**Why publishable:** Starkt paper-material. Visar **practical utility**.

### 3️⃣ Ill-Posed Detection (`exp_003_illposed`)
**Fråga:** Kan CognOS identifiera dåliga frågor?

**Metrics:**
- detection_accuracy
- reframing_success_rate
- false_positive_rate

**Why publishable:** Där CognOS ska excellera. Divergence semantics är gjord för detta.

---

## Environment Setup

**Rekommenderad lokal setup med Ollama:**

```bash
cd /media/bjorn/iic/cognos-standalone/research

# Auto-setup
./setup_research_env.sh

# Or manually:
python3 -m venv --copies .venv  # Use --copies for mounted filesystems
source .venv/bin/activate
pip install -r requirements.txt
```

**Du har redan dessa Ollama-modeller:**
- `qwen2.5:7b` — **Rekommenderad** (bästa reasoning)
- `phi3:mini` — Snabbare för tester
- `tinyllama` — Mycket snabb men svag

**Se [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) för detaljer om modellval.**

---

## Running Experiments

**Aktivera environment:**
```bash
cd /media/bjorn/iic/cognos-standalone/research
source .venv/bin/activate
```

**Kör experiment 1:**
```bash
python run_exp_001_divergence.py
```

**LLM Backend (auto-detect):**
1. Försöker Ollama först (`qwen2.5:7b`)
2. Fallback till Groq om GROQ_API_KEY satt
3. Fallback till Mock för testing

**Monte Carlo iteration:**
```python
for i in range(N):
    result = run_orchestrator(question)
    log_results(result)

aggregate_metrics()
```

**N = 30-50 per fråga** räcker för publication.

---

## Output Structure

**Efter körning:**
```
exp_XXX/
  ├── config.yaml         # Reproducible settings
  ├── raw_data.json       # All iterations with full trace
  ├── metrics.json        # Computed metrics
  └── reflection.md       # 1-page analysis (fill in observations)
```

---

## Analysis Workflow

1. **Run experiment** → generates raw_data.json + metrics.json
2. **Review data** → look for patterns
3. **Fill reflection.md** → observations, architectural implications
4. **Aggregate** → compare across experiments
5. **Write paper** → TEST_PROJECT.md har struktur

---

## Publication Timeline

| Week | Action | Output |
|------|--------|--------|
| 1 | Run exp_001_divergence | Activation data |
| 2 | Run exp_002_epistemic_gain | Baseline comparison |
| 3 | Run exp_003_illposed | Detection accuracy |
| 4 | Fill reflection.md for all 3 | Qualitative insights |
| 5 | Aggregate + draft paper | First version |
| 6 | Iterate + submit | ArXiv/conference |

**Paper title:**  
*"CognOS: A Recursive Epistemic Validation Framework for LLM Systems"*

**Key contribution:**  
> Conflict → Assumptions → Geometry → Integration → Meta-loop

---

## Reflection Template

Each experiment gets 1 page:

```markdown
# Experiment XXX

## Objective
What we tested.

## Method  
How we tested (N iterations, metrics).

## Observations
What we saw.

## Unexpected Findings
Surprises, edge cases.

## Architectural Implications
What this tells us about CognOS design.

## Next Steps
What to test/fix next.
```

---

## Status

- ✅ Research design (TEST_PROJECT.md)
- ✅ Environment setup (ENVIRONMENT_SETUP.md)
- ✅ LLM backend (Ollama support)
- ✅ 3 experiment configurations
- ✅ Reflection templates
- ⏳ **Run experiments** (N=30-50 per question)
- ⏳ **Fill reflections**
- ⏳ **Write paper**

---

**Remember:** Det starkaste bidraget är arkitekturen, inte metrics.

> Conflict → Assumptions → Geometry → Integration → Meta-loop

**Detta är vad vi publicerar.**
