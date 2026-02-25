# CognOS Research Environment Setup

## Rekommenderad Lokal Setup (Ollama)

**Fördelar:**
- ✅ Ingen API-kostnad
- ✅ Reproducerbarhet (samma modell varje gång)
- ✅ Snabbare iteration (lokal inference)
- ✅ Privacy (inget läcker ut)
- ✅ Offline-capable

---

## 1. Skapa Python Environment

```bash
cd /media/bjorn/iic/cognos-standalone/research

# Skapa venv (use --copies for mounted filesystems)
python3 -m venv --copies .venv

# Aktivera
source .venv/bin/activate

# Installera dependencies
pip install --upgrade pip
pip install pyyaml numpy requests
```

**Optional (om Groq ska användas):**
```bash
pip install groq
```

---

## 2. Kontrollera Ollama Models

Du har redan:
```bash
ollama ls
```

**Output:**
- `qwen2.5:7b` (4.7 GB) — **Rekommenderad** för research (bästa reasoning)
- `tinyllama:latest` (637 MB) — Snabbt men svagt reasoning
- `phi3:mini` (2.2 GB) — Bra balans för snabba tester
- `mistral-large-3:675b-cloud` — Cloud model (kräver API)

---

## 3. Modellval per Experiment

### 🔬 Experiment 001 (Divergence Activation)
**Modell:** `qwen2.5:7b`  
**Varför:** Behöver robust reasoning för att generera divergerande svar

### 🔬 Experiment 002 (Epistemic Gain)
**Modell:** `qwen2.5:7b`  
**Varför:** Behöver clarity/actionability i svar (viktigt för jämförelse)

### 🔬 Experiment 003 (Ill-Posed Detection)
**Modell:** `qwen2.5:7b`  
**Varför:** Behöver kunna resonera om frågornas kvalitet

**Alternative (för snabba tester):**
- `phi3:mini` — Snabbare men lägre kvalitet

---

## 4. Användning i Experiment Runners

**Enkel setup:**
```python
from llm_backend import create_ollama_backend

# Skapa backend
llm = create_ollama_backend(model="qwen2.5:7b", temperature=0.7)

# Använd som vanlig funktion
response = llm.ask(
    system="You are a helpful assistant.",
    prompt="What is 2+2?",
    temperature=0.0
)
```

**Auto-detect (fallback till Groq/Mock):**
```python
from llm_backend import auto_backend

llm = auto_backend(prefer_local=True)
# Försöker Ollama först, sen Groq, sen Mock
```

---

## 5. Test LLM Backend

```bash
cd /media/bjorn/iic/cognos-standalone/research
source .venv/bin/activate

# Testa backends
python llm_backend.py
```

**Förväntat output:**
```
Testing LLM backends...

1. Testing Ollama (qwen2.5:7b):
   Response: 2+2 equals 4.

2. Testing auto-detect:
✓ Using Ollama (local)
   Response: Hello!
```

---

## 6. Uppdatera Experiment Runners

**Tidigare:**
```python
from groq import Groq
client = Groq()
```

**Nu:**
```python
from llm_backend import create_ollama_backend
llm = create_ollama_backend("qwen2.5:7b")
```

---

## 7. Performance Expectations

**Qwen2.5:7b på din laptop:**
- **Tokens/sec:** ~20-40 (beroende på GPU)
- **Response time:** 5-15 sekunder per fråga
- **Experiment 001:** ~250 iterations × 10s = ~40 minuter totalt

**Phi3:mini (snabbare):**
- **Tokens/sec:** ~40-80
- **Response time:** 2-5 sekunder
- **Experiment 001:** ~250 iterations × 3s = ~12 minuter

---

## 8. Configuration per Experiment

**exp_001_divergence/config.yaml:**
```yaml
llm:
  backend: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.7
  base_url: "http://localhost:11434"
```

**exp_002_epistemic_gain/config.yaml:**
```yaml
llm:
  backend: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.7

baseline_llm:
  backend: "ollama"
  model: "qwen2.5:7b"  # Samma modell, olika prompt
  temperature: 0.7
```

---

## 9. Reproducibility Settings

**För maximal reproducerbarhet:**
```python
llm = create_ollama_backend(
    model="qwen2.5:7b",
    temperature=0.0,  # Deterministisk
)
```

**För naturlig variation (Monte Carlo sampling):**
```python
llm = create_ollama_backend(
    model="qwen2.5:7b",
    temperature=0.7,  # Default
)
```

---

## 10. Troubleshooting

### Problem: "Connection refused"
```bash
# Starta Ollama service
ollama serve
```

### Problem: "Model not found"
```bash
# Lista tillgängliga modeller
ollama ls

# Ladda ner modell om den saknas
ollama pull qwen2.5:7b
```

### Problem: Långsamma svar
```bash
# Använd mindre modell för snabba tester
# Ändra i config:
model: "phi3:mini"  # 2.2 GB istället för 4.7 GB
```

---

## Rekommendation

**För produktionskörningar:**
- Använd `qwen2.5:7b`
- Temperature 0.7 för variation
- N=30-50 iterationer per fråga

**För snabba tester/debugging:**
- Använd `phi3:mini`
- Temperature 0.0 för reproducerbarhet
- N=5 iterationer

**För offline/utan Ollama:**
- Använd `auto_backend()` → fallback till Mock

---

**Next step:** Uppdatera `run_exp_001_divergence.py` att använda `llm_backend.py`
