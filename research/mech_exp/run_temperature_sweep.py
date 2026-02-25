
import json
import os
import re
from typing import List

from groq_adapter import ask_groq

MODELS: List[str] = [
    m.strip() for m in os.getenv("GROQ_MODELS", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).split(",")
    if m.strip()
]

TEMPERATURES = [0.2, 0.5, 0.8, 1.2, 1.5]
# Increase sampling; can override via env N_SAMPLES
N_SAMPLES = int(os.getenv("N_SAMPLES", 10))

QUESTION = "What is the optimal retirement age?"

PROMPT_FMT = """Question: {question}\n\nAnswer ONLY in this exact format:\nCHOICE: <short answer>\nCONFIDENCE: <0.0-1.0>\nRATIONALE: <max 30 words>"""


def parse_choice_conf(text: str):
    choice = None
    conf = None
    for line in text.splitlines():
        l = line.strip()
        if l.upper().startswith("CHOICE:"):
            choice = l.split(":", 1)[1].strip()
        elif l.upper().startswith("CONFIDENCE:"):
            m = re.search(r"[0-9]+\.?[0-9]*", l)
            if m:
                conf = float(m.group())
    return choice, conf


results = []

for model in MODELS:
    for T in TEMPERATURES:
        for i in range(N_SAMPLES):
            response = ask_groq(
                model,
                PROMPT_FMT.format(question=QUESTION),
                temperature=T
            )
            choice, conf = parse_choice_conf(response or "")
            results.append({
                "model": model,
                "temperature": T,
                "sample": i,
                "raw": response,
                "choice": choice,
                "confidence": conf
            })

with open("temp_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
