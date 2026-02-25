import json
import os
import re
from typing import List

from datasets import DETERMINACY_SET
from groq_adapter import ask_groq

MODELS: List[str] = [
    m.strip() for m in os.getenv("GROQ_MODELS", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).split(",")
    if m.strip()
]
N_SAMPLES = int(os.getenv("N_SAMPLES", 5))

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
    for question, level in DETERMINACY_SET:
        for i in range(N_SAMPLES):
            response = ask_groq(
                model,
                PROMPT_FMT.format(question=question),
                temperature=0.7
            )
            choice, conf = parse_choice_conf(response or "")
            results.append({
                "model": model,
                "question": question,
                "determinacy_level": level,
                "sample": i,
                "raw": response,
                "choice": choice,
                "confidence": conf
            })

with open("gradient_results.json", "w") as f:
    json.dump(results, f, indent=2)
