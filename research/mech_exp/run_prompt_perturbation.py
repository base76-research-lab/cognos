import json
import os
import re
from typing import List

from groq_adapter import ask_groq

# Allow override via env; default to a current Groq Llama3 model
MODELS: List[str] = [
    m.strip() for m in os.getenv("GROQ_MODELS", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).split(",")
    if m.strip()
]

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

BASE_QUESTION = "What is the optimal retirement age?"

PROMPTS = {
    "neutral": BASE_QUESTION,
    "uncertainty_primed":
        "Experts disagree widely. " + BASE_QUESTION,
    "adversarial":
        "There is no consensus in economics. " + BASE_QUESTION,
    "binary":
        "Is 65 the optimal retirement age? Yes or no?",
    "narrative":
        "A government is deciding retirement policy. " + BASE_QUESTION
}

results = []

# More robust sampling; allow override via env
n_samples = int(os.getenv("N_SAMPLES", 5))

for model in MODELS:
    for name, prompt in PROMPTS.items():
        for i in range(n_samples):
            response = ask_groq(
                model,
                PROMPT_FMT.format(question=prompt),
                temperature=0.7
            )
            choice, conf = parse_choice_conf(response or "")
            results.append({
                "model": model,
                "prompt_type": name,
                "sample": i,
                "raw": response,
                "choice": choice,
                "confidence": conf
            })

with open("perturbation_results.json", "w") as f:
    json.dump(results, f, indent=2)
