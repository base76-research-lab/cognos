import subprocess
import json

def ask_ollama(model, prompt, temperature=0.7):

    cmd = [
        "ollama",
        "run",
        model,
    ]
    try:
        result = subprocess.run(
            cmd,
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        out = result.stdout.decode().strip()
        err = result.stderr.decode().strip()
        if err:
            print(f"[ollama stderr] {err}")
        if not out:
            print(f"[ollama warning] Tomt svar för prompt: {prompt!r}")
        return out
    except Exception as e:
        print(f"[ollama error] {e}")
        return f"[error] {e}"
