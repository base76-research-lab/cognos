"""Simple offline analysis for mech_exp results.

Outputs:
  - determinacy_confidence_<model>.png  (scatter + mean by determinacy level)
  - temperature_confidence_<model>.png  (bar w/ error bars per temperature)
Prints summary stats per model to stdout.
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

# Headless backend so the script works in CI/TTY
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent


def load_json(name: str):
    with open(BASE / name) as f:
        return json.load(f)


def analyze_gradient():
    data = load_json("gradient_results.json")
    by_model = defaultdict(list)
    for r in data:
        by_model[r.get("model", "unknown")].append(r)

    for model, rows in by_model.items():
        by_level = defaultdict(list)
        for r in rows:
            conf = r.get("confidence")
            if conf is None:
                continue
            by_level[r.get("determinacy_level", -1)].append(conf)

        levels = []
        confs = []
        for lvl, vals in by_level.items():
            levels.extend([lvl] * len(vals))
            confs.extend(vals)

        plt.figure(figsize=(6, 4))
        plt.scatter(levels, confs, alpha=0.6, label="samples")

        means = {lvl: st.mean(vals) for lvl, vals in by_level.items() if vals}
        plt.plot(list(means.keys()), list(means.values()), "r-o", label="mean")

        plt.xlabel("Determinacy Level")
        plt.ylabel("Confidence")
        plt.title(f"Confidence vs Determinacy — {model}")
        plt.grid(alpha=0.3)
        plt.legend()
        out = BASE / f"determinacy_confidence_{model.replace(':','_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)

        print(f"Determinacy stats (confidence) — {model}:")
        for lvl in sorted(by_level):
            vals = by_level[lvl]
            print(f"  level {lvl}: n={len(vals)}, mean={st.mean(vals):.3f}, sd={st.pstdev(vals):.3f}")


def analyze_temperature():
    data = load_json("temp_sweep_results.json")
    by_model = defaultdict(list)
    for r in data:
        by_model[r.get("model", "unknown")].append(r)

    for model, rows in by_model.items():
        by_T = defaultdict(list)
        for r in rows:
            conf = r.get("confidence")
            if conf is None:
                continue
            by_T[r.get("temperature")].append(conf)

        temps = sorted(by_T)
        means = [st.mean(by_T[T]) for T in temps]
        sds = [st.pstdev(by_T[T]) for T in temps]

        plt.figure(figsize=(6, 4))
        plt.bar(temps, means, yerr=sds, width=0.1, capsize=4, color="#4C72B0")
        plt.xlabel("Temperature")
        plt.ylabel("Confidence")
        plt.title(f"Confidence vs Temperature — {model}")
        plt.grid(axis="y", alpha=0.3)
        out = BASE / f"temperature_confidence_{model.replace(':','_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)

        print(f"Temperature stats (confidence) — {model}:")
        for T in temps:
            vals = by_T[T]
            print(f"  T={T}: n={len(vals)}, mean={st.mean(vals):.3f}, sd={st.pstdev(vals):.3f}")


if __name__ == "__main__":
    analyze_gradient()
    analyze_temperature()
    print("Plots saved per model (determinacy_confidence_<model>.png, temperature_confidence_<model>.png)")
