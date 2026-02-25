#!/usr/bin/env bash
# exp_009 — sweep över tau-värden
# Kör från exp_009_closed_loop_gate/

set -e
cd "$(dirname "$0")/.."

echo "=== exp_009: tau-sweep ==="

for TAU in 0.50 0.67 0.80; do
    echo ""
    echo "--- tau=$TAU ---"
    python run.py --config configs/mnist.yaml --mode all --tau "$TAU"
done

echo ""
echo "=== Klar ==="
