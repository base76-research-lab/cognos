#!/bin/bash
# setup_research_env.sh — Quick setup for CognOS research environment

set -e

echo "=================================================="
echo "CognOS Research Environment Setup"
echo "=================================================="

# Navigate to research directory
cd "$(dirname "$0")"

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (with --copies for mounted filesystems)..."
    python3 -m venv --copies .venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt -q

# Check Ollama
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama found"
    echo ""
    echo "Available models:"
    ollama ls
    echo ""
    
    # Check for qwen2.5:7b
    if ollama ls | grep -q "qwen2.5:7b"; then
        echo "✓ qwen2.5:7b available (recommended)"
    else
        echo "⚠️  qwen2.5:7b not found"
        echo "   Run: ollama pull qwen2.5:7b"
    fi
else
    echo "⚠️  Ollama not found"
    echo "   Install: https://ollama.ai"
fi

# Test LLM backend
echo ""
echo "Testing LLM backend..."
python llm_backend.py

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. python run_exp_001_divergence.py"
echo ""
