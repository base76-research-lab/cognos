#!/bin/bash
# setup_home_env.sh — Setup research environment in home directory
# Avoids symlink issues on mounted filesystems

set -e

echo "=================================================="
echo "CognOS Research Environment Setup (Home Directory)"
echo "=================================================="

# Work from home directory
WORKSPACE_DIR="$HOME/tests/cognos-research"
SOURCE_DIR="/media/bjorn/iic/cognos-standalone"

echo "Workspace: $WORKSPACE_DIR"
echo "Source: $SOURCE_DIR"
echo ""

# Create workspace
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Create venv (no --copies needed in home dir)
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements from source
echo "Installing requirements..."
pip install -r "$SOURCE_DIR/research/requirements.txt" -q

# Add source to PYTHONPATH
echo ""
echo "Setting up paths..."
echo "export PYTHONPATH=\"$SOURCE_DIR:\$PYTHONPATH\"" > venv/bin/activate_cognos
cat >> venv/bin/activate_cognos << 'EOF'

# Activate venv
source ~/tests/cognos-research/venv/bin/activate

# Helpful aliases
alias run-exp1='python "$SOURCE_DIR/research/run_exp_001_divergence.py"'
alias cd-research='cd "$SOURCE_DIR/research"'

echo "✓ CognOS environment activated"
echo "  Source: $SOURCE_DIR"
echo "  Workspace: ~/tests/cognos-research"
echo ""
echo "Commands:"
echo "  run-exp1           - Run experiment 001"
echo "  cd-research        - Navigate to research dir"
EOF

# Create symlinks to experiment configs (optional)
echo "Creating config symlinks..."
ln -sf "$SOURCE_DIR/research/exp_001_divergence" exp_001_divergence 2>/dev/null || true
ln -sf "$SOURCE_DIR/research/exp_002_epistemic_gain" exp_002_epistemic_gain 2>/dev/null || true
ln -sf "$SOURCE_DIR/research/exp_003_illposed" exp_003_illposed 2>/dev/null || true

# Create run scripts
cat > run_exp_001.sh << 'EOF'
#!/bin/bash
source ~/tests/cognos-research/venv/bin/activate
export PYTHONPATH="/media/bjorn/iic/cognos-standalone:$PYTHONPATH"
cd /media/bjorn/iic/cognos-standalone/research
python run_exp_001_divergence.py
EOF
chmod +x run_exp_001.sh

# Check Ollama
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama found"
    echo ""
    echo "Available models:"
    ollama ls
    echo ""
    
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

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To activate environment:"
echo "  source ~/tests/cognos-research/venv/bin/activate_cognos"
echo ""
echo "Or directly run experiments:"
echo "  cd ~/tests/cognos-research"
echo "  ./run_exp_001.sh"
echo ""
