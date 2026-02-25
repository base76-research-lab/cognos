#!/bin/bash
# setup-cognos-standalone.sh
# Builds CognOS as a standalone Python package ready for PyPI + GitHub

set -e

DEST="/media/bjorn/iic/cognos-standalone"

echo "🔨 Setting up CognOS standalone product..."
echo ""

# 1. Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "$DEST"/{cognos,tests,examples,docs}
cd "$DEST"

# 2. Copy core modules (already have them)
echo "📋 Copying core modules..."
cp /media/bjorn/iic/cognos/core/confidence.py cognos/
cp /media/bjorn/iic/cognos/core/divergence_semantics.py cognos/

# 3. Create __init__.py
echo "📝 Creating cognos/__init__.py..."
cat > cognos/__init__.py << 'EOF'
"""
CognOS — Epistemic Integrity Layer for Agentic AI

The operating system for decision-aware AI systems.
"""

__version__ = "0.2.0"
__author__ = "Björn Wikström"
__license__ = "MIT"

from .confidence import compute_confidence
from .divergence_semantics import synthesize_reason, frame_transform, convergence_check

__all__ = [
    "compute_confidence",
    "synthesize_reason",
    "frame_transform",
    "convergence_check",
]
EOF

# 4. Create setup.py
echo "📝 Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cognos-ai",
    version="0.2.0",
    author="Björn Wikström",
    author_email="bjorn@homelab.se",
    description="Epistemological integrity layer for agentic AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bjornshomelab/cognos",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
    extras_require={"dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"]},
)
EOF

# 5. Create pyproject.toml
echo "📝 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cognos-ai"
version = "0.2.0"
description = "Epistemological integrity layer for agentic AI systems"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [{name = "Björn Wikström", email = "bjorn@homelab.se"}]
keywords = ["ai", "confidence", "uncertainty", "epistemic", "metacognition"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = ["numpy>=1.20.0"]

[project.optional-dependencies]
dev = ["pytest>=6.0", "black>=21.0", "flake8>=3.9"]

[project.urls]
Homepage = "https://github.com/bjornshomelab/cognos"
Repository = "https://github.com/bjornshomelab/cognos.git"
EOF

# 6. Create LICENSE
echo "📝 Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Björn Wikström

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# 7. Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
EOF

# 8. Create README.md
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# CognOS — Epistemic Integrity Layer for Agentic AI

An open-source operating system for decision-aware AI systems. CognOS combines epistemic and aleatoric uncertainty to help AI agents know when they know — and when they should ask for help.

## What is CognOS?

CognOS is **externalized metacognition** for AI. It answers a fundamental question:

> *When should an AI system act autonomously, and when should it escalate to humans?*

Instead of binary auto/escalate decisions, CognOS provides four nuanced decision types:
- **auto** — high confidence, act autonomously
- **synthesize** — conflicting but coherent perspectives, combine them
- **explore** — noise, gather more information
- **escalate** — too risky, require human judgment

## Core Formula

```
C = p × (1 - Ue - Ua)

where:
  p   = prediction probability [0, 1]
  Ue  = epistemic uncertainty (variance of MC samples)
  Ua  = aleatoric/semantic risk (ambiguity + irreversibility + blast_radius) / 3
  C   = decision confidence [0, 1]
```

## Installation

```bash
pip install cognos-ai
```

## Quick Start

```python
from cognos import compute_confidence

result = compute_confidence(
    prediction=0.85,
    mc_predictions=[0.84, 0.86, 0.85, 0.87],
)

print(result['decision'])  # 'auto', 'synthesize', 'explore', or 'escalate'
print(result['confidence'])  # 0.803
```

## Three Layers

### Layer 1: Confidence Engine
Combines probabilistic and semantic uncertainty into a single confidence score.

### Layer 2: Divergence Semantics
When perspectives conflict, extract *why* they differ.

### Layer 3: Convergence Control
Stop recursion when the system has converged.

## License

MIT License — see LICENSE file

## Citation

```bibtex
@software{wikstrom2026cognos,
  title={CognOS: Epistemic Integrity Layer for Agentic AI},
  author={Wikström, Björn},
  year={2026},
  url={https://github.com/bjornshomelab/cognos},
}
```

See docs/ for research notes and full paper draft.
EOF

# 9. Initialize git
echo "🔧 Initializing git repository..."
git init
git config user.email "bjorn@homelab.se"
git config user.name "Björn Wikström"

# 10. Add and commit
echo "📦 Creating initial commit..."
git add .
git commit -m "Initial commit: CognOS v0.2 — Epistemic integrity layer for agentic AI"

# 11. Verify
echo ""
echo "✅ CognOS standalone product ready!"
echo ""
echo "📋 Directory structure:"
ls -la
echo ""
echo "🔗 To push to GitHub:"
echo "   git remote add origin https://github.com/bjornshomelab/cognos.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📦 To publish to PyPI:"
echo "   pip install build twine"
echo "   python -m build"
echo "   twine upload dist/*"
EOF
