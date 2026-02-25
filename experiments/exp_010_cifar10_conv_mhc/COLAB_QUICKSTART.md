# exp_010 i Google Colab — Quickstart

Mål: köra CIFAR10 closed-loop-testet (`exp_010_cifar10_conv_mhc`) direkt i Colab.

## 1) Setup (cell 1)

```python
from pathlib import Path

REPO = "https://github.com/base76-research-lab/cognos.git"
ROOT = Path("/content/cognos")

if not ROOT.exists():
    !git clone {REPO}

%cd /content/cognos/experiments/exp_010_cifar10_conv_mhc
!pip -q install -r requirements.txt
```

## 2) Smoke test (cell 2)

```python
!python run.py --config configs/cifar10_colab_smoke.yaml --mode all
```

Output:
- `/content/results/exp_010_smoke/results_all.json`

## 3) Full run (cell 3)

```python
!python run.py --config configs/cifar10_colab_full.yaml --mode all
```

Output:
- `/content/results/exp_010_full/results_all.json`

## 4) Snabb summering (cell 4)

```python
import json
from pathlib import Path

path = Path('/content/results/exp_010_full/results_all.json')
if not path.exists():
    path = Path('/content/results/exp_010_smoke/results_all.json')

data = json.loads(path.read_text())
print(f"Loaded: {path}")
for row in data:
    seed = row.get('seed')
    two = row.get('two_layer', {})
    ol = two.get('open_loop', {})
    cl = two.get('closed_loop', {})
    go = row.get('gate_only', {})
    print(f"seed={seed} | open cw={ol.get('cw_rate')} | closed cw={cl.get('cw_rate')} | gate_only cw={go.get('cw_rate')}")
```

## 5) (Valfritt) Spara till Drive

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/CognOS/exp_010"
!cp -r /content/results/exp_010_* "/content/drive/MyDrive/CognOS/exp_010/"
```
