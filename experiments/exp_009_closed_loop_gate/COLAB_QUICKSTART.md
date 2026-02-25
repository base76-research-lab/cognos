# exp_009 i Google Colab — Quickstart

Mål: köra `exp_009_closed_loop_gate` direkt i Colab med minimalt friktion.

## 1) Setup (cell 1)

```python
import os
from pathlib import Path

REPO = "https://github.com/base76-research-lab/cognos.git"
ROOT = Path("/content/cognos")

if not ROOT.exists():
    !git clone {REPO}

%cd /content/cognos/experiments/exp_009_closed_loop_gate
!pip -q install -r requirements.txt
```

Om repot tillfälligt är privat, använd PAT-flöde istället:

```python
import getpass
token = getpass.getpass("GitHub PAT: ")
!git clone https://x-access-token:{token}@github.com/base76-research-lab/cognos.git /content/cognos
```

## 2) Smoke test (cell 2, 2–5 min)

```python
!python run.py --config configs/mnist_colab_smoke.yaml --mode all
```

Output skrivs till:
- `/content/results/exp_009_smoke/results_all.json`

## 3) Full run (cell 3)

```python
!python run.py --config configs/mnist_colab_full.yaml --mode all
```

Output skrivs till:
- `/content/results/exp_009_full/results_all.json`

## 4) Snabb summering i notebook (cell 4)

```python
import json
from pathlib import Path

path = Path('/content/results/exp_009_full/results_all.json')
if not path.exists():
    path = Path('/content/results/exp_009_smoke/results_all.json')

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

## 5) (Valfritt) Spara resultat till Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/CognOS/exp_009"
!cp -r /content/results/exp_009_* "/content/drive/MyDrive/CognOS/exp_009/"
```

## Tips

- Börja alltid med smoke-test innan full 5-seed körning.
- Om du får timeout: kör `--mode closed_loop` separat först.
- För OOD-variant (P2/P3 extension), skapa ny config med annat dataset och separat `results_dir`.
