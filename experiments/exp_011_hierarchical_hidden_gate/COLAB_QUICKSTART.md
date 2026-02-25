# exp_011 i Google Colab — Quickstart

Mål: testa hierarkisk hidden-gate ablation mot open/closed/gate-only baselines.

## 1) Setup (cell 1)

```python
from pathlib import Path

REPO = "https://github.com/base76-research-lab/cognos.git"
ROOT = Path("/content/cognos")

if not ROOT.exists():
    !git clone {REPO}
else:
    %cd /content/cognos
    !git pull --ff-only

%cd /content/cognos/experiments/exp_011_hierarchical_hidden_gate
!ls -la configs
!pip -q install -r requirements.txt
```

## 2) Smoke run

```python
!python run.py --config configs/cifar10_colab_smoke.yaml --mode all
```

Output:
- `/content/results/exp_011_smoke/results_all.json`

## 3) Full run

```python
!python run.py --config configs/cifar10_colab_full.yaml --mode all
```

Output:
- `/content/results/exp_011_full/results_all.json`

## 4) Snabb summering

```python
import json
from pathlib import Path

full_path = Path('/content/results/exp_011_full/results_all.json')
smoke_path = Path('/content/results/exp_011_smoke/results_all.json')

path = full_path if full_path.exists() else smoke_path
payload = json.loads(path.read_text())
print(f"Loaded: {path}")
print("Decision criteria:", payload.get('decision_criteria', {}))

for row in payload.get('results', []):
    seed = row.get('seed')
    ol = row.get('two_layer', {}).get('open_loop', {})
    cl = row.get('two_layer', {}).get('closed_loop', {})
    go = row.get('gate_only', {})
    t1 = row.get('two_stage_no_residual', {})
    t2 = row.get('two_stage_with_residual', {})
    print(f"seed={seed} | open={ol.get('cw_rate')} | closed={cl.get('cw_rate')} | gate={go.get('cw_rate')} | ts_no_res={t1.get('cw_rate')} | ts_res={t2.get('cw_rate')}")
```
