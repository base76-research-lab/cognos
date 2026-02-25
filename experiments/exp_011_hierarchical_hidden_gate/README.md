# exp_011 — Hierarchical hidden-gate ablation

## Syfte

Testa om tvåstegsgating med hidden gate (och optional residual analysis) kan slå gate-only under samma coverage-budget.

## Modes

- `open_loop`
- `closed_loop`
- `gate_only`
- `two_stage_no_residual`
- `two_stage_with_residual`
- `all`

## Körning

```bash
python run.py --config configs/cifar10.yaml --mode all
```

## Colab

- Guide: `COLAB_QUICKSTART.md`
- Notebook: `exp_011_colab.ipynb`

## Beslutskriterier

- Gate_only robust: CW minskar i medel och i minst 4/5 seeds.
- Closed_loop potentiell: CW minskar + rimlig acc/coverage-normalisering.
- Training mismatch: gate_only förbättrar CW medan closed_loop försämrar CW stabilt.

## Extra loggar

- CW bland gate-pass vs gate-drop
- Confident-correct rate
- `cw_pass`, `cw_drop`, `acc_pass`, `acc_drop`

## Tau-beteende

- `tau_quantile` beräknas globalt per seed och variant på eval-fördelningen (inte per batch/per epoch).
- Gatebeslut använder denna kvantiltröskel via `gate_mask(entropy, tau_quantile)`.
