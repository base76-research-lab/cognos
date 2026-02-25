# Public Release Checklist (base76-research-lab/cognos)

## 1) Repository visibility
- Set repository visibility to **Public** in GitHub settings.

## 2) Initial publish
- `git add .`
- `git commit -m "Public bootstrap: CognOS tests, analyses, publications, Colab setup"`
- `git push -u origin main`

## 3) Verify public assets
- Colab quickstart: `experiments/exp_009_closed_loop_gate/COLAB_QUICKSTART.md`
- Test suite: `experiments/test_*.py`
- Analysis scripts: `experiments/analyze_*.py`
- Papers: `publications/papers/*.pdf`

## 4) Post-publish sanity
- Open repo URL in incognito and verify files are downloadable.
- Start one Colab run from quickstart and confirm `results_all.json` generated.

## 5) Optional hardening
- Enable branch protection on `main`.
- Add GitHub Actions for smoke test (`exp_009` smoke config).
