# CognOS migration — 2026-02-25

Syfte: centralisera allt CognOS-relaterat till en canonical repo.

Canonical repo:
- `base76-research-lab/cognos`

Genomfört:
- Basen från tidigare cognos-standalone ligger nu i canonical repo.
- Legacy-innehåll från tidigare `workspace/01_RESEARCH/cognos` är kopierat till:
  - legacy/workspace_01_RESEARCH_cognos
- Setup-script kopierat till:
  - scripts/setup-cognos-standalone.sh
- Gamla paths ersatta med redirect-README.

Nästa steg:
1. git add .
2. git commit -m "Consolidate CognOS into canonical repo"
3. git push -u origin main
