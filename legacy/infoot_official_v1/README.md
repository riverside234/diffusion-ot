# Preserved official-based InfoOT workflow

`snapshot.json` records the source commit and SHA-256 of every preserved file.
The core, training/evaluation modules, CLIs, configs and focused tests were
copied byte for byte before the next revision. This is the project's prior
implementation based on [official InfoOT](https://github.com/chingyaoc/InfoOT),
including its previous project extensions; it is not an unmodified upstream
checkout. Shared Stage 1A models, data loaders and integrations use the parent
repository. `PROJECT_README.md` preserves the earlier documentation.
The four referenced Stage 1A configs are preserved too. `legacy/.gitattributes`
disables newline conversion for the snapshot so hashes survive Linux checkout.

Run from the project root in a fresh process:

```bash
python3 scripts/run_legacy_infoot.py check
python3 scripts/run_legacy_infoot.py test -q

python3 scripts/run_legacy_infoot.py train \
  --config configs/stage1b_infoot/plain_sit_b2.yaml

python3 scripts/run_legacy_infoot.py evaluate \
  --alignment-config configs/stage1b_infoot/plain_sit_b2.yaml \
  --eval-config configs/stage1b_eval/quick_sit_b2.yaml
```

Relative config arguments resolve inside this snapshot, using the frozen CLI's
original behavior. Configured data/checkpoint/output paths still resolve under
their `project_root` (the Linux project directory). Existing output locations
are preserved; choose a copied config with a different output directory for a
separate reproduction run. Fused and projection-trained configs are preserved
as well. The launcher verifies all hashes and module paths before running, so
these commands cannot silently use the revised active InfoOT modules.
