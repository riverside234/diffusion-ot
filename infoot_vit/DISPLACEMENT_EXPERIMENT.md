# Next experiment: preserve source detail while transporting features

Use `compare_displacement.py` for the next **fit, then mapping test**. It fits
once and compares the existing estimator with one new candidate using that same
complete fit. This is an implemented, numerically tested experiment, **not yet a
visually validated blur fix**. The real feature tensors and dog checkpoint are
not available in this local checkout.

The first lab attempt finished **15,963 of 16,000 pairs** and stopped with
33 outer line-search stalls and four inner-certificate failures. Mapping and
generation did not run. The [compact analysis](../results/infoot_vit/displacement_next_analysis.json)
records the evidence; the original result folder was deleted at the user's
request. Resolve these numerical failures before repeating a full fit with
unchanged settings. The command below defines the comparison protocol, not a
validated fix for those remaining failures or the blurred images.

The completed `blur_spread_20261011T174849Z` experiment showed that native dog
tokens remain clear at the mapped variance, whereas increasing mapped variance
does not resolve blur. The next change therefore preserves source information
before it is lost through conditional averaging. No gain/temperature sweep is
included.

## Run once on the lab machine

From the repository root, using the existing CUDA environment and this updated
code, run:

```bash
python infoot_vit/compare_displacement.py \
  --fit-config infoot_vit/configs/grouped_partial.yaml \
  --source-bank data/infoot_vit/cat_train_4000 \
  --target-bank data/infoot_vit/dog_train_4000 \
  --query-bank data/infoot_vit/cat_val16 \
  --reference-run results/infoot_vit/grouped_partial_20261011T073017Z_39614958_20261011T164734Z_771361 \
  --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt \
  --output-dir results/infoot_vit/displacement_check \
  --device cuda --batch-size 4 --threads 4
```

`displacement_check` must not exist; use another new name for a repeat. The
4000-image bank paths are the **original bank identities**. Fitting still selects
the exact same **2000 training images per domain, seed 42**. Do not substitute
newly extracted banks or use validation images for fitting.

The runner checks the reviewed checkpoint hash
`361fccb4827ab67a67b89232c860c458fb99809733082dbe0da8a5b2b29f6d73`
before fitting. If `latest.pt` has changed, point `--checkpoint` at the preserved
matching file. It reads the saved **raw weights, 40 Euler steps, guidance 1.5,
seed 20260903**, ordered 16 query IDs and per-query noise seeds from the reviewed
report. There are no sampler override flags in this comparison.

If a matching fit has already completed, replace `--fit-config` and the two
training-bank flags with `--mapping <complete-fit-directory>`. This avoids a
redundant fit; the projection change does not itself require refitting. The
existing strict repair tool can supply such a completed fit. An unrepaired
15,575/16,000 snapshot is rejected; this runner has no failed-pair bypass.

## What changes

| Component | Next experiment |
|---|---|
| Image fit h / reg / lam | 0.36 / 0.06 / 0.070, unchanged |
| Partial fit h / reg / lam | 0.35 / 0.04 / 0.035, unchanged |
| Transported mass / threshold | 0.80 / 0.05, unchanged |
| Image / patch projection h | 0.20 / 0.20, unchanged |
| Selected pairs | Same top-8 policy per training source; require 16,000 successes |
| Numerical solver | Existing v6 stall refinement, strict certificates, float64 computation / float32 plans |
| Decoder | Same frozen checkpoint; no training or co-training |
| Baseline values | Existing KDE conditional mean of target tokens (`barycentric`) |
| Candidate values | Original query token plus a conditional mean of fitted source-to-target displacements |

The candidate is `q + E[y_j - x_i | q]`, using the original partial-plan edges.
Both source query kernels and the original target-density correction contribute
to the edge weights. It does **not** choose a top-1 dog, sharpen the generated
pixels, inflate variance, or use nearest target tokens as a replacement.

The exact definition, its limits and synthetic checks are documented in
[`conditional_displacement.py`](infoot_helper/conditional_displacement.py) and
the [experiment rationale](../docs/analysis/infoot_vit_displacement_20261011/README.md).
It is a different projection-value estimator, not a claim of equivalence to the
original InfoOT conditional projection. The original fitting objective and
partial constraints remain intact.

The identity/constant-displacement cases preserve unseen query residuals exactly
up to floating-point precision. This offers a specific mechanism for retaining
detail and expression. It does not prove that SigLIP cat-to-dog semantic changes
are locally constant. Wrong matches, retained cat traits and decoder mismatch
remain possible; these must be judged in the actual images.

## Read the result

The result directory contains **only two files** for Git push/pull and analysis:

- `experiment_report.json`: fitting reports and failed-pair histories, both
  mapping manifests and statistics, generation controls, hashes and review fields.
- `comparison_grid.png`: all seven rows and all queries at original resolution;
  zoom in to inspect individual columns. No duplicate row images or panels.

Fitted plans, per-pair logs, mapped tensor caches and `conditioning_variants.pt`
with exact shared noise stay under **`outputs/infoot_vit/`**, already ignored by
Git. Their location is recorded as `work_directory` in the report. An optional
`--work-dir <new-directory-outside-results>` selects a different local storage
location. Keep that directory on the lab machine for resume or further mapping;
the two analysis files do not replace a loadable fitted artifact.

The comparison has these seven rows:

1. Original cat.
2. Top-1 routed dog RGB reference, **not paired ground truth**.
3. Native dog features with every token.
4. Native dog features with the mapped mask.
5. Unmapped cat features through the dog decoder with that same mask.
6. Existing mapped baseline.
7. Conditional-displacement candidate.

The five decoder conditions receive clones of the **same instantiated starting
noise**. The runner verifies identical baseline/candidate masks, confidence,
image routes, rejection and bandwidths before decoding. Native references come
from the current completed fit, so they may change relative to the earlier
incomplete fit; the new two mapped rows are the controlled estimator comparison.

Review every column for clear fur/eyes/face boundaries, coherent dog anatomy,
source head turn, eyes/mouth expression, coat color and markings, and increased
reference copying. Columns 2, 6, 12, 14 and 15 of the reviewed set are especially
useful expression/pose/copying cases. A sharper cat, generic reference dog,
sharper facial artifacts, or higher token variance is not a pass.

The report's `quality_review` starts **unreviewed** with an empty verdict for each query.
`status: complete` in the execution report means the fit, mapping and decoding
finished; it does not certify image quality. Retain the candidate only if the
matched images improve both clarity and translation. The default mapper remains
the previous estimator until that evidence exists.

If the process fails, the single report records the error and available fitting
diagnostics, including full failed-pair histories. Completed work remains in
the ignored work directory.
After a fit has completed, use `--mapping` and a new output directory to retry
mapping/decoding. An interrupted fit can be resumed with the existing fitting
CLI and its exact saved configuration; do not restart the full fit merely
because a later decoder load failed.
