# Displacement experiment: preserve source detail while transporting features

The completed **v7 run was reviewed on 12 October 2026**. All 37 formerly failing
pairs passed their numerical replay, and the fresh full fit completed all
16,000 pairs with zero failures. Conditional displacement improves visible
detail in most columns, but many faces remain cat-like or develop mixed anatomy;
column 12 has a substantial facial-quality regression. **Retain v7 fitting and
keep displacement experimental.** The default estimator remains unchanged.

The [detailed local review](../docs/analysis/infoot_vit_displacement_v7_20261012/README.md)
records numerical checks, matched controls and observations for all 16 columns.
The next test is implemented as `--experiment residual_ablation`. It reuses the
exact completed v7 fit and compares plan-edge target means with and without the
source residual. No new fit or co-training is needed.

## Next experiment: residual removal with UMAP

After syncing this code to the lab machine, run from the repository root:

```bash
python infoot_vit/compare_displacement.py \
  --experiment residual_ablation \
  --mapping outputs/infoot_vit/displacement_v7_4deed6d4/fit/grouped_partial_20261012T014020Z_df15945b \
  --reference-run results/infoot_vit/displacement_v7 \
  --query-bank data/infoot_vit/cat_val16 \
  --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt \
  --output-dir results/infoot_vit/residual_ablation_v8 \
  --device cuda --batch-size 4 --threads 4
```

The fitted directory is the `mapping` recorded in the completed v7 report. Keep
it on the lab machine; the shared report and PNG cannot replace its plan tensors.
The new output directory must not already exist. If `latest.pt` changed, use the
preserved checkpoint matching v7. Missing UMAP dependencies stop the experiment
before mapping; install them only if needed with
`python -m pip install umap-learn matplotlib` (also listed in root requirements).

The primary comparison is:

- Residual retained: `conditional_displacement`, `T1(q) = E_w[y_j] + q - E_w[x_i]`.
- Residual removed: `conditional_edge_mean`, `T0(q) = E_w[y_j]`.

Both use exactly the same original plan edges, target-density correction, query
kernels, routing and confidence gates. This is a single endpoint ablation with
no gain sweep. `T0` remains distinct from `barycentric`, which uses the
KDE-smoothed target conditional mean `E[y_k]`; that blurry baseline is included
as a third mapped control. The default mapper remains barycentric.

The runner checks the completed v7 mapper identity, decoder configurations,
checkpoint, ordered queries, routing, confidence and masks. It also checks the
instantiated noise against v7's recorded hash before sampling. All decoder
conditions receive identical starting noise. Fitting settings and both
projection bandwidths remain unchanged at their reviewed values.

The experiment writes **three files** under the new result directory:

1. `experiment_report.json`: all three mapping reports, fitting metadata,
   generation controls, feature statistics, UMAP settings/versions/coordinates
   and an initially unreviewed quality checklist.
2. `comparison_grid.png`: eight rows in order: original cat, routed dog RGB
   reference, native dog with all tokens, native dog with mapped mask, unmapped
   cat through dog decoder, barycentric, displacement, and residual-removed edge
   mean. The last two rows are the primary comparison.
3. `umap.png`: six panels comparing the three mappings, with image means above
   and individual patches below. Numbers in the image-mean panels match grid
   columns. Cat/dog training anchors and source queries accompany each mapping.

The UMAP uses the existing `infoot_test.py --umap` style: blue source-training
circles, orange target-training circles, dark-blue query circles, green mapped
X markers, gray arrows and the same legend labels. Both plots call the same
panel-drawing helper in `infoot_helper/umap_plot.py`. Each method occupies a
separate column with shared axes per row.

There is one common reducer per UMAP row, fitted on sampled real training-bank
features only; all query variants use `transform` in that common space. Every
method uses the same valid query token positions. Image means are used only
for this display, never for fitting transport or conditioning the decoder.
Settings are fixed: seed 42 for fit/transform, one worker, Euclidean distance,
15 neighbors (capped for tiny test fixtures), minimum distance 0.1, and at most
1,000 training points per domain or query patches per method. Every valid query
image is included in the mean view. No feature normalization is added.
The two UMAP rows are different coordinate systems; visual proximity does not
establish dog semantics, sharpness or translation quality. See the official
[transform](https://umap-learn.readthedocs.io/en/latest/transform.html) and
[reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)
documentation for the display protocol.

UMAP is required in this experiment. A plot failure records a failed execution
and preserves mapped caches under the ignored work directory; it cannot silently
produce a completed result without the requested plot. There is no separate
UMAP JSON, no per-image PNG export and no fit directory under results.

Removing the residual may reduce retained cat traits or facial artifacts, but
may restore blur or lose pose/expression. Inspect all columns and especially
2, 6, 12, 14 and 15. Accept only a coherent, clearer dog translation that also
preserves source correspondence. Neither completion, higher variance nor a
UMAP cluster is a quality pass. The actual experiment still needs to run on the
lab machine; local tests use synthetic features and a stubbed decoder.

## Earlier v7 experiment

The rest of this guide documents the completed v7 comparison. Its fit commands
are historical reproduction commands, not the next experiment. Use fresh output
names if reproducing it, since the v7 reports already exist.

`compare_displacement.py` fits once and compares the existing estimator with one
candidate using that same complete fit. The real feature tensors and dog
checkpoint are not available in this local checkout.

The first lab attempt finished **15,963 of 16,000 pairs** and stopped with
33 outer line-search stalls and four inner-certificate failures. Mapping and
generation did not run. The [compact analysis](../results/infoot_vit/displacement_next_analysis.json)
records the evidence; the original result folder was deleted at the user's
request. The v7 solver now evaluates log-plan stationarity without repeated
subtraction and allows an accepted small damped step to continue after precision
refinement. Convergence still requires the original **undamped** residual;
objective, mass, capacity and convergence tolerances are unchanged. Synthetic
checks and the 37 real GPU replay cases pass. The subsequent complete v7 fit
also has no failed or unfinished pairs.

The completed `blur_spread_20261011T174849Z` experiment showed that native dog
tokens remain clear at the mapped variance, whereas increasing mapped variance
does not resolve blur. This experiment therefore preserves source information
before it is lost through conditional averaging. No gain/temperature sweep is
included.

## Reproduce the completed comparison

From the repository root on the lab machine, using the existing CUDA environment,
first pull the committed updated code. Then run this **single Bash block**. The
first command retries exactly the 37 recorded failed training pairs, without a
router refit, per-pair files or checkpoints. It saves one JSON report and exits
nonzero if any pair fails. `&&` starts the full experiment only after all 37 pass.

```bash
python infoot_vit/check_failed_pairs.py \
  --source-bank data/infoot_vit/cat_train_4000 \
  --target-bank data/infoot_vit/dog_train_4000 \
  --output results/infoot_vit/remaining_pairs_v7.json \
  --device cuda --threads 4 \
&& python infoot_vit/compare_displacement.py \
  --fit-config infoot_vit/configs/grouped_partial.yaml \
  --source-bank data/infoot_vit/cat_train_4000 \
  --target-bank data/infoot_vit/dog_train_4000 \
  --query-bank data/infoot_vit/cat_val16 \
  --reference-run results/infoot_vit/grouped_partial_20261011T073017Z_39614958_20261011T164734Z_771361 \
  --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt \
  --output-dir results/infoot_vit/displacement_v7 \
  --device cuda --batch-size 4 --threads 4
```

`remaining_pairs_v7.json` and `displacement_v7` must not exist; use new names for
a repeat. Passing this targeted check is not proof that all 16,000 pairs will
converge; the full runner independently requires a complete fit before mapping.
If the check fails, return only its JSON report for analysis and do not start
the full fit. No feature/checkpoint tensors are needed to read that report.

The 4000-image bank paths are the **original bank identities**. Fitting still selects
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
| Numerical solver | v7 stable stationarity calculation and certified damped progress, strict certificates, float64 computation / float32 plans |
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

Together with `remaining_pairs_v7.json`, this workflow produces **three analysis
files total**. The deleted `displacement_next` directory is not recreated.

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
