# Displacement experiment: preserve source detail while transporting features

The completed **v7 run was reviewed on 12 October 2026**. All 37 formerly failing
pairs passed their numerical replay, and the fresh full fit completed all
16,000 pairs with zero failures. Conditional displacement improves visible
detail in most columns, but many faces remain cat-like or develop mixed anatomy;
column 12 has a substantial facial-quality regression. **Retain v7 fitting and
keep displacement experimental.** The default estimator remains unchanged.

The [detailed local review](../docs/analysis/infoot_vit_displacement_v7_20261012/README.md)
records numerical checks, matched controls and observations for all 16 columns.
The completed **v8 residual ablation** returned almost the same blurry images as
barycentric (mean RGB difference 0.235 on the 0-255 scale). Variance after the
decoder projector was 0.4285, versus 0.4265 for barycentric and 0.7996 with the
residual retained. Neither endpoint passes the combined quality goal.

## V9 review: sharpness recovered, translation still fails

Reviewed all 16 columns of `results/infoot_vit/averaging_ablation_v9`, including
native-resolution tiles, mask controls and the UMAP. This is a useful diagnostic
success, **not an accepted translation method**. Dominant-dog argmax restores
clear fur and dog anatomy in many images, but outputs largely follow the routed
reference dog's appearance, pose and background. The source yawn (2), sideways
pose (6), closed eyes (14) and raised paw (15) are not preserved.

The run used the exact completed v7 fit (16,000 successful pairs), checkpoint,
query order, routing and initial noise. V9's original-mask edge-mean conditioning
statistics and generated pixels equal v8 exactly. Reported code hashes and PNG
hashes verified; per-dog decomposition reconstructs the original map to maximum
absolute error `7.82e-14`, confidence to `2.22e-15`, and the variance identity to
`5.11e-15`. The saved lab tensors are not in this local checkout; this review
checks their recorded identities, the report, code and rendered results.

| Shared-mask condition | Input spatial variance | After decoder projector |
|---|---:|---:|
| All-dog mean | 0.5443 | 0.4129 |
| Dominant-dog mean | 0.7858 | 0.5054 |
| Dominant-dog patch argmax | 1.3511 | 0.8779 |
| Native dog control | 1.4593 | 0.8902 |

Argmax reaches **98.6%** of native projector variance, versus **56.8%** for the
single-dog mean. Across common valid tokens, 57.8% of conditional target-token
dispersion lies within dogs. In column 12, where the original image route gives
one dog 94.97%, the within-dog fraction is 94.35%; the mean still looks soft and
argmax is visibly clearer. These fractions describe feature dispersion, not
fractions of image blur. The dominant conditional distribution averages an
effective 27.25 target patches per valid query token.

The common mask retains 2,728/3,136 tokens (86.99%), removing another 281 from
the original 3,009 valid tokens. The original-mask row must remain a control:
mask changes alone visibly alter some expressions, and the source-token/common-
mask control already loses the wide-open yawn and raised paw. Argmax duplicates
55.39% of its selected patch occurrences across queries. This changes the
decoder's attention weighting; it does not establish a spatial rearrangement.

**Architecture finding:** `TokenConditionedSiT` adds positions to latent queries,
but applies only a shared per-token projector to the image conditions. Its
cross-attention treats those conditions as an unordered set. Permuting tokens
and their masks leaves its output unchanged (verified with nonzero attention
weights). Therefore the recorded 4.6–6.0-patch neighbor jumps are a description
of the selected indices, not evidence of a decoder-visible spatial scrambling.
A one-to-one permutation of a complete reference dog's tokens would still feed
the same set. Spatial sorting or assignment alone cannot recover source pose.

| Column | Visual finding for dominant-dog argmax |
|---|---|
| 1 | Clearer fur, eyes and nose; strongly follows the Chihuahua reference. |
| 2 | Clearer dog face; loses the source's wide-open yawn and squint. |
| 3 | Clear long-haired dog; adopts the reference's tan markings and face. |
| 4 | Sharp Schnauzer-like face, dominated by reference appearance. |
| 5 | Copies the tan long-haired dog and grass instead of the tabby appearance. |
| 6 | Frontal retriever; loses the source's sideways head pose. |
| 7 | More detail, but enlarged round eyes and changed head orientation. |
| 8 | Clear white dog, close to the routed reference rather than a new correspondence. |
| 9 | Reference dog's blue eyes and coat replace source appearance. |
| 10 | Dark shaggy reference dog replaces the white cat; mouth opens. |
| 11 | Reference shepherd appearance; source narrow-eyed expression is lost. |
| 12 | Clearer white retriever; source gray markings and feline head pose are lost. |
| 13 | Copies the lying bulldog pose rather than the upright source. |
| 14 | Clearer tan dog; closed eyes are not retained. |
| 15 | Clearer small dog; raised paw is absent. |
| 16 | Sharp pit bull appearance, blue eyes and reference head tilt. |

The UMAP has the existing test style, two reducers fitted only to real training
anchors, matched query records and finite coordinates. Selecting one dog spreads
image means toward individual reference regions; argmax and one-dog mean often
have similar image coordinates despite different sharpness. This is not a
quality score; column 13 is also away from the main dog cluster.

## Next experiment: preserve source structure during generation

Run the new `edit_source.py` experiment on the lab machine after syncing code:

```bash
python infoot_vit/edit_source.py \
  --mapping outputs/infoot_vit/displacement_v7_4deed6d4/fit/grouped_partial_20261012T014020Z_df15945b \
  --reference-run results/infoot_vit/averaging_ablation_v9 \
  --query-bank data/infoot_vit/cat_val16 \
  --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt \
  --output-dir results/infoot_vit/source_edit_v10 \
  --device cuda --batch-size 8 --threads 4
```

No refitting or remapping is performed. Preserve the v9 work directory
`outputs/infoot_vit/averaging_ablation_v9_ca9e6f84`, especially
`conditioning_variants.pt`. The default reads its recorded path; if relocated,
add `--reference-work-dir /path/to/that/directory`. The runner checks its hash,
all condition identities, common mask, source tokens, native references,
checkpoint, configs and exact instantiated baseline noise. It does not download
models or regenerate a missing cache.

All three primary methods use **exactly the same v9 argmax dog tokens and mask**:

1. Original noise-to-image baseline: 40 midpoint Euler steps, CFG 1.5, original noise.
2. Source-initialized denoising: encode the source RGB with the VAE posterior
   mode and training scale, start at `0.5 * source_latent + 0.5 * original_noise`,
   then use the final 20 steps of the same grid.
3. Source/target velocity editing: start an edit state at the source latent.
   At each of those 20 times, evaluate the source- and target-conditioned fields
   with paired noisy states, then update by their velocity difference. Source
   and target fields use the same frozen dog decoder and CFG 1.5. Each step has
   fresh deterministic per-query noise; no inversion, gradient optimization,
   raw source-feature residual, finishing-stage sweep or per-image tuning.

The third method adapts [FlowEdit Algorithm 1](https://arxiv.org/abs/2412.08629)
to image-token conditions and this model's noise-to-data time convention:
`s=(1-t)*noise+t*source`, `target_state=s+(edit-source)`, and
`edit += dt*(v_target-v_source)`. This is an experimental adaptation; the paper's
results do not establish its quality for this decoder. The second method uses
the source-noising principle of [SDEdit](https://arxiv.org/abs/2108.01073), adapted
to the existing deterministic linear-flow sampler. Neither changes model weights.

The dog decoder's source-cat conditioning is outside its training domain. The
VAE latent gives a direct spatial source signal, but this still needs a real
run: it may retain too much cat anatomy or continue copying the reference.
The preset start time is a first controlled test, not a tuned optimum.

Exactly **three result files** are produced: `experiment_report.json`,
`comparison_grid.png`, and `umap.png`. Eight grid rows are source RGB, source VAE
reconstruction, routed dog RGB, native/common-mask dog control, v9 argmax/noise
baseline, source-conditioned denoising control, argmax/source denoising, and
argmax/velocity editing. Raw images, latents and generated feature tensors stay
in the ignored work directory. Failure retains those diagnostic caches.

The UMAP reuses the same plotting/reducer code, colors, markers, arrows and
column numbers. This time it plots **generated RGB re-encoded with the frozen
SigLIP encoder**, using all output patches, since plotting the three identical
input conditions would hide the experiment's effect. Training banks alone fit
each reducer. Conditioning masks are unchanged; full output-image UMAP masks
are a separate display choice. RGB change/reference distances are diagnostics,
not a quality ranking. Review all 16 outputs for coherent dogs, visible detail,
source pose/expression and reference copying before accepting either method.

## Completed v9: separate averaging within and across dogs

Use `--experiment averaging_ablation` with the completed **v8** result directory
as the reference. It reuses the exact v7 fit and the same plans, bandwidths,
support calibration, decoder weights and starting noise. After syncing the code,
run from the repository root on the lab machine:

```bash
python infoot_vit/compare_displacement.py \
  --experiment averaging_ablation \
  --mapping outputs/infoot_vit/displacement_v7_4deed6d4/fit/grouped_partial_20261012T014020Z_df15945b \
  --reference-run results/infoot_vit/residual_ablation_v8 \
  --query-bank data/infoot_vit/cat_val16 \
  --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt \
  --output-dir results/infoot_vit/averaging_ablation_v9 \
  --device cuda --batch-size 8 --threads 4
```

The output directory must be new. Keep the original fitted plans and checkpoint
on the lab machine. No refit or bank override is permitted. UMAP uses the same
dependencies and shared plotting code as v8.

The three primary readouts are:

1. **All dogs, edge mean:** the original v8 residual-removed map, on the common
   comparison mask.
2. **Dominant dog, patch mean:** aggregate the same fitted edge weights within
   one dog and decode that dog's conditional target-patch mean. The dog is the
   highest original post-pruning image route, with stable-ID tie breaking. It
   stays fixed across all query patches and matches the native reference.
3. **Dominant dog, patch argmax:** for that same dog and distribution, select the
   highest-weight target token at each query position. Ties use the first saved
   row-major patch index. No source residual, nearest-neighbor retrieval, gain
   or temperature sweep is added.

Readouts 1 versus 2 diagnose averaging across dogs. Readouts 2 versus 3 test
replacing within-dog patch averaging by a hard selection. Argmax can duplicate
patches, remove content from the conditioning set and copy the reference; it does not preserve
soft-plan marginals and is not a new default transport estimator. Source-image
uncertainty is still marginalized inside each dog's distribution.

The observer collects
`A[j,q,k] = sum_i theta[i,j] * g[i,j,q] * w[i,j,q,k]`, where `w` is the original
plan-edge target-patch distribution including the target-density correction.
The sums over `k` give dog masses. Normalizing within dogs gives `beta[j,q,k]`;
normalizing dog masses gives `pi[j,q]`. The mixture
`sum_j pi[j,q] * sum_k beta[j,q,k] * y[j,k]` must reconstruct the unchanged v8
edge mean and confidence within float64 tolerance.

For each query position the report checks the law of total variance:
`total target-token dispersion = within-dog patch dispersion + between-dog mean dispersion`.
These are conditional squared Euclidean dispersions per feature dimension.
They are separate from spatial variance within a conditioning map; their
fractions are not percentages of image blur. Input, LayerNorm and learned-
projector spatial variances are also recorded for every decoded condition.

The three primary readouts share the original v8 valid mask intersected with
positive dominant-dog mass and dominant-dog conditional confidence at least the
original 0.05 threshold. Conditional confidence is that dog's retained mass
divided by its original image route. A separate original-mask mixture row
isolates the effect of dropping additional tokens. Coverage and excluded
positions are recorded. If a query has no common tokens, the runner records a
failed diagnostic and stops before decoding, without filling unsupported tokens.

Exactly **three files** are saved in the result directory:

- `experiment_report.json`: identity checks, original mapping report, variance
  decomposition, masks, selected patch indices, concentration, duplication and
  adjacency diagnostics, decoder statistics, noise controls and UMAP coordinates.
- `comparison_grid.png`: nine rows: source cat; routed dog RGB; native dog with
  all tokens; native dog with common mask; cat tokens through dog decoder with
  common mask; all-dog edge mean with original mask; all-dog edge mean with
  common mask; dominant-dog mean; dominant-dog patch argmax.
- `umap.png`: the final three readouts in shared image-mean and patch spaces,
  fitted only on real training banks and using the existing test UMAP style.

Tensors, selected-dog probabilities and exact noise stay in the ignored
`outputs/infoot_vit/` work directory. The CPU accumulator holds one query's
target-patch masses (586.2 MiB for 2,000 dogs and 196 tokens); moments are computed
in blocks of 16 targets on the mapping device. All-target feature maps and
all-query mass tensors are never stored. Mapping caches are released before
the decoder is loaded.

Review especially column 12, already routed 94.97% to one dog in v8. A sharp but
incoherent argmax output only diagnoses averaging. If the dominant-dog mean stays
blurry, cross-dog mixing alone is insufficient. Judge anatomy, source pose,
expression/color and reference copying together with patch duplication. Execution
completion and UMAP proximity remain separate from visual acceptance. The actual
experiment needs the lab run; local tests use synthetic features and a stub decoder.

## Completed v8: residual removal with UMAP

The completed v8 experiment used this command (use a new output name to repeat):

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
UMAP cluster is a quality pass. The completed v8 run confirmed the blur/detail
tradeoff described above.

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
