# SigLIP2 feature-map InfoOT: fit and test

This is an **offline mapping experiment** for the completed PDAE v2/v2-L models.
It extracts frozen SigLIP2 features, fits transports on training images, and maps
held-out images without refitting. It does not launch co-training or change the
DiT projector, cross-attention layers, positional embeddings, losses, or learned
checkpoint keys.

## What changed

| File | Responsibility |
|---|---|
| `bank/bank_cat.py`, `bank/bank_dog.py`, `bank/build_bank.py` | Original RGB → the existing frozen SigLIP2 encoder → unpooled `[N,196,768]` banks. No CNN, VAE-encoded input, pooling, or additional feature normalization. |
| `infoot_fit.py` | Validate resources; fit/resume; save every accepted outer iteration's latest raw plan in a **new fit directory**. |
| `infoot_test.py` | Project a disjoint validation/test bank; optionally run fixed-checkpoint diffusion inference. |
| `infoot_compare.py` | Compare saved mappers on the same queries and fit populations. |
| `infoot_helper/feature_bank.py` | Sharded tensor banks, stable IDs, representation identity, fingerprints. |
| `infoot_helper/conditional.py` | Fixed-training-bandwidth balanced scoring and partial confidence projection. |
| `infoot_helper/partial.py` | Transported-marginal MI autodiff, POT partial subproblems, feasibility and full-objective line search. |
| `infoot_helper/fit_mapping.py`, `mapping.py` | Mapping artifacts, pair inventory, resume, streamed projection and mandatory masks. |
| `infoot_helper/evaluate_mapping.py` | Existing PDAE sampling/decoding and original-image loading. |
| `infoot_helper/run_logging.py` | Per-attempt console capture, structured events, timing, failure tracebacks and run status. |
| `../src/diffusion_ot/evaluation/stage1a_eval.py` | Backward-compatible optional `condition_padding_mask` passthrough, including sampler chunking. |
| `../tests/test_infoot_vit_mapping.py` | Numerical, cache, CLI, resume, rejection and consumer regressions. |

The copied legacy helpers remain available. `infoot_helper/infoot.py` was not
overwritten: its repaired `conditional_score()` already uses source training
distances for bandwidth scaling. The new `BalancedModel.project()` is the tested new-sample entry
point. The separate `../infoot/` CNN experiments remain unchanged. At inspection,
`infoot_vit/` still contained copies of the pooled CNN bank/fit/test scripts;
there was no existing SigLIP whole-map, grouped or partial mapper to reuse.

## Start with feature banks

Run from the repository root in the existing lab environment. For partial mode,
POT needs its log-domain partial solver. The verified local version is
`0.9.7.post1`; install explicitly if the capability check says it is unavailable:

```bash
python -m pip install -r infoot_vit/requirements.txt

python infoot_vit/bank/bank_cat.py --split train
python infoot_vit/bank/bank_dog.py --split train
python infoot_vit/bank/bank_cat.py --split val
python infoot_vit/bank/bank_dog.py --split val
```

Defaults use `configs/stage1a_pdae_v2_l/{cat,dog}.yaml` to find the pinned local
SigLIP2 snapshot and data manifests. Both domains and queries must have the same
encoder snapshot, feature layer and RGB preprocessing. A domain's trained DiT
checkpoint is **not** needed for bank extraction: SigLIP2 is frozen and shared.
No model is downloaded by the bank, fit, or projection command. The existing
SigLIP2 download script remains the installation path.

The extractor reuses `OriginalImageLoader` (the same center crop/resize as PDAE)
and `FrozenSiglipPatchEncoder` (the saved PIL processor). It uses no augmentation.
Every bank records train/val/test split, original sample records, image IDs,
14×14 row-major patch order, all-valid masks, selected post-LN feature layer,
preprocessing, encoder file hashes and float32 storage. Existing output banks
are never silently overwritten. Choose another `--output-dir` for a new bank.

### Size matters: exact full support

Run `--dry-run` before fitting. A whole-map plan has `N_cat*N_dog` entries. A
complete partial pair bank has `N_cat*N_dog*196*196` float64 plan entries, plus
iteration checkpoints, kernels and reports. For example, **32×32 images already
need about 0.293 GiB just for the final pair plans**; 4,000×4,000 need about
4,580 GiB. Global patch baselines also require patch-count-squared dense kernels.
The mandatory `latest.pt` checkpoints add a second copy of the plans. The disk
guard now uses `required_plan_storage_gib`, including those copies and shared
pair kernels; `plan_storage_gib` still describes only the primary final plans.
For 32×32 image pairs, the two plan copies alone need about **0.586 GiB**.
Keep extra free space for metadata/logs and atomic writes. Estimates are not
measured peak process memory or a disk-space reservation.

No command silently drops images/patches, pools tokens, applies PCA, or replaces
exact routing with nearest neighbors. Resource limits in YAML fail explicitly;
raise them deliberately if the machine and disk can support the exact fit.

For a **separate, explicitly smaller pilot**, build named banks, then point all
comparison configurations to those same banks:

```bash
python infoot_vit/bank/bank_cat.py --split train --max-images 16 --output-dir data/infoot_vit/cat_train_16
python infoot_vit/bank/bank_dog.py --split train --max-images 16 --output-dir data/infoot_vit/dog_train_16
```

`--max-images` selects the first N sorted stable image IDs and records that
selection. It does not claim to be a representative random subset. All 196
patches of each selected image remain in the fit.

## Fit modes and saved plans

| Mode | What is fitted/projected |
|---|---|
| `patch_global` | Balanced FusedInfoOT on individual unpooled patches, with saved fixed bandwidth state. |
| `whole_map` | A balanced FusedInfoOT router on `[N,196*768]`; one image-weight vector mixes all patch positions. First baseline. |
| `grouped_patch` | The image router plus a global patch model; image weights are shared, patch scores normalize **within** each target image using the global KDE. |
| `grouped_partial` | Balanced image router plus a full bank of independently fitted fixed-mass partial patch plans, one per training image pair. Requested target experiment. |

```bash
python infoot_vit/infoot_fit.py --config infoot_vit/configs/whole_map.yaml --dry-run
python infoot_vit/infoot_fit.py --config infoot_vit/configs/whole_map.yaml

python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --dry-run
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml
```

To use pilot banks, add these overrides to **each** fit command:

```bash
--source-bank data/infoot_vit/cat_train_16 --target-bank data/infoot_vit/dog_train_16
```

Each invocation creates a new directory such as
`outputs/infoot_vit/grouped_partial_<UTC timestamp>_<unique ID>/`. Nothing is
saved over the old `data/infoot_test/cat_to_dog_plan.pt`.

```text
manifest.json                    # Configuration, supports, IDs, versions, hashes, status
fit_report.json                  # Convergence, residuals, scales, resource estimates
plans/image.pt                   # Balanced image plan + fitted scale state + trace
plans/image/latest.pt            # Updated while the image fit runs
plans/image/iterations.jsonl
plans/patch.pt                   # Global patch plan, when applicable
plans/pair_kernels.pt            # Shared fit kernels/scales and fit-only support thresholds
plans/pairs/<indices_short-ID-hash>.pt  # Gamma, a,b,r,c,s/report, full IDs and order
plans/pairs/<indices_short-ID-hash>/latest.pt
plans/pairs/<indices_short-ID-hash>/iterations.jsonl
pairs.jsonl                     # Complete accepted pair inventory and file hashes
pair_diagnostics.jsonl          # Per-pair objective, caps, mass and raw retention summaries
```

Interrupted partial fits resume **completed accepted pairs**; an interrupted
individual pair is refitted from its feasible initialization. The image/global
models are reused once complete. Intermediate `latest.pt` is diagnostic state,
not permission to use a nonconverged plan at inference.

```bash
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --resume outputs/infoot_vit/<exact-fit-directory>
```

Use the same overrides as the initial fit. Resume checks data, support order,
configuration, numerical source hashes and runtime fingerprints. Different
settings require a new fit; failures/stalled solves remain explicitly failed.
No marginal renormalization hides a failed solve.

Optional YAML `image_model: outputs/infoot_vit/<completed-whole-map-fit>` and
`patch_model: outputs/infoot_vit/<completed-patch-global-fit>` reuse saved models
with identical fit banks and solver settings. The manifest records the reused
artifact IDs. Partial plans are never shared across different keep-mass settings.

### Initial hyperparameters and bandwidths

The example configs use raw features, Euclidean distances, `h=0.4`, `lam=0.1`,
`reg=0.05`, and **explicit training-mean cost normalization**. These are
engineering starting settings, not validated best settings. Cost normalization
changes the relative MI/entropy strengths compared with the old unscaled CNN
recipe; it never changes projected target feature values. `--h/--reg/--lam`
override the balanced solver; patch-pair settings are under `partial.solver`.

Kernel widths follow the local convention:
`h * sqrt(mean(training_pairwise_distances**2)/2)`. Projection starts with the
same width (`bandwidth_multiplier: 1.0`) and never estimates a query-batch scale.
An explicit multiplier is part of the fit configuration/artifact identity; it
rebuilds projection kernels and calibrates partial support thresholds on fit
data. There is no query-time threshold calibration or hidden bandwidth sweep.

All current offline numerical work uses CPU float64; saved RGB banks stay
float32 and mapped features return to the query dtype/device. GPU solver and
lower-precision paths have not been validated or enabled.

## Saved logs for tuning and debugging

Logging is automatic for actual fits, mapping tests and comparisons; no extra
flag or shell redirection is required. A successful run, a failed run and each
resume attempt get a separate `logs/<UTC timestamp>_<unique ID>/` inside their
output directory. `--dry-run` remains read-only and prints to the terminal.

```text
logs/<attempt>/
  run.json                  # running/completed/failed/interrupted; arguments, times, environment
  stdout.log                # console output, also shown live
  stderr.log                # warnings/errors printed to stderr
  events.jsonl              # flushed progress, solver iterations, pairs/queries, final status
  traceback.txt             # full exception stack on failure/interruption
  error.json                # exception type/message on failure/interruption
  requested_config.txt      # fit input, including before preflight validation
  inspection.json           # validated fit settings, bank identities, versions, resource estimate
  fit_report.json           # successful fit summary (also at the fit directory root)
  queries.jsonl             # mapping diagnostics, one completed query per line
  mapping_report.json       # successful mapping summary
```

Files specific to fitting or mapping appear only for that operation. Per-plan
`plans/.../iterations.jsonl` and `latest.pt` remain available. Iteration JSONL
records include the attempt ID and elapsed time, so an interrupted/refitted pair
can be distinguished from earlier attempts. `events.jsonl` also records which
model/pair was active if failure occurs before an accepted first step. Inner
solver warnings captured by POT wrappers are in the iteration's `inner.warnings`.

Mapping saves each query's diagnostics **before** enforcing the all-invalid
policy. A rejected-query failure therefore retains confidence/rejection evidence
without creating a successful mapped artifact. The test CLI also captures failed
bank/model loading and optional diffusion generation in the same attempt log.
Mapping success followed by generation failure leaves the valid mapped cache
and marks the enclosing test run failed.

Use these signals together:

| Concern | Saved evidence |
|---|---|
| Solver convergence | Objective components, inner residual/warnings, plan L1 change, pair status and cap/mass residuals. Partial steps also record accepted L1 change, backtracks and objective change; a tiny accepted step is not labeled convergence. |
| Narrow-kernel numerics | Balanced `gradient_log_floor_entries` counts ratio entries protected by the existing log floor. Nonzero counts signal a numerically sharp kernel/plan; they are not evidence of semantic quality. |
| Overly smooth/query-independent mapping | Per-query image weights, normalized entropy/effective targets, within-image effective patches; compare mapped corresponding-patch variance across images with source/target/query statistics. |
| Excessive rejection | Confidence quantiles, valid-token fraction, all-invalid IDs, separate OT-rejected and support-invalid retained mass. |
| Norm/spread contraction | `mapping_report.json` includes valid-token norms/variance, corresponding-patch variance across images and image-mean variance for queries, mapped outputs and fixed supports. |
| Reproducibility | Fit/mapping manifests contain ordered IDs, frozen encoder identity, numerical source hashes and saved settings; generation reports pin checkpoint hash, seeds, sampler and guidance. |

Keep the whole fit/test output directories when sharing a failed run. For a
small tuning report, share `fit_report.json`, `manifest.json`, and the latest
attempt's logs; include translation/confidence grids and `generation_report.json`
when generation ran. Feature metrics are not independent visual-quality scores.

Ctrl-C saves an interrupted status/traceback. A hard kill or power loss can leave
`run.json` at `running`; the last flushed event then identifies progress. Resume
preserves accepted pairs and can recover an incomplete final inventory write,
backing up the truncated bytes in that attempt's logs. Interior corruption is an
error. Each interrupted individual pair still restarts from its initialization.

### Review fixes (2026-10-10)

- Balanced fitting now safely differentiates its existing floored KDE objective
  when narrow kernels/low entropy create zero densities; the previous direct
  `0/0` MI derivative could abort a feasible solve. Healthy-case derivatives
  agree with the legacy formula. The copied legacy standalone solver is not the
  supported fit entry point; use `infoot_fit.py`/`BalancedModel.fit()`.
- Partial-solver stall records reach the final iteration checkpoint/log.
- Partial effective-patch counts now weight `exp(entropy)` by matched routing
  mass, fixing spuriously inflated diversity readings.
- Invalid chunk/top-k/seed settings and nonfinite resource limits fail early;
  mapped conditioning validates finite values and confidence shapes/ranges.
- Generation checks query/result ID order before loading PDAE and uses stable-ID
  tie breaking for its reference row.
- Required plan storage includes mandatory latest checkpoints; failed report
  writing no longer double-counts elapsed fit time.

These numerical-source changes intentionally invalidate **resume** fingerprints
of earlier incomplete fits. Start a new fit for them. Completed artifacts remain
loadable; mapping caches record current projection dependency hashes. No loss
weights, bandwidths or architecture were changed by this review.

## Project and optionally generate

```bash
# Numerical mapping only: no DiT is loaded and no OT is fitted.
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<fit-directory> --query-bank data/infoot_vit/cat_val --dry-run
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<fit-directory> --query-bank data/infoot_vit/cat_val

# Fixed dog PDAE checkpoint: original cat / routed top-1 dog reference / translated dog.
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<fit-directory> --query-bank data/infoot_vit/cat_val --generate --train-config configs/stage1a_pdae_v2_l/dog.yaml --eval-config configs/stage1a_eval/pdae_v2_l.yaml --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt --weights ema --steps 50 --solver euler --guidance 1.5
```

Use numbered checkpoints for reproducible comparisons. Noise is seeded per
stable query ID; default count is 16. `--chunk-size` affects resource use, not
query randomness or projection. The patch-global control has no image router,
so its grid contains source/translation rows only. The routed dog is a reference,
**not paired ground truth**.

Each test creates a fresh `results/infoot_vit/.../` directory (or the new
`--output-dir` supplied by the caller), containing:

- `mapped.pt`: features `[B,P,D]`, confidence `[B,P]`, boolean validity `[B,P]`,
  IDs and original records, loaded atomically with `load_mapped()`.
- `manifest.json`: mapper/query/cache identities, projection settings, ordering,
  image weights/concentration, feature norms/variance, mapping time, support and
  rejection diagnostics. Partial outputs always include confidence and masks.
- With `--generate`: `translation_grid.png`, `confidence_grid.png`, and
  `generation_report.json` recording the checkpoint hash, protocol, per-image
  noise seeds, sampler, guidance and confidence policy.

For dog→cat, swap the train banks in fitting, use `dog_val` queries, and choose
the cat PDAE training config/checkpoint during generation.

## Partial objective and rejection contract

For each training image pair, `Gamma>=0`, `Gamma.sum(1)<=a`,
`Gamma.sum(0)<=b`, and `Gamma.sum()=s`. All patches enter the problem. Entropic
plans can have positive mass in **every row** even at `s<1`.

The objective is `<C,Gamma> - lam*J(Gamma) + reg*sum(Gamma*(log(Gamma)-1))`.
`J=M*I_hat` uses `Gamma/M` and **its transported row and column marginals** in
the KDE densities. PyTorch float64 autodiff includes the mass and both marginal
derivatives. It does not backpropagate into SigLIP, DiT, or through a solver.
POT solves each exact fixed-mass entropy subproblem in log space. Convex
backtracking checks the full objective; stalls are distinguished from
convergence. `lam=0` directly solves entropic partial OT. `s=1` uses the verified
balanced local-pair subproblem; it is not the global patch baseline.

The image conditional scorer is decomposed into shared pair responsibilities
`Theta[i,j]`. Summing over source images reproduces the original target-image
weights `alpha[j]`. Each query patch uses the **same** pair routing budget.
Within a pair, conditional weights use the **original** target proposal
`b/(K_Y@b)`, whereas the fitting MI uses transported marginals.

Confidence estimates retained source mass:
`g_raw=(K_Q @ Gamma.sum(1))/(K_Q @ a)`. Fit-only leave-one-out log-density gating
handles out-of-support queries; this is a heuristic, not a probability of
anatomical correctness. `support_calibration: fixed` with a finite
`support_log_threshold` is also supported. `disabled` is an explicit diagnostic
ablation, used for the pure `s=1` confidence-one regression.

The output is `sum(Theta*g*candidate)/sum(Theta*g)`, with unnormalized confidence
`sum(Theta*g)`. True zero retained mass gives a zero placeholder and invalid
mask. Numerical underflow is separately diagnosed; tiny conditional scores
retry in log space. A support-valid confidence underflow is an error, not a
fabricated match. Matching plus rejection recovers each target-image routing
budget. Matched-only image proportions may vary by patch.

**Raw fitted retention**, **smoothed query confidence**, **top-k omitted routing
mass**, **support-invalid mass**, and **thresholded mask coverage** are separate
diagnostics. The average confidence of unseen queries need not equal `s`.

`MappingResult.conditioning()` translates mapper `True=valid` to the existing
PDAE `True=padding` convention. Confidence is not multiplied into features and
then erased by LayerNorm. All-invalid queries fail with IDs by default. Setting
`all_invalid_policy: bypass` explicitly uses the existing tested zero-image-
attention-residual behavior; no new gate or fabricated valid token is added.

## Controlled comparisons

Include `grouped_partial` with `s=1` to isolate rejection from the change to
local pair plans. A suggested mass sweep is **1.0, 0.9, 0.8, 0.7**; none is
assumed optimal. Each command creates a different artifact:

```bash
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --keep-mass 1.0
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --keep-mass 0.9
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --keep-mass 0.8
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --keep-mass 0.7

python infoot_vit/infoot_compare.py --mappings outputs/infoot_vit/<whole-map-fit> outputs/infoot_vit/<s1-pair-fit> outputs/infoot_vit/<s08-pair-fit> --query-bank data/infoot_vit/cat_val
```

The comparison verifies identical fit-bank fingerprints and query IDs. Include
the `patch_global.yaml` and `grouped_patch.yaml` fits when resources permit.
Keep diffusion checkpoint, sampler, guidance and seed identical for image
comparisons. Check source color/pattern, target realism, pose, and whether
rejection removes useful rare attributes. Same-feature-space similarity is not
independent evidence of better images.

Optional `selection: argmax|sample` and `top_k_images` act once per entire image.
Tie breaks use stable target IDs; sampling uses a hash of seed and query ID.
Full support (`mean`, `top_k_images: null`) is the reference. These controls do
not avoid fitting the complete pair bank. Sparse banks, variable grids, learned
rejection, and joint hierarchical optimization are not implemented.

## Verification and provenance

The local legacy solver was inspected at repository commit
`53b00c55dff75765e465a3fa2d1e0f4885f39fb0`; its file SHA-256 before this work was
`ed649404707657d4f53796eda0197b76fd70a95287d3abe245853ced1a12f075`.
The repo's upstream audit reference pins InfoOT to
`352efd202f5b475dc170a8d08a99049689d5ee1a`. Actual local source hashes and POT/
PyTorch versions are saved with every fit. The balanced checked loop reuses
local `FusedInfoOT` geometry and `migrad`, adds explicit convergence checks and
optional recorded scalar cost normalization, and preserves the original class.

CPU synthetic tests cover: lossless layout, correct fit sample counts, unequal
domains, direct conditional ratios and target-density correction, batch/order/
chunk invariance, ID joins, no inference solves, gradients including changing
marginals, `s=1`, `lam=0`, negative costs, entropy convention, monotonic accepted
partial objectives, mass/cap feasibility, dense-versus-streamed aggregation,
confidence rescaling, compact-kernel zero retention, support rejection,
serialization/tampering, interrupted resume, CLI commands, and the existing
PDAE mask path without changing parameter counts or checkpoint keys.

The existing `tests/test_local_infoot_pipeline.py` covers legacy fixed-bandwidth
scoring; `tests/test_pdae_v2.py` covers pretrained equivalence and attention masks.
One pre-existing legacy test expects `numIter=50` although the unchanged
`infoot/infoot_fit.py` uses 100. It is separately reported, not fixed by changing
the old experiment.

Latest local review regression: **130 passed, 1 deselected**, using PyTorch
`2.14.0+cpu` and POT `0.9.7.post1`. The single warning is from the existing
legacy encoder-alignment test's Sinkhorn iteration budget. Command (using the
local test environment's Python):

```bash
python -m pytest tests/test_infoot_vit_mapping.py tests/test_infoot_vit_logging.py tests/test_infoot_vit_mapping_review.py tests/test_infoot_vit_solver_review.py tests/test_pdae_v2.py tests/test_stage1a_rgb_eval.py tests/test_local_infoot_pipeline.py -k "not test_bank_fit_script_passes_raw_features_and_cli_lam" --basetemp tmp/vrfinal -p no:cacheprovider -q
```

The original mapping test module contributes 33 cases; review/logging tests add
25 cases. The one deselection is the pre-existing legacy iteration-default
mismatch described above. Syntax compilation and
`git diff --check` also pass. Pair filenames use indices plus a short ID hash
to avoid unnecessarily long Windows paths; full IDs and fingerprints remain
in the saved inventory and plan state.

No real-image bank extraction, full AFHQ fit, GPU solver validation or diffusion
generation on lab checkpoints has been run during this implementation. The
generation test uses mocked checkpoint/data loading; the sampler mask test uses
the existing tiny real PDAE branch. Numerical correctness is not a claim of
translation quality or of global optimality.

### References and attribution

- [Official InfoOT solver at the audited revision](https://github.com/chingyaoc/InfoOT/blob/352efd202f5b475dc170a8d08a99049689d5ee1a/infoot.py): balanced FusedInfoOT and the kernel/scoring reference; preserve the local repairs.
- [InfoOT paper, sections 4–5](https://arxiv.org/html/2210.03164v2): fused transport and conditional projection.
- [POT partial API](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.entropic_partial_wasserstein) and [implementation](https://pythonot.github.io/_modules/ot/partial/partial_solvers.html): fixed-mass cap constraints and log-domain entropy solver.

The grouped routing, transported-population KDE extension, pair bank and
confidence policy implement the supplied project plan. They are not claimed as
official InfoOT features or empirically superior methods. Grouping alone gives
no anatomical/one-to-one correspondence or real-image-manifold guarantee.
