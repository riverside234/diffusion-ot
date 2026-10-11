# SigLIP2 feature-map InfoOT: fit and test

This is an **offline mapping experiment** for the completed PDAE v2/v2-L models.
It extracts frozen SigLIP2 features, fits transports on training images, and maps
held-out images without refitting. It does not launch co-training or change the
DiT projector, cross-attention layers, positional embeddings, losses, or learned
checkpoint keys.

**Separate low-rank experiments:** see [`lowrank/README.md`](lowrank/README.md).
`configs/grouped_patch_lowrank.yaml` uses balanced global patch factors at
transport/kernel ranks 256/1024. `configs/grouped_partial_lowrank.yaml` uses capacity-constrained
pair factors at rank 64/64, transported mass 0.8, and eight saved target pairs
per training source. Both use sampled InfoOT, 2,000 train images/domain
(seed 42), float64 fitting and float32 factors, with checked per-experiment
storage budgets. The balanced 256/1024 recipe estimates 4.35 GB under a 5 GB
guard; the partial experiment retains its 4 GB guard. Use
`infoot_fit_lowrank.py`; dense modes remain available.
The balanced recipe retains normalized positive kernel features. Both fit
bandwidths are now `h=0.7`; `projection.bandwidth_multiplier=0.2/0.7` requests
router and patch projection `h=0.2`, with a separate mandatory projection-kernel
accuracy gate. This requested setting has not passed real-bank acceptance.
It rejects excessive Gaussian/density approximation errors before transport fitting. Run
`infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --kernel-check-only`
to save diagnostics and a bounded exact-Gaussian comparison first. This does
not change the partial experiment's kernel method; see the low-rank README
for thresholds, comparison methods and continuation commands.

All current `infoot_vit/configs/*.yaml` select **`device: cuda`**. The device
applies to routers, kernels, transport solvers and mapping. Use `--device cpu`
explicitly on CPU systems, or `--device cuda:1` for another GPU. Unavailable
requested CUDA raises an error. `infoot_test.py`/`infoot_compare.py` default to
the artifact's saved device and accept the same override; older artifacts
without device metadata retain CPU behavior. Storage verification and I/O use
CPU tensors; the numerical pipeline uses the selected device.

## What changed

| File | Responsibility |
|---|---|
| `bank/bank_cat.py`, `bank/bank_dog.py`, `bank/build_bank.py` | Original RGB → the existing frozen SigLIP2 encoder → unpooled `[N,196,768]` banks. No CNN, VAE-encoded input, pooling, or additional feature normalization. |
| `infoot_fit.py` | Validate resources; fit/resume; save solver iteration diagnostics and the active checkpoint in a **new fit directory**. |
| `infoot_fit_lowrank.py`, `lowrank/` | Balanced global and partial pair InfoOT factors; shared objective, kernels, checkpointing and projection. |
| `infoot_helper/device.py` | Explicit CPU/CUDA selection and artifact-to-device transfers. |
| `infoot_test.py` | Project a disjoint validation/test bank; optionally run fixed-checkpoint diffusion inference. |
| `infoot_compare.py` | Compare saved mappers on the same queries and fit populations. |
| `infoot_helper/feature_bank.py` | Sharded tensor banks, stable IDs, representation identity, fingerprints. |
| `infoot_helper/conditional.py` | Fixed-training-bandwidth balanced scoring and partial confidence projection. |
| `infoot_helper/partial.py` | Transported-marginal MI autodiff, POT partial subproblems, feasibility and full-objective line search. |
| `infoot_helper/partial_batch.py`, `pair_batch_fit.py` | Exact dense pair batches, independent stopping/failures, per-pair registration and checkpoint I/O. |
| `benchmark_partial_batch.py` | Matched serial/batched solver timing and CUDA peak-memory measurement on deterministic synthetic pairs. |
| `infoot_helper/fit_mapping.py`, `mapping.py` | Mapping artifacts, pair inventory, resume, streamed projection and mandatory masks. |
| `infoot_helper/sampling.py`, `pair_selection.py`, `storage.py` | Deterministic training subset, persisted router pair selection and checked float32 disk storage for `grouped_partial`. |
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

### Whole-map projection tuning without refitting

The reviewed 4,000-image `whole_map` fit converged at iteration 638 with mean
fitted-row top-1 weight 74.8%. Its held-out projection at h=0.21 was much more
diffuse: median effective targets 1,895 and median top-1 weight 1.72%, with
repetitive, blurred generation. See the [review and matched comparison commands](../docs/analysis/infoot_vit_whole_map_20261010/README.md).

`configs/whole_map.yaml` keeps fit h=0.35, reg=0.06, lam=0.075, sets the outer
budget to 1,200, and requests projection h=0.10 through multiplier `0.10/0.35`.
This projection setting still needs visual validation. Reuse an existing fit
with `infoot_test.py --projection-bandwidth 0.10`; editing the YAML alone does
not change a saved mapping. The override is an absolute h, uses the saved
training distance scales and target-density correction, and never fits a plan.
It is recorded in each fresh mapping output and generation report.

Compare 0.21/0.15/0.10 with the same validation bank, checkpoint and noise seed.
Whole-map `--top-k-images 0` keeps all targets; `--top-k-images 4` keeps and
renormalizes four weights, logging discarded mass. `--top-k-images 1` supplies
one real target's complete feature map: a useful decoder/retrieval control,
not evidence of source-preserving translation. The grid's top-1 reference row
alone does not describe the all-target averaged condition used by default.
The overrides also work for `grouped_patch`, `grouped_partial`,
`grouped_patch_lowrank` and `grouped_partial_lowrank`. `patch_global` accepts
bandwidth changes but has no image router for `--top-k-images`.

For grouped modes, **`--projection-bandwidth` is the absolute image-router h**:
the multiplier is `requested_h / saved_router_fit_h`, and that same multiplier
scales the patch fit bandwidth. Example: router fit h=0.35 and patch fit h=0.45,
with `--projection-bandwidth 0.20`, gives router projection h=0.20 and patch
projection h≈0.25714 when no independent patch bandwidth is set. Dense
`grouped_partial` now also accepts **`--patch-projection-bandwidth`**, or
`projection.patch_bandwidth` in YAML. That absolute patch bandwidth takes
precedence over the shared multiplier. Old artifacts without this field keep
their original behavior. Each query logs both actual bandwidths.

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<partial-fit> --query-bank data/infoot_vit/cat_val --projection-bandwidth 0.20 --top-k-images 4 --generate --steps 40 --guidance 2.0
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<lowrank-fit> --query-bank data/infoot_vit/cat_val --projection-bandwidth 0.20 --top-k-images 4 --generate
```

Partial mapping truncates image routing, then intersects routes with the
**unchanged saved pair inventory** and renormalizes retained mass. This override
does not change `fit_pair_top_k`: a top-8 fit cannot recover unfitted pairs by
requesting more targets. Excluded-pair mass, top-k discarded mass, and partial-OT
rejection remain separate diagnostics. No usable saved route is an explicit
error. Missing/corrupt selected files remain errors too.

Dense partial mapping rebuilds exact projection kernels and recalibrates support
thresholds using training patches only. Low-rank mapping re-evaluates the saved
random bases, means and scales at the requested bandwidth, updates both support
factors and target-density correction, and recalibrates partial support thresholds.
It never fits a new kernel basis or transport plan. A changed bandwidth triggers
a sampled exact-Gaussian comparison saved as `logs/<run>/projection_kernel_quality.json`
in the **test output**, including failed acceptance. Balanced low-rank non-fit
bandwidths still require acceptance; partial low-rank retains its configured
kernel error policy (currently `warn`). Top-k-only overrides reuse the original
kernel path. The fitted directory and its fingerprints remain unchanged; new
mapped artifacts record actual projection settings. Passing `--dry-run` checks
metadata only and does not certify numerical kernel accuracy.

Whole-map logs now include actual projection h, source-neighborhood concentration,
raw and selected target concentration, raw top-k mass, and mapped-to-target
feature-spread ratios. Read these alongside generated images; balanced masks
are all valid by construction and do not measure semantic correctness.

### Storage-efficient `grouped_partial`

Only `grouped_partial` uses the sampling, sparse pair inventory, float32 disk
plans/kernels and completed-checkpoint cleanup described here. `whole_map`,
`patch_global` and `grouped_patch` retain their full-support, float64 behavior.

The supplied `configs/grouped_partial.yaml` selects **2,000 training images per
domain without replacement, seed 42**, fits the 2,000×2,000 image router, and
sets **`fit_pair_top_k: 8`**. Sampling runs on sorted stable IDs with Python's
seeded `random.sample`; selected IDs are sorted for storage. The manifest saves
the algorithm, seed, population count and ordered sample IDs. Fewer than 2,000
available images is an error; there is no replacement or silent size reduction.
All 196 patches of each selected image are retained.

Existing full training banks work directly: fitting selects an in-memory subset
and preserves the original bank identity. Held-out checks include **all original
bank IDs**, including training rows omitted from the fit. To reduce feature-bank
extraction/storage too, explicitly build separate sampled train banks:

```bash
python infoot_vit/bank/bank_cat.py --split train --sample-images 2000 --sample-seed 42 --output-dir data/infoot_vit/cat_train_2000_s42
python infoot_vit/bank/bank_dog.py --split train --sample-images 2000 --sample-seed 42 --output-dir data/infoot_vit/dog_train_2000_s42

python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --source-bank data/infoot_vit/cat_train_2000_s42 --target-bank data/infoot_vit/dog_train_2000_s42 --dry-run
```

Bank sampling is opt-in and train-only; the extractor's default remains all
images. Train/validation/test manifests are never mixed. Replace `--dry-run`
with an actual fit only after reviewing its resource estimate.

For each fitted source image, the saved router's conditional probabilities at
the configured image projection bandwidth rank target images, with stable target-ID tie breaking. The
eight selected pairs and probabilities are saved in `pair_selection.json`.
Selection uses the serialized router promoted to float64, so new and resumed
fits use the same router. Only these pairs are solved. Set `fit_pair_top_k: null`
or pass `--full-pairs` to retain every pair; K at least the target count is also
equivalent to full-pair mode.

At 2,000 images/domain, K=8 gives **16,000 pairs**. Float32 partial-plan arrays
need about **2.29 GiB**, versus **572.44 GiB** for all 4,000,000 pairs. Shared
patch kernels add about **0.57 GiB** and the image plan about **0.015 GiB**.
`required_plan_storage_gib` also allows for active latest/atomic-write copies
and a temporary shared-kernel copy. It excludes container metadata, IDs and
logs and retained failed checkpoints: reserve extra disk space. Float64 fitting is still substantial work;
these estimates are neither measured peak memory nor a disk reservation.

### Exact GPU batches for the top-8 experiment

The earlier `results/vit_infoot_top8/test2` run passed the image router in 141
iterations, but all 9,797 recorded pairs exhausted the inner 10,000-step budget.
Passing mass/capacity checks alone did not establish an optimal partial plan.
The batched solver now groups each capacity projection with fixed mass, using
exact log-domain dual block updates. The current recipe incorporates the later
generated-image review below:

| Stage / YAML section | `h` | `reg` | `lam` | Outer budget |
|---|---:|---:|---:|---:|
| Image router: `solver` | 0.36 | **0.06** | 0.070 | 1,200 |
| Partial patch pairs: `partial.solver` | **0.35** | **0.05** | 0.025 | 600 |

At fixed lam=.070/reg=.06, h=.35 gave 85.42% mean top-8 retained mass but
98.69% normalized top-1 probability. The h=.375 trial lost too much coverage
(25.47% mean, 1.81% median). The h=.36 follow-up converged in 564 iterations:
**69.21% mean / 73.88% median retained mass**, with **94.47% normalized top-1**.
This improves coverage over .375, but still leaves 30.79% discarded mass on
average and little contribution from the other retained targets.

The matched reg=.0575 trial converged in 258 iterations and recovered mean
retained mass to **80.88%**, but normalized top-1 rose to **98.40%** (effective
retained targets 1.091). The other seven targets receive only 1.60% of normalized
weight on average. The coverage gain did not produce meaningful multi-target
routing on these training-source probes. This does not prove reference copying
or establish held-out image quality.

The active recipe therefore **restores reg=.06**, keeping h=.36/lam=.070 as a
working baseline for full fitting and matched held-out image evaluation. It
still has concentrated routing and is not a validated optimum. Do not continue
lowering entropy or raising MI solely to improve retained mass. The supplied
`--tune` runs do not measure patch blur, confidence masks or generated images.
Patch settings, mass, confidence, top-8 and .20/.20 projection stay fixed.
Run without `--tune` to save a mapping; `--reg .0575` reproduces the sharper
comparison settings if needed. See [the entropy-trial result and decision](../docs/analysis/infoot_vit_router_reg0575/README.md),
[the earlier h=.36 review](../docs/analysis/infoot_vit_router_h036/README.md)
and [the bandwidth comparison](../docs/analysis/infoot_vit_router_h0375/README.md).
Use a fresh fit for changed settings. To reproduce an older h=.35 recipe,
also restore `projection.bandwidth_multiplier` to `.20/.35`; a `--h` override
alone changes the effective projection bandwidth as well as fitting h.

The active recipe uses `partial.solver.inner_acceleration: newton`: after 100
log-domain block updates, safeguarded dual Newton steps accelerate unfinished
members every 10 updates. PyTorch linear solves use chunks of at most 64 pairs.
Convergence requires relative primal change, KKT residual and relative duality
gap all at most `inner_tolerance=1e-10`, plus the original mass/capacity checks.
The objective and tolerances are unchanged. `inner_acceleration: none` keeps
the prior batched path; the serial POT path uses its original updates.
The earlier test2 router was overly diffuse: mean effective targets 1,943/2,000
and mean top-8 retained probability only 0.445%. The current recipe uses narrower
image kernels and less image entropy; the later fitting-log review below confirms
sharper routing. Patch MI pressure was reduced separately.
`partial.keep_mass: 0.80` and confidence `threshold: 0.05` are **coverage-first
starting values, not visually validated optima**. No successful patch mapping
was produced by test2. Mass is not a patch count; threshold filters smoothed
query confidence and does not fix solver convergence or top-8 discarded mass.
See [the test2 analysis](../docs/analysis/vit_infoot_top8_test2/README.md), and
[the earlier router failure](../docs/analysis/vit_infoot_top8_300/README.md).

The top-8 dense partial recipe uses `projection.bandwidth_multiplier: 0.20/0.36`
(the YAML stores the numeric value) and `projection.patch_bandwidth: 0.20`:
projection `h` is **0.20 for both stages**, while fit `h` is 0.36/0.35.
**New pair selection uses the same image projection bandwidth
as mapping** (previously it incorrectly used the broader fitting bandwidth).
Selection records its bandwidth and retained/discarded routing mass in
`pair_selection.json` and `pair_selection_report.json`. This setting is stored in new mapping artifacts; editing
the YAML does not change existing fits or their resume fingerprints. With
`--generate`, `infoot_test.py` defaults to 40 Euler steps and guidance 2.0 for
`grouped_partial`. Explicit `--steps`/`--guidance` overrides remain supported;
other modes keep defaults of 50 steps and guidance 1.5. Resolved sampling settings
are printed by `--dry-run` and recorded in generation logs/reports.

Start a **new** fit with the revised recipe; changing settings/source invalidates
resume fingerprints. Existing complete artifacts remain loadable.

```bash
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --source-bank data/infoot_vit/cat_train_4000 --target-bank data/infoot_vit/dog_train_4000
```

CLI `--h/--reg/--lam/--max-outer-steps` affect only `solver`. The independent
`--partial-h/--partial-reg/--partial-lam/--partial-max-steps` affect only
`partial.solver`. Use `--keep-mass` for a fresh matched mass ablation.

#### Generated-image review: revise fitting, retain broad projection

The `grouped_partial_20261010T200111Z_f15deac3_20261010T235605Z_66fa3e`
test used 15,987 successful pairs out of 16,000. Its images remain blurred:
projection averages an effective 102 target patches per token, and mapped token
variance is 20.3% of the dog bank's. Even the two >99.5% top-1 image routes
retain broad patch mixtures. Failed routes lose only 0.23% mass on average.
See the [analysis, reproducible statistics and comparison commands](../docs/analysis/infoot_vit_grouped_partial_20261010/README.md).

The [added fitting-log review](../docs/analysis/infoot_vit_grouped_partial_20261010/fitting_review.md)
confirms that the image router converged in 107 steps with 2.49 effective targets
and 92.3% mean top-1 probability. All 13 patch failures had log residuals just
above `1e-10` but tiny duality gaps/constraint errors. Solver v3 adds the
budget-boundary certificate above; it records `relative_plan_delta_l1`,
`kkt_error` and `convergence_reason` alongside the original log residual.
Fitting-code fingerprints changed: use a fresh fit directory.
That budget-only revision proved insufficient in `results/infoot_vit/v2`:
7,168 terminal pairs failed their first outer update after **20,000** inner
steps. The earlier analytic control omitted production mean-cost scaling and
understated the difficulty. Solver v4 adds the acceleration above; a correctly
scaled 196-patch control now converges in 261–271 updates on CPU. This is
numerical evidence, not a replay of the lab tensors or an image-quality result.
See the [v2 failure analysis and fix](../docs/analysis/infoot_vit_partial_v2/README.md).

The active `pair_failure_abort_batches: 1` stops after an entirely failed batch,
preserving all completed pairs, failure records and checkpoints. Null disables
the guard. Reports separate unattempted pairs and interrupted in-flight pairs.
Before another full fit, replay the saved failing input on the lab GPU:

```bash
python infoot_vit/replay_partial_pair.py outputs/infoot_vit/grouped_partial_20261011T011800Z_011068f8/logs/20261011T011800Z_b6ec10be/first_failed_pair.pt --device cuda --output outputs/infoot_vit/v2_pair_replay.json
```

This records objective/constraint diagnostics, elapsed time and peak allocated
GPU memory without changing the original run. Exit code 2 means unsuccessful;
only a converged replay supports proceeding to a fresh full fit. The new code
fingerprint prevents resuming the old v3 run as though its solver were unchanged.

The earlier **fitting** change was patch `h: .45 -> .35`, `reg: .10 -> .05`;
patch `lam: .025`, mass `.80`, threshold `.05`, and image-router fit settings
were held fixed for that comparison. The latest candidate now changes only
image fitting `reg: .06 -> .0575`, holding h=.36 and lam=.070, as described above.
Projection stays at **.20**,
with .30 available for comparison;
top-1 truncation is not enabled. These are hypotheses requiring lab images,
not a measured optimum. Keep guidance/steps fixed while comparing fits.

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<new-fit> --query-bank data/infoot_vit/cat_val16 --count 16 --projection-bandwidth 0.20 --patch-projection-bandwidth 0.20 --generate --steps 40 --guidance 1.5 --solver euler --seed 20260903
```

New batched fitting logs `pair_geometry.jsonl` on each completed pair, and
`pair_batch_report.json` summarizes the successes in that attempt even if other
pairs fail. Metrics include retained-mass-weighted fitted row entropy/effective
patch count/top-1 probability and mean-normalized cost contrast. Complete
`fit_report.json` and `pair_diagnostics.jsonl` cover all registered successes.
Mapping now separates fitted-plan concentration, query patch-neighborhood
concentration and final projected-patch concentration, and reports variance
within each token set. Support thresholds are recalibrated on training patches
when patch projection h changes. No plan is fitted during mapping.

After fitting, compare thresholds on the **same validation IDs and noise seeds**
without refitting or modifying the fit artifact:

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<new-fit> --query-bank data/infoot_vit/cat_val --count 16 --confidence-threshold 0.10 --generate
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<new-fit> --query-bank data/infoot_vit/cat_val --count 16 --confidence-threshold 0.05 --generate
```

The override and resulting masks are recorded in each mapping manifest. Each
partial `mapping_report.json` also reports counterfactual valid-token coverage
and all-invalid image counts at thresholds 0.05/0.10/0.20/0.30 plus the active
value. That sweep does not generate extra images or choose a threshold. Use
validation to select it, then keep it fixed for test; keep the explicit
all-invalid error policy.

`grouped_partial.yaml` now sets **`device: cuda`**, **`pair_batch_size: 1024`**,
and `pair_checkpoint_every: 10`. Selected pairs are fitted as `[B,196,196]`
costs, cached per-image kernels, plans and solver updates. The last batch uses
its actual size. Batching preserves patch features, sampling, router selection,
per-pair mean cost scaling and the configured bandwidths, MI, entropy and transported mass. This is
the **dense** experiment; the separate low-rank experiments are unchanged.

POT 0.9.7's log-domain partial routine accepts one cost matrix and remains the
serial reference. Its three-set Dykstra loop can be very slow near saturated
supports. The batched implementation reuses PyTorch's matmul, reductions, sorting
and shared MI autodiff. Each row/mass and column/mass block uses exact capped
log water filling, with vector inequality potentials and one mass multiplier.
It solves the same entropic partial subproblem, with no change to the InfoOT
MI, cost scaling, entropy or feasible set. It does not approximate
plans/kernels or substitute balanced marginals for partial constraints. Only the
existing `keep_mass: 1` control uses balanced log-Sinkhorn, as in the serial
reference. Inner stopping is checked at iterations 1, 11, 21,
etc.; a single host check per ten updates replaces per-pair synchronization.
Converged/failed inner states are frozen on device. Outer iterations compact
the remaining active pairs and retain independent full-objective line searches.

Each pair records its inner error/count, primal-dual gap, mass/cap residuals, outer count,
objective terms, accepted step and status. A failed member does not cancel its
neighbors or later batches. A converged member is atomically saved, reloaded,
validated and durably journaled immediately, then its redundant `latest.pt` is
removed. Nonconvergence is never registered as a successful fit. The overall
run fails after processing all batches if any pair failed; details remain in
`pair_failures.jsonl` and `pair_batch_report.json`. Save/validation failures also
retain the checkpoint and allow other members to finish. The first failed pair
also saves exact float64 cost/kernel inputs in the attempt's
`first_failed_pair.pt` for reproduction without copying full banks. Batch
summaries are refreshed after every batch and on solver/callback interruption.

Per-pair iteration metrics are logged every outer iteration. Float32 checkpoints
are written at the first step, every ten steps, and every terminal state;
change `pair_checkpoint_every` to adjust I/O frequency. Numerical computation
stays float64 on the selected device; host transfers occur for packed diagnostics
and checkpoint/artifact I/O. Cached kernels are rebuilt in float64 and checked
against their float32 snapshots on resume. Unfinished/failed pairs restart from
their original feasible initialization, preserving the existing recovery policy;
quantized checkpoints are diagnostics, not silently repaired warm starts.
Registered successful pairs are validated and reused without refitting.

The resource estimate for 2,000 images/domain, 196 patches and B=1024 is about
**35.21 GiB working memory** (conservative workspace allowance), with a
**48 GiB configured limit** for the 80-GB lab GPU. Active checkpoints/temporary plans add about
300 MiB; required plan/kernel storage is about 3.74 GiB before logs, file/container
overhead and failed checkpoints. These are estimates, not measured GPU peaks.

```bash
# Default: CUDA, 1,024 pairs, a fresh fit directory.
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --dry-run
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml

# A smaller explicit CUDA batch, or the original POT serial CUDA reference.
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --pair-batch-size 64
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --serial-pairs

# Explicit CPU batched execution (batch size 1 still uses the batched solver).
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --device cpu --pair-batch-size 1

# Resume with exactly the original configuration and overrides.
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --resume outputs/infoot_vit/<fit-directory>

# Lab CUDA benchmark: identical seeded synthetic pairs, including MI.
python infoot_vit/benchmark_partial_batch.py --pairs 1024 --pair-batch-size 1024 --repeats 3 --output outputs/infoot_vit/partial_batch_benchmark.json
```

Missing/null `pair_batch_size` or `--serial-pairs` keeps the serial reference.
Other dense modes do not accept this setting. Completed older mappings remain
loadable; changed numerical source/config fingerprints require a fresh fit.
Benchmark JSON reports synchronized wall time, pairs/second, allocated/reserved
peak GPU memory, objective/plan agreement and convergence status. It excludes
cost/kernel setup and file I/O; production batch logs separately include those
costs. Solver failures or mismatched results set `valid_speed_comparison: false`
and are reported explicitly. This workspace has PyTorch `2.14.0+cpu`: **CUDA runtime, throughput and
peak memory remain unverified**. The benchmark records `unverified` when CUDA
is unavailable, without silently timing CPU instead.

Run `--dry-run` before fitting. Other modes' dense patch kernels/plans can be
much larger. Resource limits fail explicitly; raise them only when the machine
can support the estimate. No method pools tokens or applies PCA.

For a **separate, explicitly smaller pilot**, build named banks, then point all
comparison configurations to those same banks:

```bash
python infoot_vit/bank/bank_cat.py --split train --max-images 16 --output-dir data/infoot_vit/cat_train_16
python infoot_vit/bank/bank_dog.py --split train --max-images 16 --output-dir data/infoot_vit/dog_train_16
```

`--max-images` selects the first N sorted stable image IDs and records that
selection. It does not claim to be a representative random subset. For a
16-image `grouped_partial` pilot, also pass `--sample-images 16` to fitting to
override the YAML's 2,000-image requirement. Use the same population for every
controlled comparison.

## Fit modes and saved plans

| Mode | What is fitted/projected |
|---|---|
| `patch_global` | Balanced FusedInfoOT on individual unpooled patches, with saved fixed bandwidth state. |
| `whole_map` | A balanced FusedInfoOT router on `[N,196*768]`; one image-weight vector mixes all patch positions. First baseline. |
| `grouped_patch` | The image router plus a global patch model; image weights are shared, patch scores normalize **within** each target image using the global KDE. |
| `grouped_partial` | Balanced image router plus independent fixed-mass partial patch plans for saved selected training-image pairs. Defaults: 2,000 images/domain, top 8 pairs/source; full-pair mode remains available. |

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
plans/image/latest.pt            # Active/failed fit only for grouped_partial
plans/image/iterations.jsonl
plans/patch.pt                   # Global patch plan, when applicable
plans/pair_kernels.pt            # Shared fit kernels/scales and fit-only support thresholds
plans/pairs/<indices_short-ID-hash>.pt  # Gamma, a,b,r,c,s/report, full IDs and order
plans/pairs/<indices_short-ID-hash>/latest.pt  # Active/failed pair only
plans/pairs/<indices_short-ID-hash>/iterations.jsonl
pair_selection.json             # Expected pair set, router hash, ranks/probabilities
pairs.jsonl                     # Accepted selected pair inventory and file hashes
pair_diagnostics.jsonl          # Per-pair objective, caps, mass and raw retention summaries
pair_failures.jsonl             # Batched failure journal; preserved across retry attempts
pair_batch_report.json          # Latest batched attempt counts, timing, failure summary
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

For `grouped_partial`, each final plan is atomically written, flushed, reloaded
and validated before registration in the manifest or flushed pair journal.
Only then is that plan's redundant `latest.pt` removed. Iteration logs and
unfinished/failed checkpoints remain. Resume verifies completed registered
plans and finishes interrupted cleanup; it never treats an unregistered final
file as proof of success. An interrupted individual solve restarts from its
initialization, while accepted pairs and the saved selection are reused.

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
`h * sqrt(mean(training_pairwise_distances**2)/2)`. Projection uses the saved
multiplier (0.20/0.36 in the top-8 dense partial recipe; 0.2/0.7 in balanced low-rank,
1.0 in partial low-rank) and
never estimates a query-batch scale.
An explicit multiplier is part of the fit configuration/artifact identity.
Test-time overrides are recorded in the separate mapped artifact; they rebuild
projection kernels and calibrate partial support thresholds on training data.
There is no calibration from held-out query data or hidden bandwidth sweep.
The partial low-rank fit config still requires multiplier 1; its test-time
override supports other positive multipliers using the saved basis.

### Preview a dense partial fit with a few failed pairs

Testing normally requires a complete fit. For a **stopped, failed batched
`grouped_partial` fit**, `--allow-failed-pairs` explicitly permits inference
using its registered successful pairs (for example, 1,587 successes out of
1,600 selected pairs). No transport is fitted during testing:

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<failed-fit-directory> --query-bank data/infoot_vit/cat_val --allow-failed-pairs --generate --steps 40 --guidance 2.0 --count 16
```

This also works with `--projection-bandwidth`, `--top-k-images` and
`--confidence-threshold`. It does not change the original manifest's `failed`
status or any fitted files, so the fit can still be resumed. All registered
successes must pass file hashes, identities, convergence and plan-constraint
validation. Missing selected pairs must **exactly** match the final
`pair_batch_report.json` and its `pair_failures.jsonl` journal. Unknown missing
files, corrupted successes, active/interrupted fits and failed low-rank fits
are not accepted by this option. A metadata-only `--dry-run` checks coverage
and identities, but full numerical/file validation occurs on actual loading.

After image top-k/selection, routing mass is partitioned into:

- `fit_pair_discarded_routing_mass`: pairs intentionally excluded at fitting.
- `failed_pair_discarded_routing_mass`: selected pairs with recorded failures.
- `successful_pair_retained_routing_mass`: usable, validated saved pairs.

These sum to one. Only successful routes are renormalized; partial-OT rejection
and support-confidence masking are evaluated afterward. A query with no usable
successful routing mass errors, even with the all-invalid-mask bypass enabled.
The failed-pair count fraction alone does not measure its effect on a query;
inspect the per-query discarded routing mass and matched validation images.

`logs/<run>/incomplete_fit_snapshot.json` preserves the read-only snapshot and
its separate mapper identity. The mapped manifest, `mapping_report.json` and
`generation_report.json` label the result with `incomplete_fit` provenance;
the latter also lists failed-pair discarded mass by query ID. A later complete
fit has a different mapper identity. Its historical failure journal does not
cause recovered pairs to be skipped.

All fitting and projection arithmetic uses float64 on the configured CUDA/CPU
device. Saved feature banks stay float32 and mapped features return to the query
dtype/device. Lower-precision fitting is not enabled.

### Float32 storage contract (`grouped_partial` only)

New artifacts use schema `siglip_infoot_mapping_v3`: plans and shared kernels
are float32 on disk, with dtype/version metadata and content hashes. Original
solver convergence and feasibility checks remain float64 and unchanged.
Stored row/column marginals are recomputed in float64 **from the quantized
plan**, never copied from the original plan. Reports record maximum/L1 error,
row/column/mass error and positive entries that underflowed to zero.

Storage validation separately allows the round-to-nearest bound
`|float32(x)-x| <= 2^-24*x + 2^-150` for nonnegative entries. For example, a
row with cap `a`, original feasibility tolerance `tau` and M entries receives
`tau + 2^-24*(a+tau) + M*2^-150`, plus float64 reduction roundoff. Mass and
columns use analogous bounds. This does **not** relax the solver's stopping
rules or renormalize stored plans. Confidence clipping is limited to the
validated solver/storage roundoff budget and its size is logged.

Mapping promotes stored plans/kernels to float64. Resumed fitting reconstructs
the original float64 kernels from immutable feature banks and verifies their
float32 copies match the saved kernels; it does not fit with rounded kernels.
Storage policy and selected ordered IDs participate in fit fingerprints.
Completed legacy v2 artifacts remain loadable; earlier incomplete fits need a
new directory because their numerical-source fingerprints differ.

## Saved logs for tuning and debugging

### Console-only image-router tuning

Use `--tune` to run the **image router only**, using the same training-bank
sampling/seed, float64 solver, cost scaling and convergence checks as a full fit.
It creates no output directory, log/error files, checkpoint or transport-plan
artifact. Plans exist only in memory; this run cannot be resumed or passed to
`infoot_test.py`. Patch fitting, persisted pair selection and low-rank patch-kernel
audits are skipped. Converged dense `grouped_partial` runs preview the existing
top-K selection rule in memory at **projection h**, reporting retained/discarded
probability and concentration after retained-edge normalization. This is a
float64 pre-storage preview; float32 storage can slightly affect values/ties.
Preview time is separate from router `seconds`. No pair plans are fitted or
saved. Router convergence does not validate patch fitting or held-out image
quality. The normal command without `--tune` still saves full fit diagnostics.

```bash
# Current top-8 grouped_partial trial; includes matched .20/.20 projection.
python infoot_vit/infoot_fit.py --config infoot_vit/configs/grouped_partial.yaml --tune

# Low-rank recipe: these overrides change image_solver.*, not patch kernel.h.
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --tune --h 0.7 --lam 0.075 --reg 0.075
```

`--tune-log-every 1` prints every outer iteration; the default is 25 plus the
first and final iterations. The terminal summary includes weighted objective
terms, residuals, router concentration, settings, bank identities and hashes of
the ordered sampled IDs. Nonconvergence returns exit code 2; numerical errors
remain visible in the terminal without creating error files. `--tune` rejects
`--resume`, `--output-root`, `--dry-run` and, for low-rank, `--kernel-check-only`.
`patch_global` has no image router and does not support this mode.

### Weighted objective terms

Balanced `image/patch FusedInfoOT` progress and the low-rank experiment's image
router print the three **signed, weighted objective contributions**:
`cost = <Gamma, C / cost_scale>`, `mi_term = -lam * MI`, and
`entropy_term = -reg * H(Gamma)`, where `H = -sum(Gamma * log(Gamma))`.
Thus `objective = cost + mi_term + entropy_term`. The line also shows `lam`
and `reg` (the formula's epsilon); `cost` has coefficient 1 after the configured
cost scaling. These three fields are already persisted in the balanced solver's
iteration JSONL/checkpoint reports. Printing reuses those values without another
MI/kernel computation. Partial and low-rank **patch** solver records retain their
existing `entropy = sum(Gamma * (log(Gamma)-1))` convention and `+reg * entropy`;
their `entropy` field is unweighted, unlike the `entropy_term` above.

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
  image_report.json         # router settings/history/residuals, including budget stops; also at fit root
  patch_report.json         # analogous balanced global-patch report, when applicable
  fit_report.json           # successful fit summary (also at the fit directory root)
  queries.jsonl             # mapping diagnostics, one completed query per line
  mapping_report.json       # successful mapping summary
```

Files specific to fitting or mapping appear only for that operation. Per-plan
`plans/.../iterations.jsonl` remains available. `latest.pt` remains for active or
failed `grouped_partial` solves and for the other modes. Iteration JSONL
records include the attempt ID and elapsed time, so an interrupted/refitted pair
can be distinguished from earlier attempts. `events.jsonl` also records which
model/pair was active if failure occurs before an accepted first step. Inner
solver warnings captured by POT wrappers are in the iteration's `inner.warnings`.
Balanced model progress prints at iteration 1, every 25 iterations, and terminal
states. Budget-stop errors include the raw fixed-point residual and inner error;
partial-pair settings cannot fix an image-router stop.

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
| Pair pruning | Per-query `fit_pair_retained_routing_mass`, `fit_pair_discarded_routing_mass` and renormalization; separate from `top_k_images` omission and partial-OT rejection. Summary min/mean/max and before/after image weights. |
| Storage precision | Plan/kernel quantization errors and underflow counts, original solver residuals, stored marginal/mass residuals and explicit storage bounds. |
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
- Required plan storage includes active latest/temporary writes for the compact
  partial experiment and retained latest copies for other modes; failed report
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

# Top-8 grouped_partial recipe; pin sampling explicitly for reproducible comparisons.
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/<grouped-partial-fit> --query-bank data/infoot_vit/cat_val --count 16 --generate --weights ema --steps 40 --solver euler --guidance 2.0
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

For saved pair set S, after any optional image-selection operation, compute
`rho = sum_{(i,j) in S} Theta[i,j]` and use
`Theta_saved = 1_S * Theta / rho`. Log `1-rho` as
`fit_pair_discarded_routing_mass`; it is **not** partial-OT rejection. A query
with no usable saved routing mass fails explicitly. Intentional exclusions
require no file; every selected pair must exist and pass its checksum and
state validation. Mapping never selects new pairs or runs a fitting solver.

Confidence estimates retained source mass:
`g_raw=(K_Q @ Gamma.sum(1))/(K_Q @ a)`. Fit-only leave-one-out log-density gating
handles out-of-support queries; this is a heuristic, not a probability of
anatomical correctness. `support_calibration: fixed` with a finite
`support_log_threshold` is also supported. `disabled` is an explicit diagnostic
ablation, used for the pure `s=1` confidence-one regression.

The output is `sum(Theta_saved*g*candidate)/sum(Theta_saved*g)`, with confidence
`sum(Theta_saved*g)` conditional on the retained routing. True zero retained
mass gives a zero placeholder and invalid
mask. Numerical underflow is separately diagnosed; tiny conditional scores
retry in log space. A support-valid confidence underflow is an error, not a
fabricated match. Matching plus rejection recovers each target-image routing
budget. Matched-only image proportions may vary by patch.

**Raw fitted retention**, **smoothed query confidence**, **top-k image omission**,
**fit-pair discarded routing**, **support-invalid mass**, and **thresholded mask
coverage** are separate diagnostics. On normalized saved routes, matched mass
+ partial-OT rejection + support-invalid retained mass sums to one per patch.
The average confidence of unseen queries need not equal `s`.

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

The comparison verifies identical fit-bank fingerprints, selected ordered
training IDs and query IDs. Include
the `patch_global.yaml` and `grouped_patch.yaml` fits when resources permit.
Keep diffusion checkpoint, sampler, guidance and seed identical for image
comparisons. Check source color/pattern, target realism, pose, and whether
rejection removes useful rare attributes. Same-feature-space similarity is not
independent evidence of better images.

Optional `selection: argmax|sample` and `top_k_images` act once per entire image.
Tie breaks use stable target IDs; sampling uses a hash of seed and query ID.
Full support (`mean`, `top_k_images: null`) is the reference. These controls do
not select which patch plans are fitted: that is controlled separately by
`fit_pair_top_k` in `grouped_partial`. Use `--full-pairs` for the full-pair
baseline. Variable grids, learned rejection and joint hierarchical
optimization are not implemented.

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
PDAE mask path without changing parameter counts or checkpoint keys. Compact
partial regressions additionally cover seed-42 sampling, stable probability
ties, split separation, float32 round trips/underflow, strict solver versus
storage bounds, selected-pair corruption, sparse routing accounting, full-pair
equivalence and interruptions before/after plan registration and cleanup.

The existing `tests/test_local_infoot_pipeline.py` covers legacy fixed-bandwidth
scoring; `tests/test_pdae_v2.py` covers pretrained equivalence and attention masks.
One pre-existing legacy test expects `numIter=50` although the unchanged
`infoot/infoot_fit.py` uses 100. It is separately reported, not fixed by changing
the old experiment.

Earlier dense/storage review regression: **145 passed, 1 deselected**, using PyTorch
`2.14.0+cpu` and POT `0.9.7.post1`. The single warning is from the existing
legacy encoder-alignment test's Sinkhorn iteration budget. Command (using the
local test environment's Python):

```bash
python -m pytest tests/test_infoot_vit_mapping.py tests/test_infoot_vit_logging.py tests/test_infoot_vit_mapping_review.py tests/test_infoot_vit_solver_review.py tests/test_infoot_vit_sparse_storage.py tests/test_pdae_v2.py tests/test_stage1a_rgb_eval.py tests/test_local_infoot_pipeline.py -k "not test_bank_fit_script_passes_raw_features_and_cli_lam" --basetemp tmp/vsfinal2 -p no:cacheprovider -q
```

The mapping test module contributes 34 cases; review/logging tests add 25 cases,
and sparse-storage tests add 14. The one deselection is the pre-existing legacy iteration-default
mismatch described above. Syntax compilation and
`git diff --check` also pass. Pair filenames use indices plus a short ID hash
to avoid unnecessarily long Windows paths; full IDs and fingerprints remain
in the saved inventory and plan state.

No real-image bank extraction, full AFHQ fit, GPU solver validation or diffusion
generation on lab checkpoints has been run during this implementation. The
generation test uses mocked checkpoint/data loading; the sampler mask test uses
the existing tiny real PDAE branch. Numerical correctness is not a claim of
translation quality or of global optimality.

**Batched partial revision (2026-10-10): 145 passed, 3 CUDA-only skipped.**
The focused dense/low-rank regression below includes 28 new passing batch
cases: batch-size-1/multi-pair POT agreement, nonuniform capacities, masses
0.3/0.8/1, full 196-patch shapes, finite MI gradients, mixed convergence and
failures, line-search/iteration budgets, incomplete batches, float32 save/load,
interruption/verification failure recovery, CLI settings and benchmark reports.
CUDA placement/agreement tests are present but skipped on this CPU-only host.

```bash
python -m pytest tests/test_infoot_vit_partial_batch.py tests/test_infoot_vit_lowrank_partial.py tests/test_infoot_vit_lowrank.py tests/test_infoot_vit_mapping.py tests/test_infoot_vit_logging.py tests/test_infoot_vit_mapping_review.py tests/test_infoot_vit_solver_review.py tests/test_infoot_vit_sparse_storage.py -q
```

### References and attribution

- [Official InfoOT solver at the audited revision](https://github.com/chingyaoc/InfoOT/blob/352efd202f5b475dc170a8d08a99049689d5ee1a/infoot.py): balanced FusedInfoOT and the kernel/scoring reference; preserve the local repairs.
- [InfoOT paper, sections 4–5](https://arxiv.org/html/2210.03164v2): fused transport and conditional projection.
- [POT partial API](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.entropic_partial_wasserstein) and [implementation](https://pythonot.github.io/_modules/ot/partial/partial_solvers.html): fixed-mass cap constraints and log-domain entropy solver.

The grouped routing, transported-population KDE extension, pair bank and
confidence policy implement the supplied project plan. They are not claimed as
official InfoOT features or empirically superior methods. Grouping alone gives
no anatomical/one-to-one correspondence or real-image-manifold guarantee.
