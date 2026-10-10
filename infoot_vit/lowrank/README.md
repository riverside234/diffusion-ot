# Low-rank grouped patch and partial InfoOT experiments

These separate `grouped_patch_lowrank` and `grouped_partial_lowrank` modes fit
offline mappers and test them with an existing frozen PDAE checkpoint. Start with
[`grouped_patch_lowrank.yaml`](../configs/grouped_patch_lowrank.yaml) or
[`grouped_partial_lowrank.yaml`](../configs/grouped_partial_lowrank.yaml).
The existing `whole_map`, `patch_global`, `grouped_patch`, and `grouped_partial`
artifacts retain their formats and objectives. All six current offline YAMLs
select `device: cuda`; CPU remains an explicit supported option.

| Method | Patch support and constraints | Default ranks: transport / kernel |
|---|---|---:|
| `grouped_patch_lowrank` | One global balanced patch coupling; original uniform marginals | 256 / 256 |
| `grouped_partial_lowrank` | One capacity-constrained coupling per saved image pair; mass `0.8` | 64 / 64 |

The partial default uses 64 because 256 factors for each 196-patch pair would
exceed the existing 4 GB budget and be larger than a dense pair plan. This is a
configurable starting rank, not an accuracy claim. No rank is silently reduced.

## What is preserved, and what is approximate

- Select **2,000 train images per domain**, without replacement, seed **42**.
  Reuse the existing stable-ID sampler; save the population identity, sampling
  algorithm and ordered IDs in the manifest. A smaller population is an error.
  Held-out validation/test IDs are checked against the entire input training
  bank, including images not selected for fitting.
- Keep every **196 × 768** SigLIP2 patch feature unchanged. There is no feature
  pooling, PCA, feature-channel projection, or extra normalization of the
  features supplied to PDAE. Kernel features below are auxiliary numerical
  arrays, not replacement diffusion conditions.
- Keep the existing whole-image FusedInfoOT router, trained on the selected
  **2,000 × 2,000** image maps. Its dense image-level plan is affordable.
- Replace the global **392,000 × 392,000 patch plan** with a nonnegative,
  balanced factorization of rank at most **256**. It is not SVD compression.
- Approximate each Gaussian KDE kernel with **256 positive features**. Evaluate
  the outer sums for cost, KDE mutual information, and plan entropy using fixed
  sampled patch pairs. These are explicit changes to the dense objective.
- Fit in float64; store transport and kernel factor arrays in float32. Small
  bandwidth/random-feature parameters, sampled costs and recomputed marginal
  diagnostics stay float64. Inference promotes saved factors to float64.

## Research and reuse

[Low-Rank Sinkhorn Factorization (ICML 2021)](https://proceedings.mlr.press/v139/scetbon21a.html)
and the authors' [LOT implementation](https://github.com/meyerscetbon/LOT)
motivate the nonnegative factor constraints and mirror-descent approach.
We reuse the Dykstra constraint projection from
[POT's low-rank implementation](https://pythonot.github.io/_modules/ot/lowrank.html),
with `POT==0.9.7.post1` pinned in `../requirements.txt`. This is a private POT
helper, so its source hash and POT version are included in the fit fingerprint.
Updating POT requires new numerical tests and a new fit.

POT's ordinary `lowrank_sinkhorn` is **not** the experiment's objective solver:
it uses squared Euclidean cost and factor entropy. We retain Euclidean cost,
KDE MI and entropy of the reconstructed plan. The MI definition follows the
repository's existing implementation of
[InfoOT](https://arxiv.org/abs/2210.03164), with a new streamed factor derivative.

[Linear Time Sinkhorn Divergences using Positive Features (NeurIPS 2020)](https://papers.neurips.cc/paper_files/paper/2020/hash/9bde76f262285bb1eaeb7b40c758b53e-Abstract.html)
and the authors' [LinearSinkhorn code](https://github.com/meyerscetbon/LinearSinkhorn)
provide precedent for positive kernel features. Our row-normalized Gaussian
feature map below is a documented finite-rank approximation; it does not inherit
an unbiased-kernel claim from the paper.

## Balanced factor constraints and objective

For `n` source and `m` target patches, save

\[
Q\in\mathbb R_+^{n\times r},\quad R\in\mathbb R_+^{m\times r},\quad
g\in\mathbb R_+^r,\qquad \Gamma=Q\operatorname{diag}(g)^{-1}R^T.
\]

Enforce `Q 1 = 1/n`, `R 1 = 1/m`, and `Q.T 1 = R.T 1 = g` with a positive
`min_g`. These imply the **original uniform balanced patch marginals** for
Gamma. No full patch coupling is constructed while fitting or projecting.

For kernel factors `Fx`, `Fy`, use `Kx = Fx Fx.T`, `Ky = Fy Fy.T` implicitly.
Let `U = Fx.T Q`, `V = Fy.T R`, `dx = Fx mean(Fx)`, `dy = Fy mean(Fy)`. Then

\[
J_{ij}= (K_x\Gamma K_y^T)_{ij}
       =\sum_\ell (F_{x,i}U)_\ell (F_{y,j}V)_\ell/g_\ell,
\qquad
I(\Gamma)=\sum_{ij}\Gamma_{ij}\log\frac{J_{ij}}{d_{x,i}d_{y,j}}.
\]

The optimized objective is

\[
L=\sum_{ij}\Gamma_{ij} C_{ij}
  -\lambda I(\Gamma)
  +\varepsilon\sum_{ij}\Gamma_{ij}(\log\Gamma_{ij}-1).
\]

`C` is Euclidean distance between raw tokens divided by its fixed **training
sample mean**. Defaults are `lambda=0.1`, `epsilon=0.05`. The `-1` entropy term
differs from the older dense log by the constant `-epsilon` for a unit-mass
plan; it does not change feasible optima. This is plan entropy, not
`H(Q)+H(R)+H(g)` and not an InfoNCE objective.

Autograd differentiates small pair blocks, scatters the direct Q/R derivatives,
and accumulates the derivatives through U and V. This includes both the direct
and smoothed-joint parts of the MI derivative. Only rank-sized cross products
and linear-sized factor arrays are retained. Mirror updates use backtracking
on this sampled objective and POT's constrained projection. Explicit relative
row, column, shared-g and implied-plan residual checks remain at `1e-8` in
float64; the projection stopping tolerance is `1e-12`.

`converged_sampled_objective` means the unreduced mirror-step displacement per
step size is below the configured stationarity tolerance. It does **not** mean
a global optimum or convergence of the full dense InfoOT problem. Exhausted
budgets or a failed line search leave a failed artifact and a resume checkpoint.
POT warnings are recorded; explicit factor/plan residual checks gate acceptance.

## Partial objective and constraints

The partial method preserves the dense `grouped_partial` **per-image-pair**
objective, original uniform capacities `a=1/P`, `b=1/P`, and configured
`partial.keep_mass=s`. It does not replace the pair bank with a global partial
coupling. For each selected pair enforce:

\[
Q\mathbf1\le a,\quad R\mathbf1\le b,\quad
Q^T\mathbf1=R^T\mathbf1=g,\quad \mathbf1^Tg=s,\quad g\ge g_{min}.
\]

These imply `Gamma 1 <= a`, `Gamma.T 1 <= b`, `Gamma.sum() = s`.
The same cost and **plan entropy** terms apply, but MI uses transported
marginals. With `M=Gamma.sum()`, `P=Gamma/M`:

\[
I_{partial}(\Gamma)=M\sum_{ij}P_{ij}
\left[\log(K_x P K_y^T)_{ij}
-\log(K_x P\mathbf1)_i-\log(K_y P^T\mathbf1)_j\right].
\]

The streamed derivative includes the changing row/column marginals and `M`.
Holding them fixed would optimize a different objective. Let `sq=Q.sum(0)`
and `sr=R.sum(0)`; compute transported rows `Q(sr/g)`, columns `R(sq/g)` and
mass `sum(sq*sr/g)`. These require only factors. At mass 1 the feasible
partial objective agrees with the balanced objective for the same support,
kernels and cost scale; it still has its own unconstrained derivative.

`constraints.py` implements log-domain **KL-Dykstra** projections onto three
convex sets: row capacities, shared factor column sums, and the fixed-mass
simplex with a positive lower bound. It stores row/column correction vectors.
This is a local factor-space extension, with independent convex-reference
tests. It is motivated by
[Iterative Bregman Projections for Regularized Transportation Problems](https://arxiv.org/abs/1412.5154);
it is not claimed to be an official low-rank partial InfoOT implementation.
The balanced mode continues using POT's tested balanced factor projection.
Neither mode compresses a previously materialized dense plan.

Partial kernels use each image's **own training-patch RMS bandwidth**, matching
the dense pair method. One Gaussian random matrix per domain is shared across
images; each image keeps its own center, scale, kernel factors and support
calibration. Both modes share the positive-kernel map, sampled objective,
gradient implementation, mirror descent, checked storage and resumable unit
fitter. Independent Gaussian approximation checks are saved per image.

The default partial estimator uses 784 stratified pairs per 196×196 pair plan,
plus 2,048 IID audit pairs. These index arrays are persisted **once** and reused
across image pairs (common random numbers). Costs/scales are recomputed from
the immutable training banks during fitting/resume; they are not needed for
inference. `estimator.exact: true` is available for small references. More
sample pairs reduce estimator noise but do not remove kernel/rank bias.

## KDE approximation and fixed sampling

Bandwidths use all selected training patches:

\[
s^2=\sum_d\operatorname{Var}_{train}(x_d)
   =\tfrac12\operatorname{mean}_{i,j}\|x_i-x_j\|^2,\qquad \sigma=h s.
\]

This equals the dense training RMS-distance rule without allocating its distance
matrix. Save each domain's mean, scale, sigma and Gaussian random matrix W.
With `z=(x-mean)/sigma`, the unnormalized positive feature
`exp(w.T z - ||z||²)/sqrt(k)` has the Gaussian kernel in expectation.
We normalize each feature row to unit L2 norm using stable shifted exponents.
The common norm term cancels in this normalization. The resulting kernel is
nonnegative, positive semidefinite and has diagonal 1, **but is biased at finite
rank**. It may substantially smooth a narrow kernel. Rank 256 is an experiment,
not an accuracy guarantee.

Default seeds: sampling 42, source kernel 4201, target kernel 4202, objective
pairs 4202, independent audit pairs 4203, transport initialization 4203. Each
component uses its own local generator on the selected device. CPU and CUDA
RNG streams need not match. All states and exact selected indices
are saved; seed values can be changed in the YAML for separate experiments.

The fixed training estimator combines two strata: two uniformly sampled target
patches per source patch, and two uniformly sampled source patches per target
patch. This is **1,568,000 pairs**, with replacement within the pair sampler;
image sampling itself is without replacement. Each pair gets weight `n*m/S`.
This sum is unbiased for fixed factors and a fixed cost scale. Optimizing a
fixed finite sample can overfit it; the estimated training cost scale introduces
another explicit approximation relative to the dense all-pair mean.

Exact-mass control variates reduce variance: estimate `Gamma*(C-1)` and
`Gamma*log(n*m*Gamma)`, then restore the mass terms analytically, including
their gradients. Kernel products and densities within sampled pairs remain
exact for the chosen low-rank kernels. `estimator.exact: true` enumerates all
pairs only for tiny numerical references (at most one million pairs).

Checks written to disk:

- Independent exact-Gaussian pair probes: relative RMSE, absolute errors,
  means and diagonal error. Defaults: 4,096 pairs.
- Exact-Gaussian marginal-density checks against **all** train patches for 32
  seeded query patches, streamed in chunks.
- Independent 32,768-pair objective audit, component standard errors, and its
  gap from the training estimator. Standard errors are reported only for IID
  audit sampling, not for the training strata. Repeated use for model selection
  makes this an optimization diagnostic, not an untouched generalization test.
- Float32-versus-float64 objective audit on identical pairs after serialization.

Kernel relative RMSE above `0.5` emits a warning. Inspect this and density errors
before interpreting translated images. A poor approximation is not fixed by
more optimizer steps. Try a separate kernel-rank experiment and recheck the
budget; increasing rank or changing bandwidth also changes the approximation.
Audit standard errors measure pair sampling error only, not kernel bias or
rank-constraint error. No universal accuracy threshold is claimed here.

## Grouped mapping from saved factors

The existing image router gives weights `alpha[target_image]`. Retain its
mean/top-k/stable-ID tie-breaking options. For each query patch:

\[
A=(F_x^TQ)\operatorname{diag}(g)^{-1}(F_y^TR)^T,\qquad
s_j=\phi(q) A F_{y,j}^T/d_{y,j}.
\]

Normalize these scores **within each target image**, multiply that image's
mapped raw patch vector by its routing weight, and sum across target images.
The target-density correction is retained. A global patch softmax is not used.
Projection streams target-image chunks and smaller target-patch blocks and returns the query's original
`[B,196,768]` shape and dtype. The query patch order is preserved. It never
infers target-grid positions or refits a plan, bandwidth or random feature map.
Targets with zero routing weight are skipped before projection and validity
checks. With argmax, sampling or top-K routing, unselected zero-score targets
cannot change success or diagnostics when the target chunk size changes.
Selected targets still require valid nonzero conditional scores.

This is balanced transport: confidence is one and the condition padding mask
is all false. It does not perform partial-OT rejection. Invalid/nonfinite scores
fail explicitly rather than falling back to uniform scores. Projection bandwidth
multiplier must stay 1 because the saved low-rank kernel is fixed at fitting.

### Partial routing, confidence and masks

The router selects `fit_pair_top_k: 8` target images per training source with
stable-ID tie breaking. Persist that selection before fitting; null retains
all pairs. Inference uses only saved pairs. Intentionally excluded pairs need
no file; missing or corrupted selected pairs are errors. Retained pair routing
weights are explicitly normalized, and discarded routing mass is logged
separately from partial-OT rejection.

For each retained pair, stream scores using its factor products. Normalize
inside that target image and retain the **original target-proposal density
correction**, not the transported target marginal. Confidence is
`K(query,X) * retained_rows / (K(query,X) * a)`, evaluated from kernel factors.
The existing routing aggregation, matched/rejected mass accounting, support
threshold, confidence threshold, valid mask and all-invalid policy are reused.
Leave-one-out training support calibration also uses factors without a P×P
kernel. Prefix/suffix sums exclude self terms without subtractive cancellation.
Only validated storage roundoff is clipped from confidence; saved factors are
never repaired during projection. Zero/underflow failures remain explicit.

No new mode constructs a full patch plan or KDE matrix during fitting or
mapping. Mapping also streams conditional score blocks via
`projection.patch_chunk_size: 64`. Dense modes intentionally keep their old
dense algebra. Low-rank partial mapping keeps a bounded 32-pair device cache;
many saved pairs still mean substantial I/O and computation.

## Storage, memory, resume

At 2,000 images/domain, 196 patches/image and both ranks 256:

| Component | Decimal size |
|---|---:|
| Q and R, float32 | 0.803 GB |
| Fx and Fy, float32 | 0.803 GB |
| Image router, float32 | 0.016 GB |
| Training/audit indices and costs | 0.038 GB |
| Estimated completed directory, including parameters and 256 MiB log reserve | **1.932 GB** |
| One hypothetical dense float64 patch matrix | **1,229.312 GB** |

The configured final directory limit is **4,000,000,000 bytes**, excluding
existing banks. The fitter checks both the estimate and actual completed
directory size, including logs. Checkpoint/atomic-write copies temporarily need
about **1.606 GB** extra for transport alone; interrupted artifacts can exceed
the final budget. Keep additional disk headroom for diagnostics and filesystem
overhead.

The selected-support working-memory estimate is about **21.61 GiB**, not a
measured peak. Loading existing full feature banks before selecting 2,000
images can temporarily cost more. Float64 matrix products and repeated
constraint projections remain substantial computation; low disk usage does
not imply quick fitting. **CUDA is the default in all current offline YAMLs**.
The device propagates to image-router fitting/projection, KDE construction,
sampled costs/gradients, transport solvers and mapping. `--device cpu` is an
explicit override; requesting unavailable CUDA raises a clear error. No silent
fallback is used. CPU copies are restricted to bank/artifact I/O and logging.
Legacy artifacts without a device continue to load on CPU; mapping can override
the saved device without refitting. Float64 GPU throughput and peak memory
still need measurement on the lab hardware.

For **partial rank 64/64, 2,000 images/domain, K=8**, there are 16,000 pair
fits. Estimated factor storage is **1.810 GB**, total completed artifacts with
log/metadata reserves **3.497 GB**, and working memory **5.44 GiB**. These are
estimates, not measured peaks. The resource guard counts a per-pair log reserve;
`optimizer.log_every: 25` bounds routine iteration logging. Full-pair mode or
higher ranks can exceed the 4 GB limit and require an explicit budget change.

The manifest records ordered IDs, bank/preprocessing identities, ranks, all
mathematical settings, runtime versions, source hashes, file hashes and dtype
versions. Saved factors are checked after a round trip. Stored Q/R marginals
are recomputed in float64 **from float32 factors**, never copied from originals.
Float32 storage tolerance derives from round-to-nearest unit roundoff `2^-24`,
product/quotient error, double reductions and a subnormal allowance. This only
applies to reading quantized factors; float64 solver tolerances are unchanged.
Saved final factors are never silently balanced or renormalized.
Negligible factor entries may round to zero; their counts/errors are logged.
Zero plan entries contribute zero to entropy and MI, with the explicit
`log_floor` used for finite boundary derivatives. The shared weights g remain
strictly positive and bounded below.

After a final artifact is atomically written, flushed, loaded, verified and
registered, its redundant `latest.pt` is removed. A failed/unregistered final
does not trigger cleanup. Every attempt retains its logs. `latest.pt` embeds a
factor/metadata checksum. Resume checks it, bank/config/source identities and
constraints, promotes it to float64, and **explicitly reprojects** roundoff back
onto the original constraints; the correction is logged. Thus resumed
trajectories need not be bitwise equal to uninterrupted float64 trajectories.
Completed registered artifacts are reused. A crash between final registration
and cleanup is recoverable. A converged checkpoint interrupted before final
registration gets a fresh numerical convergence check, including one extra
iteration if it was at the budget boundary.

Only `optimizer.max_steps` may be increased on resume. Different ranks,
bandwidths, seeds, sampling, code or runtime versions require a new directory.
Rejected resumes leave the existing manifest, factors and checkpoints unchanged;
only the new attempt's logs record the error. Verification or cleanup failures
while reopening a completed fit also leave its completed manifest intact.
The image router's unfinished diagnostic checkpoint is retained, but its
unregistered solve restarts on resume; registered routers are never refitted.

## Commands

Run from the repository root in the lab environment. Existing compatible banks
work unchanged. Use the pinned optional dependencies:

```bash
python -m pip install -r infoot_vit/requirements.txt
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --dry-run
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml --dry-run
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml
```

Each fit creates a fresh `outputs/infoot_vit/grouped_patch_lowrank_<UTC>_<id>`
directory. Supply `--output-root outputs/infoot_vit_lr` to choose another parent.
The partial mode instead uses `grouped_partial_lowrank_<UTC>_<id>`.
Add `--device cuda:1` to select a GPU, or `--device cpu` for CPU execution.
For reverse translation, swap `--source-bank data/infoot_vit/dog_train` and
`--target-bank data/infoot_vit/cat_train`; fit a separate mapper. Resume with:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --resume outputs/infoot_vit/FIT_DIRECTORY --max-steps 600
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml --resume outputs/infoot_vit/PARTIAL_FIT_DIRECTORY --max-steps 400
```

Map the same 16 held-out cats and decode with the **existing frozen dog PDAE**:

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/FIT_DIRECTORY --query-bank data/infoot_vit/cat_val --count 16 --dry-run
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/FIT_DIRECTORY --query-bank data/infoot_vit/cat_val --count 16 --chunk-size 4 --output-dir results/infoot_vit/lr256_cat_to_dog --generate --train-config configs/stage1a_pdae_v2_l/dog.yaml --eval-config configs/stage1a_eval/pdae_v2_l.yaml --checkpoint outputs/pdae_v2_l_dog/checkpoints/latest.pt --weights ema --steps 50 --guidance 1.5 --solver euler --seed 20260903
```

Replace `FIT_DIRECTORY` with the printed directory. For a comparison, repeat
the same command using the partial fit directory and a new output directory.
`infoot_test.py --device cpu` can map a CUDA-fitted artifact on CPU; GPU
generation requires the appropriate explicit device override.
For a dense-versus-low-rank image comparison, repeat
the second command with the baseline mapper and a fresh output directory, using
the same query bank/count, **immutable checkpoint file**, weights, sampler,
steps, guidance and seed. Generation hashes the checkpoint and saves ordered
query IDs plus per-image noise seeds. Confirm those fields agree between
`generation_report.json` files. The grid is original source / top-1 routed
target reference / translation; the reference is not paired ground truth.

For a dense-versus-low-rank check, first build small separate banks (for example
32 training images/domain using `bank_cat.py`/`bank_dog.py --sample-images 32
--sample-seed 42 --output-dir ...`). Fit the old `grouped_patch` and the new mode
on exactly those banks, with the low-rank YAML's `images_per_domain: 32`.
Use identical image-router settings. `infoot_compare.py --mappings DENSE_DIR
LOWRANK_DIR --query-bank data/infoot_vit/cat_val --count 16` verifies equal fit
populations and compares mapping diagnostics. Do not attempt the old dense
global patch fit on all 2,000 images just to get a baseline: its dense matrices
defeat this experiment's memory target. Small-support comparisons still include
rank, kernel and estimator effects; the numerical tests isolate the algebra.

## Logs and interpretation

Fit outputs:

- `manifest.json`: reproducible identity, resources and registered file hashes.
- `fit_report.json`: last solver segment, image-router concentration, kernel
  errors, estimator definition, storage errors and same-pair quantization audit.
- `logs/<attempt>/iterations.jsonl`: objective components, gradient magnitude,
  mirror residual, step size/backtracking, feasibility, estimated patch routing
  entropy, component entropy, and periodic independent audit/standard errors.
- `image_iterations.jsonl`, `kernel_approximation.json`, `events.jsonl`, console
  logs, `run.json`, and error/traceback files within each attempt's log directory.
- Partial: `kernel_checks.jsonl` per-image Gaussian errors, and
  `pair_diagnostics.jsonl` with each pair's objective/audit, capacity/mass
  residuals and float32 audit differences. `pairs.jsonl` in the artifact root
  registers only verified final files; `pair_selection.json` defines coverage.
  Every pair has the same checked checkpoint/cleanup/resume lifecycle as the
  global balanced factor fit. Completed pairs are not refitted.

Projection saves mapped tensors, masks, per-query routing concentration and
within-image effective patch counts. `logs/<attempt>/mapping_report.json` adds
mapped-feature norm/spread and per-position/across-image variance diagnostics.
Generation saves its grid and protocol JSON. Estimated patch entropy inherits
sampling error and is not clipped to a cosmetically plausible range.

### Validation completed locally

The extension's numerical checks run on CPU. CUDA-specific execution tests are
included and skip explicitly when CUDA is unavailable. Local numerical checks
do not substitute for a lab GPU run.

On 2026-10-10, the combined low-rank and existing offline regression suite
reported **117 passed, 2 CUDA-only tests skipped**. It includes partial
objective/gradient dense references, independent convex KL-projection checks,
capacity/mass limits, factor storage, sparse/full-pair mapping, interruption
and resume, and explicit device propagation through both low-rank modes.

The CPU tests cover exact small dense objective/gradient references for the
chosen kernels, constrained descent, fixed sampling, positive kernels and exact
bandwidths, kernel/sample metadata, float32 marginals and corruption, failed
fits and budget-extension resume, interruptions around final verification and
registration, cleanup, query/target chunk invariance, target-density-corrected
grouped projection, split leakage, CLI dry runs/comparisons and matched frozen
PDAE invocation with mocked weights. A tensor-dispatch test rejects accidental
full patch-pair allocations during kernel construction and optimization. Run:

```bash
python -m pytest tests/test_infoot_vit_lowrank.py -q
python -m pytest tests/test_infoot_vit_lowrank_partial.py -q
python -m pytest tests/test_infoot_vit_lowrank_resume_review.py tests/test_infoot_vit_lowrank_routing.py -q
```

**Actual 2,000-image fitting and translated-image quality have not been measured
locally: the banks and PDAE checkpoints are on the lab computer.** Synthetic
tests establish numerical and artifact behavior, not a quality improvement.
Review kernel errors, audit gap, native/mapped feature spread and matched grids
before deciding whether rank 256 is adequate. Do not compare optimizer losses
from different sample sizes, entropy conventions or kernel settings as if they
were the same objective.
