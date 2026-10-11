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
| `grouped_patch_lowrank` | One global balanced patch coupling; original uniform marginals | 256 / 1200 |
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
- Approximate each Gaussian KDE kernel with **1,200 positive features** in the
  balanced experiment. Evaluate
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
provide precedent for positive kernel features with explicit scale compensation.
The training-moment **OPRF** candidate is based on
[Chefs' Random Tables (NeurIPS 2022), Eqs. 4–8](https://proceedings.neurips.cc/paper_files/paper/2022/file/df2d62b96a4003203450cf89cd338bb7-Paper-Conference.pdf).
It uses PyTorch QR/linear algebra, without replacing the InfoOT optimizer or
adding a transformer/attention dependency. The moment heuristic and orthogonal
directions can reduce estimator variance; this is not a guarantee for SigLIP
features at `h=0.4`. Accuracy is measured before fitting. Repeated synthetic
checks favored the legacy normalized IID method at patch `h=0.75`.
The active recipe now tests **OPRF with orthogonal directions**, rank **1200**,
fit `h=0.75` and projection `h=0.2`. The earlier acceptance evidence does not
validate this new configuration on the lab banks.

## Image-router convergence

For console-only router parameter trials, use:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --tune --h 0.75 --lam 0.15 --reg 0.005
```

This fits only the image router on the same sampled training images. No logs,
error files, checkpoints or plans are saved. `--h/--lam/--reg` override
`image_solver.*`; patch kernels, patch factors and their accuracy gates are not
evaluated in this mode. See [tuning mode](../README.md#console-only-image-router-tuning)
for printing frequency, exit statuses and limits. Remove `--tune` for a full
saved fit with all kernel acceptance checks enabled.

In the active balanced recipe, **kernel acceptance runs before either transport
fit**. The image router then runs before patch-factor optimization. Its settings are
`image_solver.h`, `image_solver.lam`, `image_solver.reg` and
`image_solver.max_outer_steps`. The patch `kernel.h` and `optimizer.*` fields
control a separate problem; `--max-steps` only changes the patch optimizer.

Following the [300-step router log review](../../docs/analysis/vit_infoot_router_300/README.md),
the router recipe used **h = 0.40, lam = 0.075, reg = 0.075, 1,200 outer steps**.
The old router used lam = 0.10, reg = 0.05, 300 steps. Its inner Sinkhorn solves
passed but its outer residual remained above tolerance. The revised weights
reduce MI pressure relative to entropy; they are a lab experiment, not a claim
of optimal translation quality. The latest requested experiment sets both
`image_solver.h` and `kernel.h` to **0.75**. Its next fitting trial uses
**image reg=0.005, lam=0.15**, and **patch optimizer reg=0.025, lam=0.05**.
Reducing image entropy targets diffuse routing. Halving both patch weights
relative to the previous .05/.10 emphasizes geometric cost while retaining
the patch MI/entropy ratio of 2. These are provisional choices, not values
determined by h alone or validated on real banks at h=0.75. Start a fresh fit;
the earlier h=0.4 failure does not establish convergence of this setting.
`projection.bandwidth_multiplier: 0.26666666666666666` sets both projection h
values to **0.2**. Actual Gaussian sigma is still h times each domain's saved
training RMS scale, rather than an absolute feature-space distance of 0.2.

### Router trial after the four-iteration h=0.7 log

The supplied `lam=0.075, reg=0.05` run converged with undamped plan delta
`1.97e-08`, below the `1e-07` tolerance. It is a converged, nearly uniform plan,
not an iteration-budget failure: mean effective targets are **1919.8/2000**.
The cost is **0.9960372**, versus **1.0** for the uniform plan under mean-cost
scaling, a reduction of only **0.3963%**. The weighted MI term is
`-5.7171349e-05`, corresponding to KDE MI of approximately **0.0007623**.

The entropy term `-0.75803287` implies joint entropy **15.1606574**, versus
**15.2018049** for uniform mass over 2000 by 2000 pairs. Equivalently, mean
row entropy is **99.46%** of its maximum and KL from uniform is **0.04115 nats**.
The large absolute entropy offset alone does not establish gradient dominance;
the routing concentration and cost gain are the useful evidence here.

That follow-up lowered **image reg 0.05 -> 0.01** to reduce entropic
smoothing, and raised **image lam 0.075 -> 0.10** modestly to retain neighborhood
structure. These roles follow the [InfoOT objective](https://proceedings.mlr.press/v202/chuang23a/chuang23a.pdf);
the exact values are a local tuning proposal, not paper-prescribed settings.
A small MI scalar does not determine its gradient scale, so it is not a reason
to increase lam by orders of magnitude. Patch-optimizer weights have no new
patch-fit evidence in this router log and remain unchanged.

Use the console-only tuning command above on the lab banks (add the same
`--source-bank data/infoot_vit/cat_train_4000 --target-bank data/infoot_vit/dog_train_4000`
overrides as the full fit). For the original h=0.7 weight comparison, explicitly
use `--h 0.7`; the active command now tests h=0.75. To isolate the entropy
change, compare `--lam 0.075 --reg 0.01` at the same h and samples. Check convergence, effective
targets, maximum row probability, MI and cost. Compare cost/MI/entropy separately;
total objectives with different weights are not directly comparable. If routing
is still nearly uniform, an isolated `reg=0.005` trial is reasonable; if higher
lam causes instability, first retry `lam=0.075` at the same reg. More iterations
will not sharpen an already converged plan by themselves.

Judge the resulting held-out routing and decoded images at projection h=0.20:
concentration alone is not image quality, and this fit log does not establish
that either weight change fixes blur. The full banks/checkpoints are lab-only;
the reported follow-up is analyzed below, but no image-quality comparison is
available here. Start a fresh
full fit after changing weights; saved plans cannot be resumed with a new objective.

### Seven-iteration follow-up and h=0.75 trial

The new supplied h=0.7 router log supports retaining `lam=0.10, reg=0.01`:

| Metric | Previous lam=.075, reg=.05 | Follow-up lam=.10, reg=.01 |
|---|---:|---:|
| Converged iteration | 4 | 7 |
| Undamped plan L1 delta | 1.97e-08 | 2.68e-08 |
| Mean effective targets (of 2000) | 1919.8 | 676.4 |
| Mean-cost-scaled transport cost | 0.9960372 | 0.9772166 |
| Unweighted KDE MI | 0.0007623 | 0.0054944 |
| Cost reduction from uniform | 0.3963% | 2.2783% |

Effective targets decrease **64.8%**, and KDE MI increases **7.21x** at the same
fit bandwidth. Routing is more selective while both runs meet the convergence
tolerance. The larger total objective (0.23794715 -> 0.83669001) reflects changed
weights and is not evidence of worse optimization. These are fitting-plan
metrics, not the held-out conditional routing weights or image-quality results.

The requested next trial raises both `image_solver.h` and `kernel.h` to **0.75**,
keeping router/patch weights, ranks and seeds fixed. The approximation-error
motivation applies to the low-rank patch KDE; the image router uses exact dense
Gaussian kernels. A broader KDE changes smoothing and the MI objective as
described in the [InfoOT paper](https://proceedings.mlr.press/v202/chuang23a/chuang23a.pdf).
It may help kernel approximation but does not guarantee lower measured errors,
and routing may become broader. Check fitting `kernel_quality.json` and compare
held-out mapping/decoded images. MI at h=.75 is a different KDE estimate, so it
is not directly comparable with the h=.7 MI as evidence of improved alignment.

The shared multiplier is now **0.20 / 0.75 = 0.26666666666666666**, preserving
both image and patch projection at **0.20**. The separate projection accuracy
audit remains removed; fitting acceptance and numerical validity checks remain.
The h=.75 experiment requires a fresh fit, not resuming the h=.7 artifact.

### Current h=0.75 router comparison and next MI trial

Both supplied CUDA runs use 2000 images per domain, seed 42, identical printed
bank IDs, ordered sample hashes and cost scale **680.5033570**, with `lam=.10`.
Only reg changes, so the component metrics below are directly comparable:

| Diagnostic | reg=.01 | reg=.005 |
|---|---:|---:|
| Converged iteration | 6 | 9 |
| Runtime, seconds | 6.527 | 6.959 |
| Undamped plan L1 delta | 1.08435e-08 | 2.45825e-08 |
| Maximum marginal residual | 2.28348e-12 | 3.31550e-11 |
| Mean effective target images | 686.0289 | 98.8913 |
| Mean maximum conditional row probability | 4.1059% | 20.1376% |
| Normalized mean row entropy | 0.8439074 | 0.5474866 |
| Mean-scaled transport cost | 0.9773680 | 0.9619906 |
| Unweighted KDE MI | 0.00389559 | 0.00776104 |

The reg=.005 run cuts effective targets **85.6%**, reduces cost **1.57%** from
the previous run (**3.80% below uniform**) and nearly doubles KDE MI at the
same h. Both satisfy the convergence/feasibility tolerances, transport mass
1.0, and report no inner warnings or gradient-floor entries. The last zero
accepted update is the convergence branch, not a stalled line search. Total
objectives across different reg values are not directly comparable.

The current `.10/.005` router is a useful baseline for full mapping, with no
evidence from these averages that further concentration would improve images.
The next configured trial therefore holds **reg=.005 and h=.75**, and changes
**only lam .10 -> .15** to test stronger neighborhood coherence. This is a
moderate experimental perturbation of the [InfoOT MI term](https://proceedings.mlr.press/v202/chuang23a/chuang23a.pdf),
not a value derived from matching loss magnitudes or a demonstrated improvement.
Preserve the baseline for comparison with these router-only commands:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --source-bank data/infoot_vit/cat_train_4000 --target-bank data/infoot_vit/dog_train_4000 --tune --h 0.75 --lam 0.10 --reg 0.005
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --source-bank data/infoot_vit/cat_train_4000 --target-bank data/infoot_vit/dog_train_4000 --tune --h 0.75 --lam 0.15 --reg 0.005
```

Compare **unweighted** MI (`-mi_term/lam`), cost, concentration, residuals and
runtime at fixed h. Multiplying lam by 1.5 increases the weighted MI term even
if the plan does not change. If the new trial brings little benefit or poorer
convergence, use the `.10` baseline; do not keep reducing reg solely to minimize
effective targets. For a full baseline fit, omit `--tune` but retain `--lam .10`.
Each changed objective needs a fresh fit, not `--resume`.

Kernel effective neighbors stay **1971.8747/1976.5002**, as expected with fixed
features and h; changing lam/reg does not refit these exact image kernels.
Their broad neighborhoods and the small MI scalar alone cannot determine the
useful MI gradient scale. `--tune` skips patch kernels/fitting, so it supplies
**no OPRF approximation-error measurement at h=.75**. Run `--kernel-check-only`
or a full fit for that. Patch weights/ranks remain unchanged, with both projection
bandwidths at .20 and no separate balanced projection audit.

Choose the final recipe using the same held-out images: source correspondence,
mapped-feature spread, blur and similarity to the top-1 reference. These logs
describe fitted-plan rows, not held-out conditional routing or decoded quality;
the mean top-1 weight also cannot rule out individual nearly one-hot rows.
The `.15` trial is configuration-validated locally and awaits the lab run.

When testing a new `grouped_patch_lowrank` fit from this recipe, make the
projection setting explicit:

```bash
python infoot_vit/infoot_test.py --mapping outputs/infoot_vit/FIT_DIRECTORY --query-bank data/infoot_vit/cat_val --count 16 --projection-bandwidth 0.20 --generate --device cuda
```

Testing reads the **saved fit configuration**, not the current YAML. With both
saved fit bandwidths at `0.75`, this override sets **image and patch projection
h to 0.20**. Check `image_projection_h` and `patch_projection_h` in the test's
`logs/<attempt>/queries.jsonl`. An older fit with different image/patch fit
bandwidths will not necessarily produce two equal projection bandwidths from
this shared multiplier. Saved transport plans remain unchanged.

The shared balanced solver now checks the **full cost - lam*MI + reg*entropy**
objective before accepting a Sinkhorn update. If needed, it halves the step
along the segment between feasible plans, preserving balanced marginals. This
follows the generalized conditional-gradient direction/line-search pattern in
[POT](https://pythonot.github.io/_modules/ot/optim.html#gcg), while retaining the
existing log-domain Sinkhorn and tensor computations. It uses a monotone
backtracking check, not POT's NumPy/SciPy Armijo implementation. It does not
change the InfoOT objective or introduce an approximation.

Convergence still requires the **undamped** fixed-point L1 residual <= `1e-7`
(or the exact entropy subproblem when `lam: 0`). A tiny accepted step alone is
a stall, not convergence. Inner tolerance remains `1e-10`, and float64
feasibility remains `1e-8`. Older completed artifacts remain loadable;
changed solver source or mathematical settings require a fresh fit directory.

`image_report.json` in the fit root and attempt log records the final status,
kernel self-mass/effective neighbors and iteration history, including on an
outer-budget failure. Iteration logs include objective components, raw and
accepted plan changes, backtracks, balanced residuals and routing concentration.
Numerical exceptions inside an inner solve retain the last successful iteration
and the attempt's traceback. Console summaries appear every 25 router steps.

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
matrix. Save each domain's mean, scale, sigma and random matrix W. The **raw
SigLIP vectors and Euclidean cost stay unchanged**. Only auxiliary kernel
features use `z=(x-mean)/sigma`.

Balanced fits support three explicitly versioned methods:

| `kernel.method` | Kernel feature construction |
|---|---|
| `oprf_gaussian_v1` (active YAML) | Full-scale positive features with a training-moment variance parameter A; orthogonal directions |
| `positive_gaussian_v1` | Full-scale standard positive features, A=0 |
| `normalized_positive_gaussian_v1` | Legacy shifted-exponential row normalization; available for comparisons and existing artifacts |

For the first two methods, with k features and d input dimensions:

\[
\phi_j(z)=\frac{(1-4A)^{d/4}}{\sqrt{k}}
 \exp\!\left(A\|w_j\|^2+\sqrt{1-4A}\,w_j^Tz-\|z\|^2\right).
\]

Gaussian-marginal directions give
`E[phi(x) @ phi(y)] = exp(-||x-y||²/(2 sigma²))`.
OPRF chooses A from the paper's moment heuristic using
`E||z+z'||² = 2/h²` for independent centered training samples. With
`kernel.orthogonal: true`, each block uses orthogonal directions and independent
chi-d radii, preserving Gaussian marginals; the last incomplete block is supported.
The coefficient and exact random matrix are saved. Queries reuse them.
This is a candidate estimator, not a full FAVOR++ transformer implementation.

**No row normalization, clipping, or per-row maximum subtraction is applied to
the new methods.** Those operations would alter the Gaussian scale. Overflow,
nonfinite features and all-zero rows fail explicitly. Finite-rank diagonals
need not equal one. Unbiased expectation does not ensure a particular saved
kernel is accurate. The legacy normalized method is nonnegative and has unit
diagonal, but is biased at finite rank; it may smooth narrow kernels substantially.

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
- Both checks above are repeated after a float32 factor round trip on the same
  samples. They test kernel approximation plus storage, not just quantization.
- A bounded exact-Gaussian comparison: four training images × 32 patches per
  domain, 16 distinct training query patches, seed `kernel.seed + 1000`.
  Compare MI, objective components, full and feasible-gradient directions,
  within-target-image weight total variation, mapped-feature error and spread.
  Full-training bandwidths are reused. The two arms share a fixed nonuniform
  feasible plan and subset-mean cost scale; both use exact sums on the subset.
  This isolates kernel effects, not optimizer or transport-rank effects.
  The plan is a randomly permuted interval-overlap coupling mixed with 20%
  uniform mass. Flat indices refer to the manifest's ordered training IDs and
  original patch order. Equal image-router weights isolate patch projection.
  These are training diagnostics, not held-out image-quality measurements.
- Independent 32,768-pair objective audit, component standard errors, and its
  gap from the training estimator. Standard errors are reported only for IID
  audit sampling, not for the training strata. Repeated use for model selection
  makes this an optimization diagnostic, not an untouched generalization test.
- Float32-versus-float64 objective audit on identical pairs after serialization.

The active balanced YAML requires `kernel.error_policy: error` at the **fitting
bandwidth**: both domains, in float64 and after storage, must have relative RMSE ≤ **0.50**, mean density
relative error ≤ **0.30**, and maximum probed density relative error ≤ **1.0**.
These configurable engineering thresholds screen gross errors; they do not
certify image quality. Failure saves diagnostics and stops **before** the router
or patch optimizer. No inaccurate final kernel is registered in this case.
An explicit `warn` policy is available for diagnostic comparisons; it does not
make a failing kernel acceptable. Older/partial recipes retain their previous
warning behavior. Reference MI/gradient/mapping errors remain visible diagnostics,
not automatically accepted merely because the pair/density gate passes.

A poor approximation is not fixed by more optimizer steps. Compare methods or
ranks using `--kernel-check-only`; changing `h` also changes the intended
kernel and locality, so it is not an isolated approximation fix. Recheck the
resource budget after rank changes. The bounded reference is capped at 512
support patches per domain and 128 queries; no full-bank dense kernel is built.
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
fail explicitly rather than falling back to uniform scores. Balanced projection
supports a positive bandwidth multiplier. **`grouped_patch_lowrank` no longer
runs a separate projection-kernel accuracy audit**, either during fitting,
artifact loading or test-time bandwidth overrides. New balanced fits save the
fitting-bandwidth audit only; older projection-audit metadata is ignored.
Both image and patch projection remain at **0.20** in the active YAML.

For a multiplier other than 1, mapping re-evaluates the saved random bases on
both training supports and queries, including target-density correction. It
retains the original mean and training scale without refitting OT, sampling
new directions or using query statistics. It never combines a narrow query
kernel with broad support kernels. Finite/nonnegative factors, positive finite
densities, nonzero selected conditional scores and finite mapped features are
still required. Passing the fit audit does not establish projection accuracy.
Extra projection factor arrays are temporary float64 memory, not another disk
copy; the resource estimate includes their memory allowance. Multiplier 1
preserves the saved-float32 projection path. The separate
`grouped_partial_lowrank` mode retains its runtime projection audit.

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

Production fitting and mapping never construct a full-bank patch plan or KDE
matrix. Only the explicitly bounded reference audit constructs small dense
matrices. Mapping also streams conditional score blocks via
`projection.patch_chunk_size: 64`. Dense modes intentionally keep their old
dense algebra. Low-rank partial mapping keeps a bounded 32-pair device cache;
many saved pairs still mean substantial I/O and computation.

## Storage, memory, resume

At 2,000 images/domain, 196 patches/image, transport rank 256 and kernel rank 1200:

| Component | Decimal size |
|---|---:|
| Q and R, float32 | 0.803 GB |
| Fx and Fy, float32 | 3.763 GB |
| Image router, float32 | 0.016 GB |
| Training/audit indices and costs | 0.038 GB |
| Estimated completed directory, including parameters and 256 MiB log reserve | **4.904 GB** |
| One hypothetical dense float64 patch matrix | **1,229.312 GB** |

The configured balanced final directory limit is **6,000,000,000 bytes**,
excluding existing banks; the partial experiment retains its 4 GB limit.
The fitter checks both the estimate and actual completed directory size,
including logs. Checkpoint/atomic-write copies temporarily need
about **1.606 GB** extra for transport alone; interrupted artifacts can exceed
the final budget. Keep additional disk headroom for diagnostics and filesystem
overhead.

The selected-support working-memory estimate is about **34.36 GiB**, not a
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

`optimizer.max_steps` and `image_solver.max_outer_steps` may be increased on
resume. Different ranks,
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
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --kernel-check-only
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml --dry-run
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml
```

Each fit creates a fresh `outputs/infoot_vit/grouped_patch_lowrank_<UTC>_<id>`
directory. Supply `--output-root outputs/infoot_vit_lr` to choose another parent.
The partial mode instead uses `grouped_partial_lowrank_<UTC>_<id>`.
Add `--device cuda:1` to select a GPU, or `--device cpu` for CPU execution.
`--kernel-check-only` audits the fitting bandwidth, saves its manifest with
status `kernel_checked` and does not fit either transport. After acceptance
passes, continue with the same config:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --resume outputs/infoot_vit/KERNEL_CHECK_DIRECTORY
```

This reuses verified kernels, then fits the router and patch coupling. A
`kernel_checked` directory cannot yet be used for mapping. Compare alternative
methods in **fresh** directories, keeping seeds, ranks and bandwidth fixed:

```bash
# Standard full-scale PRF, inheriting orthogonal:true from the active YAML.
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --kernel-check-only --kernel-method positive_gaussian_v1
# OPRF with the same orthogonal direction sampler (the active candidate).
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --kernel-check-only --kernel-method oprf_gaussian_v1
```

For IID PRF/OPRF comparisons, use a separate YAML with `kernel.orthogonal: false`.
The legacy normalized method requires `false`; the CLI sets it when that
method is selected. Different bandwidths, methods and direction
samplers are different experiments; retain each full configuration.

`--kernel-rank` supports separate rank comparisons and still enforces the 6 GB
budget. Preserve any CLI overrides when resuming. Existing completed artifacts
retain their saved kernel method and remain loadable; this code change requires
a fresh fit for old unfinished runs. Changing acceptance settings also requires
a fresh directory, so rejected runs retain their original diagnostics.

For reverse translation, swap `--source-bank data/infoot_vit/dog_train` and
`--target-bank data/infoot_vit/cat_train`; fit a separate mapper. Resume with:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --resume outputs/infoot_vit/FIT_DIRECTORY --max-steps 600
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_partial_lowrank.yaml --resume outputs/infoot_vit/PARTIAL_FIT_DIRECTORY --max-steps 400
```

For an image-router budget failure under the **same code and mathematical
settings**, use `--image-max-steps 2400` instead of `--max-steps`. This restarts
an unregistered image router from its initial plan and preserves the failed
attempt's logs. It reuses an already registered router. Keep the original
`--source-bank`/`--target-bank` overrides when resuming. The older failed run
cannot be resumed across this code/recipe change; rerun its original fit
command without `--resume` to create a fresh directory.

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
- `kernel_approximation.json`, `kernel_quality.json`, `kernel_reference.json`:
  sampled Gaussian/density errors, acceptance decisions, and the bounded
  Gaussian-vs-approximation comparison. Saved in the artifact root and attempt
  logs, including rejected runs. The reference file is absent when disabled.
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
python -m pytest tests/test_infoot_vit_kernel_accuracy.py -q
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

The kernel-accuracy tests additionally integrate PRF/OPRF against a Gaussian
using deterministic quadrature, compare exact-Gaussian MI/gradients/grouped
projection, and check error gates, preflight/resume, float32 round trips,
raw-feature preservation and legacy artifact loading. CUDA tests skip when
unavailable. Neither OPRF nor rank 1024 has been validated on the lab banks here.
For this kernel revision, the kernel-accuracy, balanced/partial low-rank,
resume, routing and dense mapping suites passed **120 tests**, with **4
CUDA-only skips**, on the local CPU environment (2026-10-10).

A local narrow-kernel stress test also illustrates why acceptance is mandatory:
256 IID Gaussian vectors in 768 dimensions (data seed 42), `h=0.4`, rank 1024,
kernel seed 4201, 4,096 pair probes and 16 density queries (audit seed 4211).

| Construction | Relative kernel RMSE | Mean density relative error |
|---|---:|---:|
| Legacy normalized IID | 0.740 | 3.797 |
| Full-scale PRF + orthogonal directions | 1.791 | 0.831 |
| OPRF + orthogonal directions | 2.943 | 0.763 |

All three fail the active limits on this synthetic draw. OPRF's density error
is lower here, but its kernel RMSE is worse; there is no across-the-board
improvement claim. These vectors are not SigLIP data. Run the saved-bank
preflight before spending time on InfoOT, and compare methods using every
diagnostic rather than choosing the smallest single error.

### Selected candidate after repeated checks

[Full comparison records](../../docs/analysis/grouped_patch_lowrank_kernel_selection/selection.json)
retain the initial sweep and two follow-up protocols. They use 256 and 1,568
Gaussian vectors/domain, 768 dimensions, multiple data/kernel seeds, 4,096
pair probes and 32 density queries. Both domains and float64/float32 round trips
must pass the original 0.50/0.25/1.0 limits. At `h=0.75`, rank 1024:

| Method | Passed paired runs | Worst kernel RMSE | Worst mean density error | Worst max density error |
|---|---:|---:|---:|---:|
| Legacy normalized IID | 12/12 | 0.3003 | 0.2056 | 0.5596 |
| Full-scale PRF + orthogonal directions | 11/12 | 0.9850 | 0.0641 | 0.3349 |
| OPRF + orthogonal directions | 11/12 | 1.0658 | 0.0632 | 0.3072 |

The comparison selected **legacy normalized IID, patch h=0.75, ranks 256/1024**.
It was more stable across these draws; lower density errors alone did not
protect the other candidates against kernel outliers. The original method
can therefore pass these synthetic checks at a broader bandwidth. At `h=0.4`,
none passed the initial sweep, including rank 1152 (4.75 GB); this is not proof
that every possible estimator/rank must fail. That historical candidate's
storage estimate was 4.35 GB, with a 5 GB guard and unchanged error limits.

**This selection broadens the target Gaussian kernel, rather than solving the
fixed-h=0.4 approximation problem.** That comparison used projection multiplier
1 and image-router h=0.4. The current requested YAML uses fit h=0.75 for both
router and patch kernels and projection h=0.2; it is a new, unvalidated setting.
Narrowing the projection kernel can increase approximation error even if the
fit kernel passes. The current balanced experiment no longer gates projection
accuracy. These are CPU synthetic results, not a real-SigLIP acceptance or
image-quality result; the real banks must pass the fitting-kernel audit before
transport fitting. Start a fresh directory for this changed configuration.
Original completed artifacts retain their own h.

For the newly requested 0.7/0.2 setting, an additional CPU check used 256
synthetic Gaussian vectors, dimension 768, data seed 42, kernel seed 4201,
rank 1024, audit seed 4211, 4,096 pairs and 32 density queries:

| h | Kernel relative RMSE | Mean density relative error | Max density relative error | Accepted |
|---|---:|---:|---:|---|
| 0.7 | 0.3598 | 0.2575 | 0.5885 | No |
| 0.2 | 0.6447 | 0.9697 | 2.4476 | No |

These are historical synthetic diagnostics, not a real-bank result. The requested
bandwidths are retained. `--kernel-check-only` now checks the fitting bandwidth
only; projection error is not measured by this command.

### Next trial after the reported h=0.7 density rejection

The supplied lab console reports source/target kernel relative RMSE of
**0.2903/0.3130** and mean density relative error of **0.1874/0.2747** at rank
1024. The target exceeds the **0.25** density limit. Other audit fields were
not supplied, so this is at least one rejection reason. Transport fitting had
not started: changing `image_solver.reg/lam` or `optimizer.reg/lam` cannot
change these kernel/density errors.

The rank-1200 normalized-kernel lab trial also failed: source/target kernel
RMSE was **0.2603/0.2919**, with mean density error **0.1754/0.2795**.
That next trial used **OPRF + orthogonal directions at rank 1200**,
as requested, and raised the mean-density acceptance limit from **0.25 to 0.30**
for fitting-kernel audits. Fit h=0.70 (now 0.75 in the active YAML), both projection h=0.20,
seeds, transport rank, objective weights, RMSE/max-density limits and the
`error` policy are retained. This changes both the estimator and one acceptance
limit; acceptance alone is not evidence that accuracy improved.
The estimated completed artifact is **4.904 GB**, with the artifact-size guard
retained at **6 GB**. Rank 2048 would be about **7.573 GB**, exceeding
that guard. The following rank-1152 probe is supporting diagnostic evidence,
not an accuracy result for the active rank-1200 candidate.

A local CPU stress check used 256 IID Gaussian vectors per domain, dimension
768, data seeds 42/43, kernel seeds 4201/4202, audit seeds 4211/4212, 4096
pair probes and 32 density queries. These are synthetic vectors, not SigLIP
features; the numbers below are float64 diagnostics under the original
0.50/0.25/1.0 limits, not a lab acceptance of the current candidate:

| Rank 1152, normalized estimator | Kernel RMSE | Mean density relative error | Max density relative error | Pass |
|---|---:|---:|---:|---|
| Source, fit h=0.70 | 0.3589 | 0.2179 | 0.4610 | Yes |
| Target, fit h=0.70 | 0.3233 | 0.2346 | 0.6289 | Yes |
| Source, projection h=0.20 | 0.6577 | 0.8248 | 1.8842 | No |
| Target, projection h=0.20 | 0.4271 | 0.9394 | 2.6052 | No |

Orthogonal PRF/OPRF at rank 1152 reduced fitting-density error on these draws
but also failed at projection h=0.20, so they were not selected as an automatic
replacement. The [OPRF paper](https://papers.neurips.cc/paper_files/paper/2022/hash/df2d62b96a4003203450cf89cd338bb7-Abstract-Conference.html)
motivates variance reduction; it does not guarantee accuracy on these banks.

Run a **fresh audit** before attempting transport fitting:

```bash
python infoot_vit/infoot_fit_lowrank.py --config infoot_vit/configs/grouped_patch_lowrank.yaml --source-bank data/infoot_vit/cat_train_4000 --target-bank data/infoot_vit/dog_train_4000 --kernel-check-only
```

This command checks the fitting kernels against the configured limits. Then
resume the printed directory with the same command, replacing
`--kernel-check-only` with `--resume outputs/infoot_vit/KERNEL_CHECKED_DIRECTORY`.
Alternatively, omit `--kernel-check-only` to audit fitting kernels and then fit
transport in one run. Start a **fresh directory** after the projection-audit
removal: implementation fingerprints prevent resuming an earlier failed run
with changed code. Existing completed artifacts remain loadable.

The latest supplied OPRF rank-1200 lab run passed fitting acceptance, with
source/target RMSE **0.3010/0.2548** and mean density error **0.06715/0.07540**,
then stopped at the former projection gate. That separate gate is now removed
at the user's request. This allows transport fitting to proceed after fit
acceptance; it does not change the projection estimator or demonstrate improved
image quality.
