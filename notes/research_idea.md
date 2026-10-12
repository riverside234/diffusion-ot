# Research Idea: InfoOT Alignment Between PDAE Latent Spaces

## Current Stage 1B pilot — 2026-09-28

**v4.5** adds own-encoder spatially-correlative reference matching to v4, with
alpha 0.20 and frozen centered-RMS calibration from the initial training bank.
It uses PDAE convolution stages 0/1, 8x8 spatial relation maps, and symmetric
row-cosine comparisons. The prior is detached; global conditional kernels and
all v4 neural objectives remain unchanged. Those objectives include global MLP
InfoNCE, flow, MI/relative alignment, variance/covariance protection, plus coarse
RGB 0.02, histogram 0.02, layout 0.02 and target patch SWD 0.01. There is no
external teacher or GAN. This tests whether richer reference layout matching
improves held-out translations; improved self-measured cost is not sufficient.

Train/evaluate `self_supervised_infonce_v4_5_sit_b2.yaml` in the respective
config directories. v4 remains the matched control. Calibration is checkpointed,
raw/EMA descriptor provenance is enforced, and evaluation saves source/reference/
output grids. See [implementation](../docs/analysis/stage1b_v4_5/implementation.md).
Full AFHQ quality is pending. Stage 2–3/4 defaults remain on the earlier baseline.

## Retained v3 control — 2026-09-27

The v3 self-supervised global MLP InfoNCE experiment follows the v2 analysis:
`configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml`.
Keep original flow-only Stage 1A initialization, native flow, neural MI,
relative alignment, matching variance/covariance, and bidirectional source
InfoNCE with separate cat/dog MLPs. Disable the live neural cosine-cost term;
its detached fused-solver cost remains. Covariance weight is 0.30 and matching
head LR is 1e-5. InfoNCE remains 0.05, with gradient through conditional W
scaled to 0.10 while its forward condition and direct generator gradients
stay unchanged at a fixed state. A paired 1.0-gradient control isolates this
change within the revised recipe. PCGrad and EMA RMS remain enabled.

This is a 2,500-step pilot, not an established quality improvement. Use the
224-reference/224-target evaluator for matched comparisons. Stage 2–3/4
defaults still select the previous PatchNCE baseline until v3 is evaluated.
The historical external-supervision formulation below is a separate control.
See [architecture notes](training_architecture.md),
[v2 evidence](../docs/analysis/stage1b_v2_5000/review.md), and
[v3 implementation](../docs/analysis/stage1b_v2_5000/implementation.md).

## Historical external-supervision recipe — 2026-09-22

Stage 1B now keeps six task groups: native flow, InfoOT/conditional alignment,
matching variance/covariance protection, source perceptual InfoNCE, target
RGB PatchGAN realism and HistoGAN source-color consistency (weight 0.02).
PCGrad is enabled separately on E/H/G; the norm-ratio
controller is disabled. Initial Stage 1A latent anchoring, null/conditioned
teacher preservation, neighborhood KL, matching contrasts, structure InfoNCE
and code recovery are off. Both cosine objectives remain off. Stage 1B starts
from the current Stage 1A DINO EMA checkpoints and writes outputs/stage1b_patch_color.
The RGB critic replaces only the DINO-feature adversarial path. Existing
perceptual/structure DINO diagnostics remain; histogram targets are detached
original-source RGB. DINO-critic and PatchGAN-only controls save the same fixed
validation images and per-image color metrics. Other losses and settings are fixed.

The sections below retain the previous research formulation and describe
optional mechanisms that are no longer all active. Use the updated
[architecture notes](training_architecture.md),
[current loss equations/graph](../docs/stage1b_current_losses.md), and
[decision record](../docs/analysis/stage1b_patchgan_histogan/review.md) for the
active recipe. The supplied 5,000-step logs motivate simplification/protection;
they do not establish a best loss subset or prove PCGrad will improve images.
RGB-uv histograms include the background and do not enforce spatial markings
or uniform exposure. No new AFHQ quality comparison has yet been measured.

Updated 2026-09-18: native Stage 1A refinement, current contrastive Experiment D,
Stage 2–3 offline alignment, and Stage 4 FID/source-SSIM. Audited against current
training code and YAML; implemented mechanisms and observed results are separate.

## Current thesis

This project asks whether unpaired domain-specific diffusion autoencoders can
learn semantic spaces that support conditional cross-domain translation. The
first case is Cat ↔ Dog on AFHQ.

The active method is **Experiment D**. It jointly adapts semantic encoders,
InfoOT matching heads, and the added conditioning components of each diffusion
generator. It applies image-level losses to decoded full InfoOT conditional
means. The pretrained SiT backbone, VAE, DINOv2 teacher, and learned null tokens
remain fixed.

The current revision differentiates the feature-dependent RMS kernel scale,
adds VICReg-inspired variance/covariance protection on matching outputs, and
uses distance-normalized neighborhood teaching. The RMS-only follow-up still
lost about 90% of matching-head spread despite nearly stable raw encoder
spread. Full RMS derivatives remove the previously demonstrated spurious
gradient but are insufficient protection. Relative-distance neighborhood
logits remove another reproducible contraction incentive; variance and
covariance penalties address spread and redundancy. The supplied fit-0.55 pilot
preserved much more spread, but remaining head concentration and generator-side
native loss trends motivate V/C weights 0.05/0.01 and lower G learning rates.
The proposed native Stage 1A prediction-preservation term is now disabled:
the target is ground-truth reconstruction, not reproduction of imperfect
Stage 1A outputs. The earlier supplied pre-correction run improves matching
floor coverage but has mixed decoded results at matched step 1,500; its native
preservation coefficient was still 0.05. It does not isolate the generator LR
effect or validate the teacher-disabled recipe. V/C weights are retained;
the current LoRA LR trial uses 3.75e-6 with adapter LR 5e-6. It uses
32-image reconstruction probes and 16-image conditional translation
panels for subsequent decisions. See `docs/analysis/stage1b_revised_vs_original/config_revision.md`,
`docs/analysis/stage1b_ground_truth/review.md`, the earlier
`docs/analysis/stage1b_fit055_v2/review.md`, and
`docs/analysis/stage1b_rmsgrad_1500/review.md` for the new evidence and GitHub audit.

The newer `results/co-training/logs/` run reaches 5,000 steps with native
conditioned preservation already disabled. Its final matching spread is
0.503/0.566, but 46.9%/21.9% of coordinates remain below the floor and matching
effective rank is 28/29 versus raw normalized encoder rank 51/52. The best mean
four-image decoded DINO probe was near step 3,000; image-quality superiority
was not established. This locates remaining concentration in matching geometry,
not proven collapse of the entire raw encoder. See
`docs/analysis/stage1b_gt_5000/review.md`.

The current implementation adds matching positive-set contrasts while retaining
KL and V/C, decoded original-source structure InfoNCE, G-only target-code
InfoNCE, and bounded decoded encoder-gradient balancing. Current YAML uses
64 native reconstruction samples and 32 decoded queries per direction, compared
with 16/4 in that supplied run. The entropy-0.02, contrastive, 64/32-batch recipe
has no supplied matching quality run yet. The separate 32k Stage 1A
refinement review is recorded in `docs/analysis/stage1a_refine_32000/review.md`.
The losses average their samples, so equal native/decoded counts are not
required. Compare the increased LoRA LR to 2.5e-6 using the same 64/32 batches;
the old logs cannot isolate both changes. See
`docs/analysis/stage1b_lora_lr_plateau.md` for the batch-scaling rationale.

Earlier frozen-generator results are controls. They showed that a converged
transport solver and improving conditional KL can coexist with weak decoded
correspondence. That evidence motivates generator adaptation and decoded-image
supervision, but it does not prove that freezing G was the only cause.

The current one-sentence hypothesis is:

> If domain encoders preserve comparable structure, InfoOT learns a useful
> conditional distribution in a shared matching geometry, and the target
> conditioning path adapts to the resulting conditional means, then a projected
> source code can guide a target-domain diffusion model while retaining source
> viewpoint, framing, and other shared attributes.

This note defines the scientific method and claims. Implementation details and
commands live in `notes/training_architecture.md` and
`docs/stage1b_experiment_d.md`. The full-bank run contract is documented in
`docs/stage23_stage4_full_dataset_plan.md`.

| Stage | Role | Data |
| --- | --- | --- |
| 1A: base and optional native refinement | Learn useful domain codes and native generation; refine against original RGB | Each domain independently, original RGB and cached VAE latents |
| 1B: Experiment D | Co-train E, matching heads and selected G parameters | Training minibatches, fixed held-out probes |
| 2: frozen banks | Encode the selected paired checkpoint | All training Cats and Dogs |
| 3: global InfoOT | Fit one coupling shared by both directions | Complete frozen training banks |
| 4: final evaluation | Translate, compute FID/source SSIM, save inspection pairs | All held-out sources and real target references |

Stages 2 and 3 share one entry point named **Stage 2–3: Frozen Bank
Construction and Global InfoOT Fitting** (short name: **Offline Alignment**).

## 1. Scientific question

Given unpaired samples

\[
x_i \sim P_X \quad\text{and}\quad y_j \sim P_Y,
\]

learn separate diffusion autoencoding branches with semantic codes

\[
z_i^X = E_X(x_i), \qquad z_j^Y = E_Y(y_j),
\]

then estimate, for a new source image, a target-domain conditioning vector that
preserves shared semantic structure without requiring a paired target image.

The central questions are distinct:

1. Do the Stage 1A codes reconstruct their own domains?
2. Does InfoOT identify conditional structure across domains?
3. Does the full conditional mean remain decodable by the target generator?
4. Does the decoded image look like the target domain and preserve source
   attributes?

A positive result on one question does not establish the others. In particular,
transport convergence and UMAP overlap do not establish image translation.

## 2. Model

Each domain has a PDAE-style branch:

\[
x_d \xrightarrow{E_d} z_d,
\qquad
(\epsilon, z_d) \xrightarrow{G_d\text{-conditioned SiT}} \hat x_d.
\]

`E_d` maps a VAE latent to a learned-LayerNorm 512-dimensional semantic code.
This raw decoder code is not unit-L2-normalized. `G_d`
is the semantic conditioning path inside a SiT-B/2 flow model: `z_proj`, token
MLPs, AdaLN-Zero injections, a final adapter, and attention LoRA. Semantic CFG
uses a learned null code trained by 0.10 condition dropout.

Stage 1A trains the Cat and Dog branches independently. AdaLN and LoRA affect
all 12 SiT blocks; LoRA rank/alpha are 64/64. This stage tests reconstruction
and unconditional/null behavior, not cross-domain alignment.

### 2.1 Native image supervision before co-training

The active Stage 1A `*_sit_b2_dino.yaml` recipes keep real-data flow matching
at coefficient 1 and add same-domain DINO image losses against **original RGB**:
perceptual weight 0.05 and patch-self-similarity structure weight 0.01. Both
update E and G. Native-code InfoNCE is disabled (weight zero), including its
recovery forward/backward and validation retrieval. The prior `*_refine.yaml`
configs retain that optional G-only objective for historical comparisons.

The new configs train for 40,000 updates from fresh E/conditioning layers on
pretrained SiT/VAE, with encoder/adapter/LoRA learning rates 1e-4/1e-4/2.5e-5.
There is no checkpoint warm start or automatic resume. They decode 32 samples
every four updates, ramp image losses over 2,000 updates, and write to
`outputs/stage1a_{cat,dog}_dino`. Perceptual strength is retained while structure
is reduced because the 32k pilot improved structural proxies sooner than
appearance fidelity; the final weights remain experimental. Validation now
actually decodes the requested 32 images rather than being capped by a
16-image loader. These changed rates/objectives are not a single-variable ablation.

The [32k log review](../docs/analysis/stage1a_refine_32000/review.md) shows the
old fresh low-LR model lagging the original Stage 1A checkpoint, despite
improving RGB validation and nearly saturated code retrieval. See
[the new recipe](../docs/analysis/stage1a_dino_recipe/review.md) for rationale.
The current Stage 1B YAML still selects the original base Stage 1A checkpoints;
promote new checkpoints only after matched original-image and translation
validation. Ordinary semantic dropout/null learning remains active, and no
cross-domain alignment or discriminator is added to Stage 1A.

### 2.2 Matching space and decoder space

Experiment D introduces a residual matching head `m_d` for each domain:

\[
u_i = m_X(z_i^X), \qquad v_j = m_Y(z_j^Y).
\]

InfoOT kernels and transport operate on `u` and `v`. The diffusion generators
still consume raw `z`. This separates the geometry needed for cross-domain
matching from the coordinates learned for reconstruction.

## 3. InfoOT formulation

Let

\[
\Gamma \in \Pi(a,b)
\]

be a transport plan between source and target reference samples. The active
solver follows the official InfoOT projected-Sinkhorn update. In the fused
variant, a fixed cross-domain structure cost is combined with the mutual-
information derivative before projection back to the transport polytope.

The structural cost compares frozen DINOv2 pooled patch self-similarity
descriptors. These descriptors are in one fixed space for both domains and aim
to emphasize layout over species identity. They are a training prior, not
ground-truth correspondence.

The active settings are:

\[
h_{\mathrm{fit}}=0.55,\qquad
\lambda_{\mathrm{MI}}=0.10,\qquad
\varepsilon_{\mathrm{ent}}=0.02.
\]

The 0.02 entropy setting is the revised active recipe, not a setting validated
by the earlier supplied logs. Stage 4 inherits it from a new Stage 2–3 fit.

Primary Stage 1B evaluation should match the selected checkpoint's saved fit
bandwidth, entropy epsilon, MI and cross-cost weights; projection remains 0.10
for the active recipe. Check an older run's saved configuration rather than
assuming today's YAML describes it. Bank size and RMS calibration also affect
the estimator, so identical hyperparameters alone do not establish identical
training, quick-evaluation, and full-bank protocols. Separate sensitivity
sweeps from the matched checkpoint comparison.

The plan starts from the independent product and must pass strict marginal and
convergence checks. Plan entropy is a diagnostic. A stable value can reflect a
stable solution and does not by itself indicate either success or a software
error.

## 4. Conditional projection

For an unseen source matching feature `u`, InfoOT estimates the density ratio

\[
r_j(u)=
\frac{\widehat f_{\Gamma}(u,v_j)}
     {\widehat f_X(u)\widehat f_Y(v_j)}.
\]

The primary target-code readout is

\[
\widehat T_{X\rightarrow Y}(u)
=
\frac{\sum_j r_j(u)\,z_j^Y}
     {\sum_j r_j(u)}.
\]

The joint-density estimate uses both the source kernel and target smoothing
kernel induced by the fitted plan. Division by the estimated target marginal
density corrects for target sampling density. The implementation therefore
must retain both target smoothing and target-density correction. It must also
stabilize denominators numerically and normalize every conditional row.

The same fitted `Gamma` is reused for new queries; OT is not re-solved per
query. `h_proj=0.10` controls conditional readout smoothing independently of
`h_fit`. The active method averages all target-bank codes and applies no top-k
truncation.

The output lies in the convex hull of the raw target codes. It can differ from
every observed target code and need not correspond to a real target image. It
is an estimated conditional mean, not a novel sample guarantee or an exact
ground-truth match.

Conditional means are expected to contract variance:

\[
\operatorname{Var}(Y)
=
\operatorname{Var}(\mathbb E[Y\mid X])
+
\mathbb E[\operatorname{Var}(Y\mid X)].
\]

For this reason, matching the marginal distribution of conditional means to
the target distribution is not part of the active objective. The prior
projection-support loss is disabled and retained only as a control.

## 5. Why Stage 1B also changes

Using the conditional expectation only after training leaves a mismatch: the
encoders and frozen generator were trained mostly on individual same-domain
codes, while inference supplies an average whose geometry depends on InfoOT.
The earlier 8,000-update guarded run improved bidirectional conditional KL by
about 5.1%, yet expected structure cost improved only about 0.57% Cat→Dog and
0.12% Dog→Cat. Projected means stayed contracted and decoded correspondence was
weak.

Experiment D addresses two different bottlenecks:

1. matching heads give InfoOT an adaptable geometry without forcing all
   alignment changes into reconstruction codes;
2. generator adaptation and decoded losses train the target conditioning path
   on the actual conditional-mean input used at inference.

This is why the Stage 1B revision is part of the Stage 2–4 conditional-
projection plan. Conditional projection itself does not mathematically require
a new loss, but the end-to-end translation system must test and, if needed,
adapt to its decoder inputs.

## 6. Experiment D objective

For each update and domain, 96 reference samples fit the minibatch plan and 32
disjoint samples act as unseen conditional queries. The full minibatch
conditional mean is computed in both directions. All 32 queries per direction
are decoded through a differentiable 20-step flow sampler.

The retained losses are:

- same-domain real-data velocity MSE, weight 1 per domain, 32 native samples;
- variance-normalized fixed Stage 1A raw-latent anchoring, weight 0.01 per domain;
- detached-plan InfoOT neural feature objective, outer weight 0.20 and inner
  MI weight 0.10 (full-ramp coefficient -0.02 on MI);
- within-domain relative-distance DINO-neighborhood KL, weight 0.10 per domain,
  on all 128 encoded samples;
- conditional teacher KL, weight 0.02 averaged over directions, on 32 disjoint
  queries and all 96 target references, teacher temperature 0.05.

Alignment, both KL terms and matching contrasts use `a(t)=min(t/1000,1)`.
Raw latent anchoring and null preservation are not ramped. The fused frozen
DINO cost guides the detached solver; no Sinkhorn or cross-cost gradient is
backpropagated to the encoders.

The matching protection adds `0.05 * mean_domains(V) + 0.01 * mean_domains(R)`
from update 1, with no warmup. For each domain's 96 unit-normalized references,
use `U = sqrt(D) * (M - mean(M))` and population covariance `C = U.T @ U / N`.
`V` averages `max(0, 0.7 - sqrt(C[j,j] + 1e-4))`; `R` averages squared
off-diagonal entries over `D*(D-1)`. The dimension scaling makes the floor
feasible for 512-dimensional unit vectors. This uses VICReg's anti-collapse
principle, with a different covariance reduction and existing semantic
objectives in place of paired-view invariance. Reference domains are kept
separate; raw decoder codes and conditional means are not its targets.
See `docs/analysis/stage1b_experiment_d_vicreg.md` for the derivation and limits.

Following the relative-distance idea in relational knowledge distillation
(https://arxiv.org/abs/1904.05068), the active neighborhood KL uses negative
squared distances divided by their own off-diagonal mean and temperature
0.20, for student and frozen teacher separately. Its live student denominator
prevents pure uniform contraction from improving neighborhood KL. This uses
KL rather than RKD's Huber loss and retains separate spread protection;
it does not make the actual OT features or decoder codes batch standardized.

### 6.1 Additional matching contrasts

The matching neighborhood contrast uses 96 references per domain, cosine
student logits at temperature 0.20, self-exclusion and frozen teacher neighbors.
The conditional contrast uses the existing InfoOT log probabilities, without
an extra temperature or a cross-domain raw-code similarity. Eight highest-score
teacher candidates plus kth-score ties are positives in each row:

\[
L_{\mathrm{positive\ set}}=-\log\sum_{j\in P_i}p_\theta(j\mid i).
\]

Weights are `0.01*a(t)` for neighborhoods and `0.005*a(t)` for conditionals,
with equal averaging across domains/directions. This objective can concentrate
on one positive; it is not average-positive SupCon. All-positive rows from ties
contribute zero and remain in the mean. Full-distribution KL and V/C stay active.
Selecting positives changes only this auxiliary loss, not the full InfoOT mean.
Contrastive training does not guarantee coordinate variance or effective rank.

### 6.2 Decoded structure, realism and target-code recovery

The decoded-image term is

\[
L_{\mathrm{image}}
=
\rho(t)
\left(
0.10 L_{\mathrm{structure}}
+
0.01 L_{\mathrm{adv}}
+ 0.01 L_{\mathrm{structure\ NCE}}
\right),
\]

where

\[
\rho(t)=\min(t/2000,1).
\]

`L_structure` compares generated-image DINO structure with the descriptor of
the VAE-reconstructed source. `L_adv` uses target-domain DINO feature
discriminators whose real inputs are VAE-reconstructed target references.
The current adversarial-only DiffAugment pilot shifts both real and generated
RGB before DINO by independently sampled integer offsets, at most 6.25% per
axis with zero padding. Gradients pass through augmented fake images to G and
the live condition path. Structure and code losses stay unaugmented; fixed
validation is also unaugmented. Per-target augmentation RNG states are saved
for exact same-recipe resume. This changes the adversarial training objective
and needs a fresh run in `outputs/stage1b_d_aug`; it is not yet a measured
solution to the earlier plateau. See the
[implementation note](../docs/analysis/stage1b_diffaugment/review.md).
`L_structure NCE` instead retrieves the cached descriptor of the **original
source RGB** among 96 disjoint source reference descriptors. This distinction
is an implementation fact; original-RGB native validation does not convert
all Stage 1B decoded training targets to original RGB.

For each direction, the separate code objective is

\[
L_{\mathrm{code}}=0.02\rho(t)\operatorname{InfoNCE}
\big(E_{\mathrm{target}}^{\mathrm{fixed}}(\hat x_0),
\operatorname{sg}(\bar z_{\mathrm{target}}),
\{\operatorname{sg}(\bar z_{\mathrm{target},j})\}_{j=1}^{32}\big).
\]

It retrieves the supplied full target mean among all current same-direction
query means, using the already generated final diffusion latent. Readout
parameters/buffers and keys are detached; selective autograd trains G only,
including blocking the conditioning path to E/head. There is no extra rollout,
cross-step queue or RGB-to-VAE cycle. Source fidelity is measured in shared DINO
space; directly comparing recovered dog codes to raw cat codes would assume
coordinate correspondence that the two independent encoders do not provide.

Both decoded InfoNCE terms use cosine temperature 0.20, mask near-positive keys
at similarity 0.95 (with numerical tolerance), and skip invalid/no-negative
rows. Inspect usable negatives and chance-aware retrieval alongside matched
cosine: zero loss with zero usable negatives is not success. Code recovery is
a conditioning-use constraint, not a guarantee of realism or source fidelity.

Semantic condition dropout remains 0.10 for same-domain training. Because the
shared adapters and LoRA also affect null-conditioned predictions, a fixed
Stage 1A teacher constrains null-branch velocity. The learned null token itself
does not update.

The optional native conditioned Stage 1A teacher loss has weight zero in the
active recipe. Stage 1A images are imperfect, so native quality is assessed
against original dataset RGB images. Pixel MSE/PSNR/LPIPS use the same
crop/resize as latent caching, with separate VAE reconstruction diagnostics.
The existing flow loss already uses real-data-derived velocity as its target;
its absolute value is reported rather than percentage change from Stage 1A.

### 6.3 Translation-first gradient balance

The encoder-only controller measures the joint Cat+Dog gradients of weighted
native reconstruction and the already-ramped decoded image objective (excluding
G-only code recovery). Its target norm ratio is `2*rho(t)`; multiply the decoded
E gradient by `clamp(target_ratio / measured_ratio, 0.25, 4)`. It logs the gradient
cosine and leaves conflicts intact. Tiny gradients/zero ramp skip adjustment.
Other E objectives and head/G decoded gradients retain their configured weights.

The 2.0 target is a pilot heuristic, not an optimal range established by papers
or these logs. It is a gradient-component ratio before clipping, not a ratio
of total gradients or actual Adam updates. Separate norm-1 clipping applies to
combined-domain E, head, G and discriminator groups. The legacy 0.25 gradient
guard/conflict projection is disabled. The G-only code objective is added by
selective autograd, so the complete update cannot be described as a single
ordinary weighted-sum backward call.

## 7. Inference

The step-2,600 review adds a correction to the neural update: backpropagate
through the recomputed RMS kernel scale as well as the feature distances.
The fixed-plan forward values and inference estimator are unchanged. Detached
RMS updates can produce a spurious radial shrinkage gradient even on unit
vectors; a finite-difference counterexample is recorded in
`docs/analysis/stage1b_experiment_d_2600/review.md`. This correction has CPU
mechanism validation; improved AFHQ translation remains an empirical question.
The six fixed validation probes show a real tradeoff: conditional KL improves
30.76% by step 2,500 while Cat/Dog reconstruction flow losses rise 7.29%/6.60%.
Matching spread falls about 85% on the same references. Active D now also
requires outer OT convergence. The original 100-update budget was exhausted
at validation step 2,000 despite feasible marginals. A later training batch
also exhausted the revised 300-update cap at a plan change of `2.01e-5`, so
the active cap is now 1,200 with the same `1e-5` tolerance and early stopping.
This is numerical headroom, not a change to the loss or semantic validation.

After selecting a Stage 1B checkpoint:

1. freeze its paired EMA encoder, matching head, and adapted generator;
2. encode versioned Cat and Dog reference/projection banks;
3. fit one global transport plan over all training matching features, using
   its transpose for the reverse direction;
4. store the coupling, marginals, full raw-code/matching-feature banks,
   paired model state, fixed RMS calibration, and provenance hashes;
5. encode a new source image;
6. pass its raw code through the source matching head;
7. calculate full density-ratio weights over the target projection bank;
8. average raw target codes with those weights;
9. condition the adapted target diffusion generator on that mean and fresh
   target noise.

This pipeline is implemented by `scripts/run_offline_alignment.py` with
`configs/stage23_offline/full_sit_b2.yaml`, writing `outputs/s23_full`.
Kernels and density terms are reconstructed from the saved banks and
calibration. All available training samples participate; batching/tiled
computation does not mean independently fitted mini-batch couplings.

The full-bank projection protocol, `full_bank_fixed_rms_v1`, freezes RMS
scales from training reference banks. Query chunk size cannot change kernel
calibration. Stage 4 consumes the completed bundle without a new fit or an
independent bandwidth override. This protocol is distinct from the quick
evaluation's query-dependent RMS scaling.

The main Cat→Dog inference path is

\[
x_{cat}
\rightarrow E_{cat}
\rightarrow m_{cat}
\rightarrow \widehat T_{cat\rightarrow dog}
\rightarrow G_{dog}
\rightarrow \hat y_{dog}.
\]

MAP target selection and conditional sampling are ablations. They return an
observed target code and test whether mean contraction is the limiting factor,
but they are not the proposed final readout.

## 8. Evaluation

Evaluation must be bidirectional and held out. Final Stage 4 and Stage 1B
diagnostics have different scopes.

### 8.1 Final Stage 4: FID, source SSIM, and inspection pairs

Run `scripts/evaluate_full_infoot.py` with
`configs/stage4_eval/fid_ssim_sit_b2.yaml`; results go to `outputs/s4_eval`.
Use all held-out validation sources in each direction, with full conditional
means, 20 sampling steps, and guidance 1.0. Report only these final metrics:

- directional FID against all real held-out target-domain RGB images,
  using Clean-FID clean Inception features;
- directional translation-versus-source SSIM against each original source
  RGB image, with mean, standard deviation, count, and per-image results.

Both source and translation use the same 256-pixel evaluation geometry.
There is no native reconstruction SSIM in this final protocol. Stage 1A
images and VAE reconstructions are not metric targets.

Save 16 deterministic source/translation pairs per direction (32 total),
with individual PNGs and labeled contact sheets. FID and SSIM use the full
evaluation corpus, not just these 16 examples. Preserve splits, counts, seeds,
and sampling settings across experiments. If a domain has fewer than 16
sources, save all and report the actual gallery count.

High source SSIM alone can reward copying, while legitimate cross-species
shape changes can lower it. FID measures distributional similarity and has
finite-sample effects. Interpret both with the saved images; neither proves
semantic correspondence. Validation data used for tuning is not a blind test.

### 8.2 Stage 1B diagnostics: reconstruction and CFG

The quick protocol fits on 512
training references per domain, projects through the complete target training
bank, and evaluates 256 validation queries per domain.

Measure Stage 1A and Stage 1B same-domain reconstruction with matched E/G pairs.
The Stage 1A baseline is E0/G0; using E0 with a changing G would confound the
reference. Report latent MSE, original-RGB pixel MSE/PSNR, LPIPS, and learned-null
behavior. Stage 1A is a diagnostic comparison, not the reconstruction target.
The training `cat_reconstruction_loss` and `dog_reconstruction_loss` are
SNR-weighted latent velocity-gap losses against targets derived from real-image
latents, noise and flow time; they are not decoded pixel errors.

### 8.3 Stage 1B diagnostics: transport and conditional geometry

Report:

- row and column marginal residuals and convergence;
- mutual information and plan entropy;
- effective target counts;
- conditional KL and expected structure cost;
- conditional-mean norm and centered variance relative to target codes;
- nearest-target distance;
- between-mean and within-conditional variance.

These diagnose the estimator but do not replace decoded-image evaluation.

### 8.4 Optional independent semantics

Use held-out proxy labels for:

- viewpoint: front, three-quarter, side, back;
- framing: close-up, medium, full body;
- coat color: black, white, gray, brown, orange, mixed.

Report P@1/5/15 where applicable and label coverage. Coat color is an
evaluation readout only; current training does not explicitly preserve color.
Unlabeled samples must not be assigned guessed values. These diagnostics are
useful research evidence but are not required outputs of current Stage 4.

### 8.5 Stage 1B diagnostic translation grids

The active Stage 1B panel saves one row of 16 conditional-mean generations
per direction. Source, MAP, sampled-target, and direct-teacher rows are omitted.
Those alternative readouts remain available as optional configured ablations;
when selected, they use the same target noise per column. Decoded DINO errors
remain teacher-agreement diagnostics, not independent evidence. Stage 4's
source/translation pair galleries are a separate final-evaluation artifact.

### 8.6 Optional diagnostic visualization

UMAP should show target-bank codes and projected-source codes with distinct,
high-contrast point types. Target dots remain translucent to reveal overlap.
Separate images are produced for barycentric and conditional projections.
Panels are colored by viewpoint, framing, or coat color and captions include
all available proxy precision values and coverage.

UMAP coordinates are nonlinear and stochastic projections of high-dimensional
features. A cluster centered inside a large target cloud may simply reflect
conditional averaging. Checkpoint selection cannot be based on the plot alone.

## 9. Decision gate for the first Experiment D run

The active pilot ends at 4,000 updates, beyond the full 2,000-step decoded
loss ramp. Continue to 5,000 or beyond only if several signals agree:

- translation quality improves while original-image reconstruction and useful
  encoder information remain acceptable; ground-truth flow loss is supporting
  evidence, not a requirement to reproduce Stage 1A predictions;
- fixed-probe conditional structure improves in both directions;
- decoded conditional means improve source structure without obvious target
  realism collapse;
- MAP/sample controls do not clearly dominate for every example;
- where labels are available, proxy viewpoint/framing/color precision supports
  the interpretation with adequate coverage;
- learned-null outputs remain stable;
- no transport or discriminator instability appears.

Training loss need not decrease monotonically because minibatches, diffusion
times/noise, plan fitting, adversarial updates, and warmups change. Fixed
validation probes and decoded examples are the primary trend evidence.

## 10. Future research ablations

The following comparisons can strengthen scientific claims; they are outside
the current Stage 4 deliverable:

1. Stage 1A encoders plus offline InfoOT;
2. earlier frozen-G Stage 1B control;
3. Experiment D without decoded losses;
4. Experiment D without generator adaptation;
5. Experiment D without matching heads;
6. conditional mean versus MAP versus sampling;
7. `h_proj` sensitivity around 0.10;
8. structure loss only versus adversarial loss only;
9. raw versus EMA checkpoint;
10. Cat→Dog and Dog→Cat;
11. original versus native-refined Stage 1A initialization, under matched Stage 1B;
12. KL + V/C alone versus added matching positive-set contrast;
13. decoded structure cosine alone versus additional structure InfoNCE;
14. G-only code InfoNCE on/off, retaining identical image losses;
15. encoder-ratio controller on/off and bounded ratio sensitivity;
16. decoded batch 4 versus 32 with fixed sample/noise evaluation and measured cost.

The latest recipe changes several mechanisms at once. These ablations are
needed to attribute any improvement to a specific loss, controller or batch size.

Do not tune `h_proj` on UMAP appearance alone. The earlier Stage 1A sweep found
0.10 near a useful geometry knee: broader values averaged many more targets but
caused stronger norm contraction and larger nearest-target distance. The
current selection should use FID, source SSIM, and decoded image inspection.
Independent semantic metrics remain a useful future extension.

## 11. Failure modes

### Incompatible semantic factors

Cat and Dog encoders may organize their codes around different attributes.
Matching heads can reshape geometry but cannot recover information absent from
the raw codes. Measure within-domain attribute predictability first.

### Conditional-mean information loss

Multimodal target conditionals can average incompatible modes. Compare the
mean with MAP and sampling, and report conditional uncertainty. A lower-variance
mean is not automatically incorrect.

### Decoder off-manifold behavior

A convex combination of valid codes need not be a code seen during Stage 1A.
Decoded training addresses this risk, but the global 512-vector may still be too
weak for precise spatial preservation.

### Generator over-adaptation

Adapters may improve translated images while harming same-domain
reconstruction or null generation. Keep the base transformer/null token frozen,
use original-image reconstruction validation and the null teacher. E0/G0
comparisons diagnose change; divergence from an imperfect Stage 1A output
is not by itself evidence of degradation. Native conditioned preservation
is disabled in the active recipe.

### Teacher shortcut

Optimizing and measuring the same frozen DINO representation can reward a
teacher-specific shortcut. Viewpoint/framing/color labels, visual inspection,
and later independent representations are necessary.

### Feature discriminator instability

The DINO feature discriminator is small but still adversarial. Its raw loss is
not a calibrated quality metric. Monitor real/fake scores, gradient norms,
generated diversity, and both domains.

### Dataset imbalance and outliers

InfoOT conditional projection is designed to average with density-ratio
weights, which can be more robust than top-k lookup. Finite KDE estimates can
still be sensitive to bandwidth, coverage, and rare regions. Report effective
targets and query-stratified results.

## 12. Claims boundary

The official InfoOT basis covers the kernelized mutual-information transport
solver and conditional density-ratio projection. This project adds:

- native Stage 1A original-RGB refinement and G-only native-code InfoNCE;
- DINO-based cross-domain structure cost;
- neural matching heads;
- joint encoder/generator co-training;
- decoded DINO structure loss;
- DINO feature discriminators;
- learned-null preservation;
- RMS-gradient correction, relational KL and matching V/C protection;
- matching positive-set contrasts and decoded structure/code InfoNCE;
- encoder-only decoded gradient balancing;
- diffusion-specific checkpoint and inference machinery.

These additions must be described as project extensions. Experiment D is a
hypothesis under evaluation, not an established improvement. CPU tests verify
mathematics, gradients, state restoration, and isolation. Only full AFHQ runs,
held-out metrics, ablations, and image review can support translation claims.
Stage 2–3/4 implementation and CPU checks do not establish full-dataset GPU
runtime, convergence, or improved translation quality; a complete AFHQ run is
still required. The latest supplied 5,000-step run disables native Stage 1A
preservation but predates the current contrastive/controller/64-native-32-decoded recipe.
Neither it nor the earlier preservation-enabled run validates the exact active
objective or the optional Stage 1A refinement.

## 13. References

- [InfoOT: Information Maximizing Optimal Transport](https://github.com/chingyaoc/InfoOT)
  — transport solver and conditional projection basis.
- [PDAE](https://proceedings.neurips.cc/paper_files/paper/2022/file/8aff4ffcf2a9d41692a805b3987e29ea-Paper-Conference.pdf)
  — semantic encoding with a pretrained diffusion model.
- [SiT](https://arxiv.org/abs/2401.08740) — stochastic-interpolant transformer
  backbone and flow formulation.
- [DINO visual descriptors](https://arxiv.org/abs/2112.05814) — frozen ViT
  descriptors used by the structural prior.
- [Splice](https://arxiv.org/abs/2311.12193) — motivation for structure-aware
  generated-image supervision.
- [ControlNet](https://arxiv.org/abs/2302.05543) — motivation for adapting
  conditioning components while preserving a pretrained backbone.
- [CycleGAN-Turbo](https://arxiv.org/abs/2403.12036) — motivation for combining
  diffusion adaptation with unpaired target-domain supervision; Experiment D
  uses a different architecture, sampler, and discriminator.

The preserved official-based project snapshot is in
`legacy/infoot_official_v1`. Historical negative or inconclusive results remain
in the tracked review documents and should be cited as controls rather than
copied back into the active method description.
