# Training Architecture and File Organization

## PDAE v2-L update (2026-10-09)

The separate RGB-conditioned experiment uses
`configs/stage1a_pdae_v2_l/{cat,dog}.yaml`: frozen SigLIP 2 supplies
`[B,196,768]` patch tokens to the shared projector and independent image
cross-attention in every SiT-L/2 block (24 blocks, hidden width 1024).
The SiT base is frozen; projector, cross-attention, learned null token, and
rank-64 attention LoRA train with real-data CosMap-weighted velocity MSE.
Mild augmentation is applied consistently to the RGB condition and freshly
encoded frozen-VAE target. There is no native-refinement loss in this variant.

The supplied cat 10k run is still improving; its config now permits 30k total
updates with the measured adapter LR `5e-5` and LoRA LR `1.25e-5`. Resume the
existing run, inspect 15k/20k images, and retain effective batch 64, EMA 0.999,
32-image validation every 1k, logging every 500 and checkpoints every 5k.
Dog stays at its existing 50k budget because no dog run was supplied.

`configs/stage1a_eval/pdae_v2_l.yaml` retains Euler-50, 16 fixed images, CFG
1/1.5/2 and correct/shuffled conditions. The new
`output.show_vae_reconstruction: false` keeps one original RGB row and six
generated rows, while preserving VAE baseline metrics. Raw and EMA grids use
the same layout; optional roundtrip grids honor the flag too. Loss smoothing
now updates every optimizer step and records `loss_ema_update_unit`; only the
old loss statistic is reset when resuming a checkpoint without that marker.

See [review, evidence, commands and remaining experiments](../docs/analysis/pdae_v2_l_cat_10000/review.md).

## Status and scope

This is the canonical implementation guide for the Cat/Dog PDAE + InfoOT
project as of 2026-09-29. It describes the active pipeline and named controlled experiments. Earlier
frozen-generator, guarded-gradient, support-loss, and matching-head-only
experiments remain available as controlled historical variants, but they are
not current recommendations.

The **revised latent-input, flow-only LoRA Stage 1A** uses
`configs/stage1a_pdae/{cat,dog}_sit_b2_lora_residual.yaml`, producing fresh
`outputs/stage1a_{cat,dog}_rescnn` checkpoints. Its residual CNN has a raw
stride-1 stem, four feature stages, two residual blocks per stage, attention
at 16x16 and a 512-code. Training settings and the latent flow loss match the
original recipe. Dedicated evaluation is `configs/stage1a_eval/residual_sit_b2_256.yaml`,
using original RGB as the image target. See the
[residual encoder protocol](../docs/analysis/stage1a_residual_encoder/review.md).
The [official PDAE source audit](../docs/analysis/stage1a_official_pdae/review.md)
pins upstream commit `4f9c29a` and distinguishes this latent-flow adaptation
from its pixel-DDPM source. Main target/sign/residual algebra checks passed.
Evaluation now honors the training-level `pretrained_config` override; unsupported
Stage 1A `loss_weighting.type` values fail instead of silently using Flow-SNR.
The current SNR curve on velocity error is an experimental weighting choice,
not an exact conversion of PDAE's epsilon objective. The Stage 1A evaluation CLI
still runs smoke/round-trip protocols only, despite reserved full-evaluation
fields in its YAML. No active training hyperparameters changed in the audit.
The [time-weighting investigation](../docs/analysis/stage1a_time_weighting/review.md)
recommends uniform velocity MSE as the first controlled comparison and CosMap
as a mild nonuniform alternative. Both are now implemented as explicit
`loss_weighting.type: uniform` / `cosmap` modes with uniform timestep sampling.
The `*_lora_residual_{uniform,cosmap}.yaml` Cat/Dog recipes differ from the SNR
control only in their loss-weighting block and output directory. The original
PDAE curve is unchanged. Logs retain unweighted flow metrics and tag the training
weighting; checkpoints reject a change of flow weighting on resume. See the
[implementation and comparison commands](../docs/analysis/stage1a_time_weighting/implementation.md).
Historical noisy-time code
gains and irreducible clean-endpoint noise make large per-bin flow MSE alone an
insufficient reason to choose a weight. Compare original-RGB fidelity and
downstream InfoOT translation under matched budgets.
The original `*_sit_b2_lora.yaml` recipes and checkpoints remain available;
existing Stage 1B runs still initialize from those original plain CNNs.
An **optional separate RGB encoder comparison** uses
`configs/stage1a_pdae/{cat,dog}_sit_b2_lora_rgb.yaml`, producing
`outputs/stage1a_{cat,dog}_rgb_lora`. The encoder reads original 256x256 RGB;
SiT still learns four-channel latent flow. It starts fresh for 50,000 updates
with the original learning rates and no DINO/refinement objectives. Existing
Stage 1B experiments retain their previous initialization;
this change does not migrate their training graph to RGB. See
[the RGB encoder protocol](../docs/analysis/stage1a_rgb_encoder/review.md).

The current Stage 1B experiment is **v6**, using
`configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml` and its matched
evaluation YAML. It retains v4.5 reference-plan fitting and adds source-aware
query selection: softmax(log(W + 1e-8) - 0.25 C_spatial/s_spatial -
0.25 C_appearance/s_appearance). Spatial costs use own-encoder relation maps;
appearance costs use 8x8 area-averaged original RGB converted to normalized Lab.
Added costs are detached. Direction-specific row-centered RMS calibration uses
only the initial training bank, is frozen/checkpointed, and does not depend on
query-batch composition. This is separate from the existing reference RMS EMA.

V6 replaces the global histogram with per-source normalized Lab patch SWD at
weight 0.04 (RGB Gaussian scales 64/32/16, patch 5, 128 random unit projections).
Source targets are detached; the private color RNG is checkpointed and validation
uses fixed seeds. Texture SWD has zero training weight and appears only under
validation `diagnostics`, with disjoint real-real reference subsets. All other
v4.5 objectives, rates, original Stage 1A initialization, PCGrad and the 10% decoded
conditional-weight gradient route are retained. Fit/projection bandwidths are
0.55/0.25 in both training and evaluation. Before/after selection costs, entropy
effective-target counts and projected-code spread are logged. No external
teacher or GAN is added. Start fresh; output roots are `outputs/stage1b_nce_v6`
and `outputs/stage1b_eval_nce_v6`. `v6_bandwidth_control` isolates projection 0.25
with all v4.5 losses retained. See the [v6 implementation](../docs/analysis/stage1b_v6/implementation.md)
for weights, resume limitations, diagnostics and test evidence.

The retained spatial-cost control is **v4.5**, using
`configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml` and its matching
Stage 1B evaluator. It retains all v4 losses/settings and adds a detached
own-encoder spatial correlation cost to reference InfoOT fitting: stages 0/1,
8x8 relation maps, diagonal exclusion, symmetric row-cosine distance, alpha 0.20.
Fixed initial training-bank centered cost spreads calibrate the mixture; scales
and IDs are saved separately from projection RMS EMA and never updated on
queries. Native latent inputs and the original flow-only Stage 1A checkpoints
are retained. No teacher, GAN or new neural spatial loss is added.

Outputs are `outputs/stage1b_nce_v4_5` and `outputs/stage1b_eval_nce_v4_5`.
The evaluator requires a compatible checkpoint and uses its selected raw/EMA
encoder maps. A saved `step_000000.pt` supplies the initial calibrated reference;
`--quick-eval` evaluates it and the final state after training. Enabled-only
spatial diagnostics and source/top-reference/output grids support the comparison.
See [implementation and tests](../docs/analysis/stage1b_v4_5/implementation.md).
The existing v4 files remain the matched control; GPU quality gains are untested.

The retained Stage 1B **v4 image-loss control** is in
`configs/stage1b_infoot/self_supervised_infonce_v4_sit_b2.yaml` with output
`outputs/stage1b_nce_v4` and a matching v4 standalone evaluator. It adds four
teacher-free image terms to v3: coarse original-source RGB 0.02, source RGB-uv
histogram 0.02, fixed local self-similarity layout 0.02, and unpaired target
Laplacian-patch SW1 0.01. No own-feature SWD/EMA reader or spectral penalty is
enabled. The source terms use original RGB; real target images come from
disjoint training references. The 0.10 conditional-weight gate applies to the
whole decoded branch, with unchanged forward conditions and live direct
generator/reference-code derivatives. PCGrad retains one decoded task
(`decoded_images`), with separate component diagnostics. Sampling state and
loss definitions are checkpointed; incompatible resumes/evaluation are rejected.
The v3 hyperparameters and 2,500-step pilot duration are retained. Schema 4
logs each enabled term, weights, per-loss gradient norms/cosines, target IDs,
and validation real-real SWD where sufficient references exist. These additions
are implemented experiments, not evidence of improved AFHQ image quality.
See [design and references](../docs/analysis/stage1b_v3_2500/loss_design.md).

The retained **v3 control** uses global MLP InfoNCE with 10% gradient through
conditional weights**, following the `results/v2/` rank-collapse review:

- training: `configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml`,
  output `outputs/stage1b_nce_v3`;
- paired control: `self_supervised_infonce_v3_control_sit_b2.yaml`, output
  `outputs/stage1b_nce_v3_g1`; only the output and W-gradient factor (1.0) differ;
- evaluator: `configs/stage1b_eval/self_supervised_infonce_v3_sit_b2.yaml`,
  224 fit references and 224 projection targets, conditional mean;
- losses: native flow 1.0 per domain, MI-only neural alignment (outer 0.20,
  inner MI 0.10), relative 0.01, variance 0.10, covariance 0.30, and
  bidirectional global MLP InfoNCE 0.05; PCGrad enabled;
- matching std floor 0.80; encoder/matching-head LR 1e-5/1e-5,
  adapter/LoRA LR 5e-6/3.75e-6, separate contrastive MLP LR 2e-4;
- batches transport/native/decoded = 256/32/32; fit/projection bandwidth
  0.55/0.10; reference-RMS EMA 0.99 and model EMA 0.995;
- 2,500-step fresh run, save/validation every 250, gradient diagnostics every
  100; original flow-only plain-CNN Stage 1A EMA initialization.

The inner solver remains fused own-encoder InfoOT with cost/MI/entropy
coefficients 1.0/0.10/0.02. Only the live neural cosine-cost gradient is disabled.
For decoded InfoNCE, `W_dec = stopgrad(W) + rho * (W - stopgrad(W))`, rho=0.10,
then `z_target = W_dec @ live_target_codes`. Forward conditions are unchanged;
InfoNCE gradients through W become 10%. Direct target-code, generator, image
readout and contrastive-MLP gradients remain live. Shared encoder totals change
through the W path. MI/relative/variance/covariance keep their original graphs.

`source_contrastive_projection_gradient_scale` is logged in decoded train/val
metrics and saved in checkpoint metadata; defaults to 1.0 for old runs. Resume
and evaluation reject a mismatched recipe. Gradient diagnostics expose the MI
term separately when MI and relative alignment are both enabled. Existing v2
and PatchNCE configs are retained; do not resume collapsed v2 heads into v3.
CPU routing/integration tests have passed; image-quality gains remain untested.
See [implementation and commands](../docs/analysis/stage1b_v2_5000/implementation.md).

The established Stage 1B baseline is **self-supervised PatchNCE with learned MLP
samplers and EMA-tracked reference RMS**:

- training/evaluation: `self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml` under
  `configs/stage1b_infoot/` and `configs/stage1b_eval/`;
- output: `outputs/stage1b_patch_mlp_rms`;
- losses: native flow, own-encoder InfoOT MI, matching variance/covariance,
  bidirectional PatchNCE; PCGrad enabled;
- initialization: original flow-only, latent-input Stage 1A checkpoints;
- Stage 2–3 / 4 defaults still use this experiment and its saved RMS history,
  pending a measured v3 quality comparison.

The external-supervision **Experiment D** remains a separate control:

- config: `configs/stage1b_infoot/structure_decoder_sit_b2.yaml`;
- evaluation: `configs/stage1b_eval/structure_decoder_sit_b2.yaml`;
- implementation reference: `docs/stage1b_experiment_d.md`;
- output: `outputs/stage1b_patch_color`.

The active entropy epsilon is now 0.02 for Stage 1B training/evaluation and
Stage 2–3 fitting. Earlier log analyses describe their original settings; this
change has not yet been validated by a new training run. Stage 4 inherits the
new entropy through a newly fitted bundle, not by overriding an existing one.

The Experiment D recipe has six task groups: native flow, InfoOT/conditional
alignment, matching variance/covariance protection, original-source perceptual
InfoNCE, target RGB PatchGAN realism and HistoGAN source palette. PCGrad is enabled separately on E/H/G; the old
ratio controller is off. Latent anchoring, null/conditioned Stage 1A teacher
preservation, neighborhood KL, matching contrasts, decoded structure InfoNCE
and generated-code InfoNCE are disabled. Both cosine losses stay off.
See [the RGB/color experiment](../docs/analysis/stage1b_patchgan_histogan/review.md),
[the preceding simplification](../docs/analysis/stage1b_minimal_pcgrad/review.md)
and [current loss equations/graph](../docs/stage1b_current_losses.md).
The preceding full recipe remains in structure_decoder_full_sit_b2.yaml.

### Controlled experiment without an external feature teacher

`configs/stage1b_infoot/self_supervised_sit_b2.yaml` is a separate experiment
using the **original flow-only latent-input Stage 1A**
`*_sit_b2_lora.yaml` EMA checkpoints. Their original configs and checkpoint
paths are unchanged; the new RGB experiment uses separate `*_lora_rgb.yaml` files.
It replaces the DINO transport cost with own matching-feature cosine cost and
uses bidirectional generated-source InfoNCE: generated dog RGB is VAE-reencoded
and read by the live dog PDAE encoder to retrieve the detached original cat
code; the reverse direction trains the cat readout. Both encoders can therefore
adapt toward shared coordinates. Native flow and intrinsic V/C protection
remain, with the existing PCGrad and optimizer settings. DINO, adversarial,
histogram, teacher KL and initial Stage 1A anchor losses are disabled.

The controlled translation coefficient is now **0.15**, ramped over 2,000
steps; outer alignment remains 0.20 and inner MI 0.10. Its schema-3 train/val
logs exclude disabled teacher/critic/anchor terms. Validation adds original-RGB
source correspondence proxies and native reconstruction MSE/PSNR on 32 fixed
queries per domain, with separate native image panels. These are read-only
checks, not extra objectives or independent target-realism measurements.
Query-dependence, isolated InfoOT gradient conflicts and solver budget/residual
diagnostics support the next tuning decision. Details and exact fields:
[controlled diagnostics](../docs/analysis/stage1b_self_supervised/diagnostics.md).

The matched evaluator is `configs/stage1b_eval/self_supervised_sit_b2.yaml`.
Outputs are `outputs/stage1b_self` and `outputs/stage1b_eval_self`. Run it fresh;
do not resume the external-supervision checkpoint with this changed objective.
See [the experiment protocol](../docs/analysis/stage1b_self_supervised/review.md)
for checkpoint provenance, code-space assumptions, commands and evaluation
criteria. No GPU quality result has yet established this experiment as an
improvement. The pretrained SiT/VAE remain; "no external supervision" refers
to the additional Stage 1B objectives.

### Controlled spatial PatchNCE experiment

`configs/stage1b_infoot/self_supervised_patchnce_sit_b2.yaml` replaces the
global source-code contrast in the preceding self-supervised experiment with
CUT's official spatial PatchNCE. The source encoder provides detached conv-stage
maps, while generated target RGB is VAE-reencoded and read by the live target
encoder. Corresponding sampled locations are positive and other locations of
the same source image are negative. The two directions train both encoders.

It uses conv stages `[0,1,2]`, up to 64 patches per map, temperature 0.20 and
weight 0.15 ramped over 2,000 steps. CUT's no-MLP sampler is used with private
checkpointed RNG; the official loss is vendored unchanged at a pinned revision.
Flow-only Stage 1A initialization, native flow, own-feature InfoOT, V/C,
PCGrad, learning rates and fit/projection 0.55/0.10 match the global-code control.
No global-code loss, external teacher or critic is added. Logs report PatchNCE
spatial retrieval/collapse rather than inactive global retrieval, and retain
fixed original-RGB/native checks and gradient/solver diagnostics.

Start a new run in `outputs/stage1b_self_patch`; use the matched evaluator
`configs/stage1b_eval/self_supervised_patchnce_sit_b2.yaml` for output
`outputs/stage1b_eval_self_patch`. This is an experimental spatial correspondence
constraint, not a reproduction of the full CUT recipe or a proven quality gain.
See [protocol, graph and commands](../docs/analysis/stage1b_self_patchnce/review.md).

**Main learned-sampler / EMA-RMS experiment (2026-09-25).** The
`self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml` training/evaluation pair adds
CUT-style per-layer MLP samplers to the EMA-RMS experiment. Each domain has its
own Linear(C,256)-ReLU-Linear(256,256) head per spatial layer, followed by L2
normalization. As in DCLGAN's two-sampler routing, cat-to-dog compares a live
dog encoder/dog sampler query with detached cat encoder/cat sampler keys;
the reverse direction trains the cat sampler. The key MLP is detached too.
These modules are separate from InfoOT's global residual matching heads.

Sampler LR is 2e-4 with separate clipping at 1.0, eager optimizer registration,
a distinct PCGrad group, raw/EMA checkpoint state and matched evaluation loading.
PatchNCE weight 0.15, temperature 0.20, patches/layers, all other losses,
E/G learning rates and EMA-RMS settings remain fixed. Training validation uses
raw samplers; the standalone evaluator loads raw or EMA consistently and scores
the same generated images in the grid. Learned retrieval is not independent
visual-quality evidence. See [implementation and comparison protocol](../docs/analysis/stage1b_patchnce_mlp/review.md).

Start fresh from the same original flow-only Stage 1A checkpoints, output
`outputs/stage1b_patch_mlp_rms`. The additional
`self_supervised_patchnce_mlp_sit_b2.yaml` pair uses live RMS and output
`outputs/stage1b_patch_mlp` for comparison against the older no-MLP control.
The original no-MLP recipes/checkpoints remain supported; new MLP checkpoints
are required when selecting `sampler: mlp_sample`.

The [RElbers InfoNCE audit](../docs/analysis/infonce_reference_audit/review.md)
independently verified PatchNCE values/query gradients without changing its
official CUT implementation. The shared global InfoNCE helper now uses
cross-entropy to avoid loss cancellation for confident FP32 predictions;
key detachment, candidate filtering and all recipe coefficients are preserved.

Current entry points and outputs:

The latest supplied 5,000-step log review is
`docs/analysis/stage1b_v2_5000/review.md`: matching rank fell to about 4 despite
trace around 0.65, while raw encoder rank stayed around 49–50. Decoded InfoNCE's
matching-head gradient was much larger than the covariance gradient. The
standalone all-bank mean also differed substantially from training support.
These motivate v3; they do not establish its best coefficients. Earlier log
reviews below describe historical runs, not validation of v3.

| Stage | Configuration | Output |
| --- | --- | --- |
| 1A: revised residual latent encoder, flow-only LoRA | `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_residual.yaml` | `outputs/stage1a_{cat,dog}_rescnn` |
| 1A: residual encoder evaluation | `configs/stage1a_eval/residual_sit_b2_256.yaml` | within the selected training run, `stage1a_rescnn_eval` |
| 1A: original latent encoder / current Stage 1B initialization | `configs/stage1a_pdae/{cat,dog}_sit_b2_lora.yaml` | `outputs/stage1a_{cat,dog}_sit_b2_cfg_adaln_all_lora_r64` |
| 1A: optional RGB encoder comparison, flow-only LoRA | `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_rgb.yaml` | `outputs/stage1a_{cat,dog}_rgb_lora` |
| 1A: historical flow + DINO experiment | `configs/stage1a_pdae/{cat,dog}_sit_b2_dino.yaml` | `outputs/stage1a_{cat,dog}_dino` |
| 1B: v4, four fixed-image losses | `configs/stage1b_infoot/self_supervised_infonce_v4_sit_b2.yaml` | `outputs/stage1b_nce_v4` |
| 1B: v4 matched evaluation | `configs/stage1b_eval/self_supervised_infonce_v4_sit_b2.yaml` | `outputs/stage1b_eval_nce_v4` |
| 1B: v3 global MLP InfoNCE, W gradient 0.10 | `configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml` | `outputs/stage1b_nce_v3` |
| 1B: v3 W-gradient 1.0 control | `configs/stage1b_infoot/self_supervised_infonce_v3_control_sit_b2.yaml` | `outputs/stage1b_nce_v3_g1` |
| 1B: v3 paired evaluation | `configs/stage1b_eval/self_supervised_infonce_v3_sit_b2.yaml` | `outputs/stage1b_eval_nce_v3` |
| 1B: established learned PatchNCE + reference RMS EMA baseline | `configs/stage1b_infoot/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml` | `outputs/stage1b_patch_mlp_rms` |
| 1B: PatchNCE baseline evaluation | `configs/stage1b_eval/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml` | `outputs/stage1b_eval_patch_mlp_rms` |
| 1B: external Experiment D control | `configs/stage1b_infoot/structure_decoder_sit_b2.yaml` | `outputs/stage1b_patch_color` |
| 2–3: Frozen Bank Construction and Global InfoOT Fitting | `configs/stage23_offline/full_sit_b2.yaml` | `outputs/s23_rms` |
| 4: final translation evaluation | `configs/stage4_eval/fid_ssim_sit_b2.yaml` | `outputs/s4_rms` |

Stage 2–3 is also called **Offline Alignment**. One entry point performs
Stage 2 bank construction and Stage 3 fitting; a separate bank-building run
is not required. See `docs/stage23_stage4_full_dataset_plan.md`.

The Experiment D recipe follows the supplied fit-0.55 pilot with V/C weights 0.05/0.01.
The current LoRA LR trial increases 2.5e-6 to 3.75e-6 while retaining adapter
LR 5e-6. Native Stage 1A prediction preservation is
disabled: reconstruction quality is measured against ground truth. It is a
fresh 4,000-step pilot with matched 0.55/0.10 evaluation.
Use `structure_decoder_fit055_control_sit_b2.yaml` to resume the previous
recipe at its original `outputs/stage1b_d_fit050` directory. Do not resume that
checkpoint with the changed objective. The original research record is
`docs/analysis/stage1b_fit055_v2/review.md`, corrected by
`docs/analysis/stage1b_ground_truth/review.md`; AFHQ improvement is unverified.

The supplied `logs-revised` run reaches train 1,700 / validation 1,500 and
still used native preservation at 0.05. The matched comparison in
`docs/analysis/stage1b_revised_vs_original/review.md` finds improved matching
floor coverage and mixed decoded results; it does not isolate the LR effect.
V/C weights remain fixed during the modest LoRA LR trial. Native flow validation uses
32 samples/domain; standalone reconstruction uses 32 images and translation
panels use 16 conditional generations per direction.
The earlier four-image decoded probe is historical; the current YAML decodes
32 queries/direction for training and fixed validation. See
`docs/analysis/stage1b_revised_vs_original/config_revision.md` for protocol changes.

The current revision uses `matching.distance_scale_gradient: full` in neural
losses plus VICReg-inspired matching variance/covariance protection and
`semantic_prior.neighborhood_geometry: rms_distance`. The RMS-only follow-up
still lost about 90% of matching-head spread; the new neighborhood objective
removes a separate cosine-logit contraction incentive. The protected cosine
control is `structure_decoder_vicreg_cosine_sit_b2.yaml` in the train/eval
folders and retains `_rmsgrad_vicreg` outputs. Changing the neighborhood mode
or temperature requires a fresh run. The
corrected-RMS-only control is `configs/stage1b_infoot/structure_decoder_rmsgrad_sit_b2.yaml`,
with a corresponding evaluation YAML of the same name. The original detached-scale D config remains at
`configs/stage1b_infoot/structure_decoder_detached_sit_b2.yaml`. Changing this
gradient rule requires a fresh Stage 1A initialization, not resume. The review
and numerical reproduction are in `docs/analysis/stage1b_experiment_d_2600/review.md`.
Active D also requires both marginal and outer convergence. The outer update
budget is 1,200 with early stopping; the original control keeps its 100-step cap.
This cap was raised from 300 after a feasible plan needed more outer updates.
Increasing only this cap is resume-compatible when the saved run already
required outer convergence and all other solver settings match. Training,
fixed validation and evaluation report the cap alongside actual iterations.
The protection definition and rerun protocol are in
`docs/analysis/stage1b_experiment_d_vicreg.md`.
The follow-up evidence, GitHub audit, new mechanism check, and comparison
protocol are in `docs/analysis/stage1b_rmsgrad_1500/review.md`.

Scientific motivation, hypotheses, and equations are in
`notes/research_idea.md`. This document owns model construction, data flow,
training stages, configuration, checkpoints, evaluation, and file boundaries.

## 1. End-to-end stages

The current pipeline is:

1. **Stage 0 — data and feature preparation**
   - create deterministic AFHQ Cat/Dog train/validation manifests;
   - cache VAE latents for the diffusion target and retain original-image source metadata;
   - cache DINOv2 descriptors only for the external-teacher Stage 1B variants.
2. **Stage 1A — separate domain PDAE training**
   - train Cat and Dog semantic encoders and conditioning paths separately;
   - use learned-null semantic CFG and rank-64 LoRA in all 12 SiT blocks;
   - original latent-input recipes/configs/checkpoints remain main and unchanged;
   - an optional comparison encodes original RGB while training latent flow only;
   - historical DINO/refinement recipes remain separate experiments.
3. **Stage 1B — self-supervised joint co-training**
   - train both encoders and matching heads;
   - adapt only the added conditioning components of both generators;
   - decode full InfoOT conditional means using EMA-tracked reference RMS;
   - train bidirectional spatial correspondence with domain-specific PatchNCE MLPs,
     alongside native flow, InfoOT MI and matching variance/covariance protection.
4. **Stage 2 — freeze the selected checkpoint and build complete banks**
   - encode reference and projection splits with the selected paired weights;
   - store raw decoder codes and matching-head features separately.
5. **Stage 3 — fit offline InfoOT and package conditional projection**
   - fit one global transport over every training Cat and Dog, using 0.55 fit bandwidth;
   - export marginals, transport, full banks, separate fit/projection RMS scales,
     paired E/head/G/samplers and saved RMS histories, and 0.10 projection bandwidth;
   - combined implementation for Stages 2 and 3: `scripts/run_offline_alignment.py`.
6. **Stage 4 — translate unseen images and perform final evaluation**
   - encode the source image;
   - compute the full conditional expectation over the target bank;
   - decode the projected raw target code with the selected target generator;
   - evaluate every validation source with directional FID and translation-versus-source SSIM;
   - save 16 source/translation pairs per direction, with fixed selection seeds;
   - implementation: `scripts/evaluate_full_infoot.py`; Linux commands are in README.
7. **Optional controlled evaluation (historical Stage 5)**
   - compare the conditional mean with MAP, sampling, and teacher controls;
   - report reconstruction, geometry, independent proxies, and decoded images.

Stages 1B through 4 must use one compatible encoder/head/generator tuple. A
Stage 1B checkpoint cannot be evaluated or exported by silently substituting a
Stage 1A generator.

The combined workflow is named **Stage 2–3: Frozen Bank Construction and Global
InfoOT Fitting**, shortened to **Stage 2–3: Offline Alignment**. Stages 2 and 3
remain distinct conceptual steps, executed by one entrypoint. Its config is
`configs/stage23_offline/full_sit_b2.yaml`, stage identifier is
`stage23_offline_alignment`, Python function is `run_stage23`, and default
output directory is `outputs/s23_rms`. Stage 4 reads that directory by default.

## 2. Backbone and domain branches

The first backbone is a latent SiT-B/2-style flow model:

```yaml
input_resolution: 256
latent_shape: [4, 32, 32]
patch_size: 2
num_layers: 12
hidden_size: 768
num_attention_heads: 12
prediction_type: velocity
```

The practical pretrained source is `BiliSakura/SiT-diffusers`. The VAE maps
images to diffusion latents. Each domain branch contains:

- a convolutional semantic encoder `E_d` producing a 512-vector with learned
  LayerNorm (`normalize_z: true`), not a unit-L2 decoder code;
- a learned null semantic token for classifier-free guidance;
- semantic `z_proj` and token MLPs;
- AdaLN-Zero conditioning injected into all 12 transformer blocks;
- attention LoRA in all 12 blocks, rank/alpha 64/64;
- the shared pretrained SiT architecture and VAE.

The new RGB Stage 1A `E_d` reads RGB in [-1,1]. Six strided convolution stages
reduce 256x256 to 4x4 before the final projection to 512 dimensions. Historical
Stage 1A and the current Stage 1B variants instead retain the three-stage
encoder over four-channel 32x32 VAE latents. The common class name
`PDAELatentEncoder` is kept for import/checkpoint compatibility; the explicit
`encoder.input_space` selects its input contract.

Stage 1A trains each domain branch independently. Experiment D begins with the
Stage 1A EMA state and keeps original SiT weights, VAE weights, DINOv2 weights,
and the learned null token fixed. It updates the semantic encoder, matching
head, added AdaLN/token conditioning layers, `z_proj`, final adapter, and
attention LoRA.

The target decoder always consumes the raw semantic code. The matching head is
used to define InfoOT geometry; its output is not sent to the generator.

## 3. Data contract

The dataset configuration is `configs/data/afhq_huggan.yaml`. Train and
validation identities must remain disjoint. Data preparation assigns stable
sample IDs so descriptor banks, proxy labels, reports, and cached latents can be
checked against the same examples.

RGB Stage 1A sets `CachedLatentDataset(include_original_images=True)` and receives
both `encoder_image` (float32 RGB in [-1,1]) and the existing `x0_latent` target.
Original pixels are loaded lazily from the metadata-checked HF source using
the same crop/resize as latent caching. One flip draw is shared by RGB and its
cached latent. Missing source identities fail; decoded VAE images are never a
substitute encoder input. Each worker reuses its source dataset and parsed
configuration. The original HF dataset must remain available for these runs.

Stage 1A may use random horizontal flips as configured. The current DINO
structure cache is computed from unflipped images, so Experiment D sets
`random_horizontal_flip: 0.0`. Changing preprocessing requires a new descriptor
cache and a new output directory.

Experiment D traverses the full shuffled training split. Per domain, each
update draws 128 samples:

- 96 references fit the current transport plan;
- 32 different samples serve as unseen conditional queries;
- the first 64 reference samples also enter same-domain reconstruction and
  null-preservation losses (these are not 64 additional images);
- all 32 reserved queries per direction enter decoded image losses.

The previous supplied logs used reconstruction batch 16 and decoded batch 4.
The current 64-native/32-decoded recipe is a changed experiment: it increases decoded
rollout work eightfold at fixed sampling steps, with potentially substantial
activation memory and wall-time cost. It has not been validated by those logs.
A query batch size is shared by decoded training and its fixed validation; the
standalone translation gallery has a separate 16-image setting.

These are rotating minibatches, not a fixed 128-image subset.

The structure cache at
`data/semantic_priors/afhq_dinov2_structure.pt` must include sample IDs, split
and domain membership, DINO model/revision/processor metadata, descriptor
definition, grid size, calibration, and a content fingerprint. The trainer
rejects missing or incompatible metadata.

Optional proxy labels are read only by evaluation:

```text
data/proxy_labels/afhq_viewpoint_framing.jsonl
```

Each line is a JSON object keyed by `sample_id`, with any available
`viewpoint`, `framing`, and `coat_color` values. This file may be ignored by Git
because it is local annotation data. Zero rows or unmatched IDs yield zero
coverage; the path itself does not create labels.

## 4. Stage 1A training and native refinement

### 4.1 Main latent-input training and optional RGB comparison

New latent training uses `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_residual.yaml`.
The original `*_sit_b2_lora.yaml` recipes remain available for old checkpoints.
Optional comparison recipes are `configs/stage1a_pdae/cat_sit_b2_lora_rgb.yaml` and
`configs/stage1a_pdae/dog_sit_b2_lora_rgb.yaml`. Cat and Dog train independently.
These contain no refinement block or DINO loss. The original latent-input
`*_sit_b2_lora.yaml` files and checkpoint locations remain unchanged.

| Setting | Residual / original latent / optional RGB flow-only (same settings) | Historical latent flow + DINO recipe |
| --- | ---: | ---: |
| Optimizer updates | 50,000 | 40,000 from fresh E/conditioning initialization |
| Encoder / adapter / LoRA LR | 1e-4 / 1e-4 / 2.5e-5 | 1e-4 / 1e-4 / 2.5e-5 |
| Native batch / accumulation | 64 / 1 | 64 / 1 |
| Horizontal flip / semantic dropout | 0.5 / 0.10 | 0.5 / 0.10 |
| EMA decay / warmup | 0.9999 / 1,000 | 0.995 / 200 |
| Validation / checkpoint interval | 1,000 / 1,000 | 500 / 1,000, numbered candidates retained |
| Clip norm | 1.0 globally | 1.0 separately for E and G |

Both use learned-null semantic CFG, all 12 AdaLN/attention-LoRA blocks, and
LoRA rank/alpha 64/64. The SiT base stays frozen. E, added conditioning layers,
LoRA and the learned null token train. CFG scale zero uses the learned null
branch; a numeric zero vector is a different control. Old non-CFG checkpoints
are incompatible with active Stage 1B.

The primary loss is real-data flow reconstruction with coefficient 1:
`x_t = t*x0 + (1-t)*noise`, target velocity `x0-noise`. The velocity-gap loss
compares the learned residual to the detached difference between that target
and the frozen base prediction. Weighting remains `pdae_flow_snr`, gamma 0.1,
fixed-uniform normalization to mean 1, minimum weight 0.001. This is a latent
velocity MSE, not decoded RGB error.

The residual latent encoder (`encoder.kind: residual_cnn_v1`) processes
`4x32x32 -> 64x32x32 -> 128x16x16 -> 256x8x8 -> 256x4x4 -> z[512]`.
The stem is Conv3x3 stride 1 without preceding GroupNorm/SiLU. Each stage has
two pre-activation GN-SiLU-Conv3x3 residual blocks; channel changes use a 1x1
shortcut. Stride-2 convolutions separate stages. One four-head residual spatial
attention block follows the blocks at 16x16. Hidden GN uses 32 groups, dropout
is zero, and final GN-SiLU/flatten/linear/LayerNorm retains the 4x4 layout.
Spatial feature indices 0/1/2/3 refer to post-block/post-attention maps at
32/16/8/4; these differ from the original plain CNN's 16/8/4 feature indices.

Residual model states use format 5 with the complete `encoder_architecture`
specification. Load validates it before applying parameters, including
shape-invisible settings such as GN groups, head count and dropout. The outer
training checkpoint remains format 3. Plain/RGB model states remain format 4;
historical plain checkpoints without architecture metadata retain their old
interpretation. Architecture changes require fresh Stage 1A training. Training
and flow-validation logs identify `encoder_kind`; decoded evaluation and
round-trip reports record the full specification. Legacy Stage 1B provenance
remains unchanged. The existing Stage 1B recipes still select the original
checkpoints and feature-stage definitions; switching to residual checkpoints
is a separate experiment.

For the optional RGB recipe the two inputs are explicitly separate:

```text
original RGB [-1,1], 3x256x256 -> trainable semantic encoder -> z[512]
cached VAE latent, 4x32x32 + sampled noise/time -> x_t -> adapted SiT(z)
                                                     -> latent velocity-gap MSE
```

The RGB encoder widths are `[64,128,256,256,256,256]`, with stride two at each
convolution; final pooling/projection uses 4x4 and LayerNorm on the 512-code.
The first convolution reads raw normalized RGB without input GroupNorm/SiLU.
In particular, it avoids per-channel GroupNorm removing source channel means
before the first learned filter. Subsequent normalization/activation stages
remain. The backbone diffusion, target, time weighting, CFG, adapter and LoRA
designs are unchanged. RGB uses six convolution stages and a raw RGB stem,
versus the original three-stage normalized latent stem. This is a controlled
recipe comparison with an architecture adaptation, not an equal-architecture
test proving that any quality change comes from RGB input alone or a
reproduction of the original pixel-diffusion PDAE architecture.

Fresh RGB branches use model-state format 4 (inside the existing training checkpoint) with an explicit
`encoder_input: {space: rgb, image_size: 256}` contract. A missing contract
denotes historical latent input. Loading a latent checkpoint into the RGB
branch is rejected; equal 512-code dimensions do not make the architectures
compatible. Resume only checkpoints produced by the same new RGB experiment.

### 4.2 Historical native refinement objectives and routing

The separate flow+DINO recipes are `configs/stage1a_pdae/{cat,dog}_sit_b2_dino.yaml`.
The older `*_refine.yaml` configs preserve the low-LR DINO+InfoNCE run. See
[the new recipe record](../docs/analysis/stage1a_dino_recipe/review.md).
For a native condition `z = E(x0)`, sample the final diffusion latent
`xhat0 = G(z, noise)` and decode `xhat = VAE.decode(xhat0)`.

| Additional objective | Full weight | Gradient destination |
| --- | ---: | --- |
| Frozen-DINO perceptual cosine: half CLS distance plus half mean corresponding patch-token distance | 0.05 | E and G |
| Frozen-DINO pooled patch self-similarity cosine | 0.01 | E and G |
| Native-code InfoNCE (legacy option) | 0 | Disabled, recovery forward/backward skipped |

Both image targets are **original dataset RGB**, using cache crop/resize and
exactly the latent batch's recorded horizontal flip. VAE reconstructions are
separate diagnostics, never a substitute target. Missing source metadata fails.
DINO weights/revision/preprocessing come from the pinned structure cache.
VAE and DINO weights are frozen while generated-input derivatives remain live.

That image-supervised recipe computes no native-code retrieval or code loss.
Logs expose zero weighted code loss/gradient norm, null `code_contrastive`, and
`code_gradient_routing: disabled`. Positive-weight legacy configs retain the
original detached-readout and G-only routing. Both image losses still update
E and G. RGB errors are diagnostics; the image objectives operate on DINO
features, not direct pixel differences.

Decode 32 samples every four optimizer updates beginning at step 1, using
20 midpoint-time Euler steps, CFG 1.0, and checkpointed sampler/VAE/DINO.
Auxiliary weights ramp over 2,000 updates. With gradient accumulation, auxiliaries
run once using the last microbatch, not once per microbatch, and are neither
divided by accumulation nor multiplied by the sparse interval. No Stage 1B
encoder-ratio controller, discriminator, matching head, or transport is used.
Branch eval mode is kept through auxiliary checkpoint recomputation and then
restored; ordinary flow training retains semantic dropout.

### 4.3 Initialization, resume, validation and promotion

```bash
# Revised latent-input residual CNN: fresh runs:
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora_residual.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora_residual.yaml --device cuda:1

# Original plain-CNN baseline / current Stage 1B source (unchanged):
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora.yaml --device cuda:1

# Optional RGB comparison: fresh runs, no --resume/initialize_from:
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora_rgb.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora_rgb.yaml --device cuda:1

# Historical alternative: fresh 40,000-update flow + DINO training; no --resume:
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_dino.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_dino.yaml --device cuda:1
```

The residual runs use `outputs/stage1a_{cat,dog}_rescnn`, 50,000 updates,
and no old checkpoint initialization. Evaluate with the corresponding residual
training YAML plus `configs/stage1a_eval/residual_sit_b2_256.yaml`; resume only
within the matching residual architecture. Standalone image metrics use
original RGB, while semantic encoding and native flow targets use VAE latents.

The separate RGB runs use new `outputs/stage1a_cat_rgb_lora` and
`outputs/stage1a_dog_rgb_lora` directories, `max_steps: 50000` and no old
checkpoint initialization. Their periodic validation remains the native flow
and conditioned/shuffled/null-code probe. Standalone
`scripts/evaluate_pdae_domain.py` obtains RGB conditions from original images
for RGB branches. The opt-in `configs/stage1a_eval/rgb_vs_latent_sit_b2_256.yaml`
scores both latent-input and RGB-input branches against original RGB, with a
separate VAE row and `stage1a_rgb_comparison` report subdirectory. Use this same
config, checkpoint step, weight mode, samples, noise and guidance for both
branches. The original `sit_b2_256.yaml` evaluator config remains unchanged.
Reports record the reference type, so historical VAE-reference scores cannot
be compared as the same original-image metric.
Matched commands are in [the RGB protocol](../docs/analysis/stage1a_rgb_encoder/review.md).

The following refinement details describe the historical alternative, not the
new RGB flow-only run. The native-supervision configs set `train.initialize_from: null` and
`train.resume_from: null`, with `max_steps: 40000`. E and trainable conditioning
layers start fresh, as do optimizer, EMA history, counters and auxiliary RNG.
The pretrained SiT/VAE are still loaded. Outputs are
`outputs/stage1a_{cat,dog}_dino`. Adding `--resume` explicitly continues that
new run and takes precedence over initialization; it restores optimizer,
EMA, counters and auxiliary noise. A changed refinement objective or cache
fingerprint is rejected. The shuffled-loader permutation/cursor is not saved,
so resume does not guarantee uninterrupted batch order or identical weights.

Validation at step 0 and every 500 updates uses EMA, existing native flow and
correct/shuffled/null-code probes, plus 32 fixed native generations at
50 steps and seed 20260918. The decoded probe uses the first validation batch
(32 images, all decoded); the flow probe uses two batches of 32. The previous
16-image loader capped decoded validation at 16, despite requesting 32. This
new panel/noise grouping requires matched re-evaluation of controls.
`validation/native_XXXXXX.png` has original-RGB / VAE / generated-native rows.
Original-RGB MSE/MAE/PSNR, DINO fidelity and latent MSE are logged; code retrieval is disabled.
Validation preserves training RNG. `logs/train.jsonl` keeps `loss`/`loss_ema`
as flow-only metrics and adds `flow_loss`/`total_loss`; `logs/refinement.jsonl`
records every auxiliary update even between regular logs.

Inspect fixed validation through 40,000 updates against original RGB, useful code dependence,
encoder spectrum, null behavior and actual images. DINO losses are training
proxies and 32 images are a screening panel. The recipe is implemented
and CPU-tested, not established as an AFHQ quality improvement.

To test refined initialization in Stage 1B, select both domain checkpoints,
change each `stage1a.<domain>.config` and `.checkpoint` to its refinement recipe
and selected checkpoint, keep `weights: ema`, and start a new output directory.
The external-supervision Experiment D YAML loads the Cat/Dog Stage 1A DINO EMA checkpoints.
Self-supervised global InfoNCE and PatchNCE instead remain pinned to the original
latent-input flow-only configs and checkpoints. RGB Stage 1B needs a separate
input-pipeline/decoded-readout migration and new paired initialization; do not
replace their checkpoint paths alone.
Rebuild downstream frozen banks/transport after selecting a different paired
model. Detailed implementation: `docs/stage1a_refinement.md`.

## 5. Stage 1B Experiment D

### 5.1 Trainable and frozen parameters

Trainable:

- Cat and Dog semantic encoders;
- Cat and Dog residual MLP matching heads;
- added AdaLN/token MLP conditioning components;
- `z_proj` and final semantic adapter;
- rank-64 attention LoRA;
- two target-domain RGB PatchGAN discriminators (70x70 receptive field).

Frozen:

- pretrained base SiT transformer weights;
- VAE;
- DINOv2 structure network;
- Stage 1A learned null token;
- fixed Stage 1A generator teacher used for null preservation; native
  conditioned prediction preservation is disabled (weight 0.0).

The code selects trainable generator components explicitly in
`src/diffusion_ot/models/generator_adaptation.py`. This prevents accidental
unfreezing of the base model.

### 5.2 InfoOT geometry

The head consumes raw LayerNorm codes and returns
`L2_normalize(L2_normalize(z) + residual_MLP(z))`; its residual starts at zero,
so initialization is normalized identity. Decoder codes retain their raw scale. The current fused solver uses:

```yaml
fit_bandwidth: 0.55
mi_weight: 0.10
entropy_epsilon: 0.02
inner_iterations: 1200  # Outer update cap, despite the historical key name.
recovery_iterations: 4800  # Extra outer updates only after failure; FP64, tighter inner accuracy.
outer_tolerance: 0.00001
require_outer_convergence: true
projection_bandwidth: 0.10
distance_scale_gradient: full
```

The fixed cross-domain cost comes from pooled DINOv2 patch self-similarity
descriptors. InfoOT kernels operate on matching-head features. Conditional
projection returns weights over raw target codes, which preserves the decoder's
coordinate system.

Fixed validation now records each probe's own solver status and the final L1
plan change. Reconstruction also crosses the current/fixed Stage 1A encoders
and generators using identical images/noise/times to locate conditioning drift.
These crossed-pair probes are diagnostics, not targets for decoded-image quality.

### 5.3 Losses

`cat_reconstruction_loss` and `dog_reconstruction_loss` are SNR-weighted
latent velocity-gap losses, using real-image VAE latents and sampled flow
times/noise. They are not decoded pixel reconstruction losses. Original RGB
images are the target for standalone native reconstruction image metrics;
Stage 1A comparisons diagnose changes but do not define image quality.

The retained weighted objectives contain:

- Cat and Dog native flow reconstruction, weight 1.0 each;
- neural InfoOT MI, effective coefficient -0.02 after alignment ramp;
- conditional structure KL, weight 0.02 after alignment ramp;
- matching variance/covariance protection, weights 0.05/0.01 without ramp;
- decoded original-source perceptual InfoNCE, weight 0.10 after image ramp;
- target RGB PatchGAN adversarial loss, weight 0.01 after image ramp, with DiffAugment;
- HistoGAN original-source RGB-uv palette, weight 0.02 after image ramp.

PCGrad treats native flow, transport (MI + conditional KL), protection,
perceptual, realism and source palette as six weighted tasks. The removed loss implementations
remain for historical controls. Scalar loss reductions and the update rule
are specified below; fewer terms are a pilot, not a proven quality optimum.

Matching protection acts on the 96 post-L2 OT references separately per domain.
Internally scale features by `sqrt(feature_dim)`, use population covariance,
penalize coordinate standard deviations below 0.70 (epsilon 0.0001), and
average squared off-diagonal covariance entries. Average domain losses equally.
This term has full configured weight from update 1 and is not ramped. It
backpropagates through matching heads and encoders; decoder codes/conditional
means are not its targets. Validation reports the same diagnostics on the
fixed training reference bank without updates. The batch covariance rank is
at most 95 for 512 features; zero off-diagonal covariance is not a feasible
goal when all coordinate variances are positive, hence the modest soft penalty.

Disabled legacy neighborhood teaching uses all 128 encoded training samples independently in
each domain. It compares teacher and student row-softmax distributions of
negative squared pair distances, each divided by its own off-diagonal mean
squared distance and temperature 0.20. The student scale is differentiable;
self-pairs are excluded. This is a relational-distillation-inspired extension,
not the original RKD Huber objective. It is invariant to uniform distance
contraction above the 1e-8 numerical floor and therefore still needs explicit
spread protection. Legacy configurations default to cosine neighborhoods.

Diagnostics now separate weighted V/C encoder and head gradients and the
head gradients from alignment, neighborhood and conditional losses. Raw
normalized and matching banks each report entropy rank, participation rank,
and top eigenvalue share. These identify spread versus spectral concentration;
gradient norms alone do not identify which objective caused contraction.
`relative_covariance_energy` divides off-diagonal energy by squared mean
coordinate variance (floored at epsilon) to expose redundancy when the entire
cloud contracts. It is diagnostic only and is not Pearson correlation.

Native conditioned preservation is optional and disabled in the active config
(weight zero, no forwards or random draws). The imperfect Stage 1A native
prediction is not a teaching target. Same-domain reconstruction trains toward
real-data velocity. The retained null-preservation term is a CFG regularizer,
not the metric used to assess image quality.

The decoded contribution ramps from zero to full strength during the first
2,000 updates. All 32 queries in each direction are projected with the full
minibatch conditional mean, decoded by a differentiable 20-step midpoint-time
Euler sampler, and passed through frozen DINOv2.

The adversarial path applies translation-only DiffAugment to both generated
and VAE-reconstructed real target RGB before RGB PatchGAN. Per-image integer
offsets are independent for real/fake, bounded by 6.25% of each image dimension,
with zero padding. The fake RGB tensors are shared by the D step (detached)
and G step (live). Perceptual InfoNCE, HistoGAN and cosine diagnostics use
unaugmented inputs. Private per-target augmentation RNG streams run on DINO's
device and are checkpointed as `decoded_augmentation_states`; draws happen
outside checkpoint closures. Validation/evaluation are unaugmented and do not
advance these streams. RGB PatchGAN bypasses DINO on adversarial inputs;
there is no extra diffusion rollout or VAE decode. See
[DiffAugment implementation](../docs/analysis/stage1b_diffaugment/review.md).
Changing the augmentation config requires a fresh experiment; older configs
default to disabled and do not require the new RNG state.

Source-image supervision uses perceptual InfoNCE and HistoGAN palette against original RGB.
It uses half CLS similarity plus half mean corresponding-patch similarity.
Structure cosine is a diagnostic against the VAE-reconstructed source, and
structure InfoNCE is off. Target realism uses spectral-normalized RGB PatchGANs
with real RGB from VAE-reconstructed target references. Standalone native RGB metrics use original RGB. These paths
should not all be described as supervision against original target images.
Projection support and both earlier gradient controllers are disabled.

### HistoGAN palette and RGB PatchGAN revision

`decoded_translation.discriminator_kind: rgb_patchgan` selects a five-convolution,
70x70 RGB critic with base width 32, spectral norm and no batch/instance norm.
The legacy DINO MLP is retained for old checkpoints and the matched control.
The new discriminator kind is checkpointed and incompatible resume is rejected.

`decoded_translation.color_histogram` uses weight 0.02, bins 64, input_size 128,
sigma 0.02. HistoGAN RGB-uv soft histograms use three log-ratio planes, intensity
weighting, inverse-quadratic kernels and joint mass normalization. Mean per-image
Hellinger distance provides a separate E/H/G PCGrad task under the image ramp.
Targets are detached original source RGB; no additional generation is needed.
The whole-image statistic includes background, lacks spatial correspondence and
mostly ignores exposure. Reliable foreground masks are not available in this pipeline.

Fixed in-training validation saves all 32 source/generated pairs in each direction,
using unchanged query/reference IDs and private diffusion seeds. Color mean and
per-image values are logged even for weight-zero controls. Standalone evaluation
adds the same metric and original-source pair grids on its existing 16-image panel.
The panels differ between raw training validation and standalone EMA evaluation.

See [experiment/commands](../docs/analysis/stage1b_patchgan_histogan/review.md).

### 5.4 Active losses and PCGrad routing

Let a = min(step/1000, 1) and r = min(step/2000, 1). Native losses sum Cat/Dog
terms; conditional KL, V/C and image losses average domains/directions.

| Task | Effective coefficient | Data / trainable path |
| --- | --- | --- |
| Native flow velocity MSE | 1 per domain | First 64 references; E/G |
| Neural MI + conditional KL | -0.02 a / 0.02 a | 96 references + 32 disjoint queries; E/H |
| Variance / covariance | 0.05 / 0.01 | 96 matching references; E/H |
| Original-source perceptual InfoNCE | 0.10 r | 32 translations and source-query keys per direction; E/H/G |
| Target RGB PatchGAN adversarial | 0.01 r | 32 translations per direction; E/H/G |
| HistoGAN RGB-uv palette | 0.02 r | Original source RGB histogram target; E/H/G |

The perceptual keys are original source RGB in query order, detached, with
up to 31 other source images as negatives. Temperature is 0.20 and near-copy
masking threshold is 0.95. Rows without valid negatives do not train. It is
relative source retrieval, not a direct RGB color penalty. Both cosine
metrics remain diagnostic only. See the current loss guide for equations.

For each combined Cat+Dog encoder, head and generator group, PCGrad collects
weighted task gradients, then projects negative dot products using the
original reference gradient in randomized task order. Projected gradients are
summed, preserving weighted-sum scale when there is no conflict. This is
symmetric, not a reconstruction-priority cap. Multiple sequential projections
can leave residual conflicts; Adam/clipping do not guarantee simultaneous
descent. PCGrad is not a proven cure for the prior image-quality plateau.

The random task ordering uses a private CPU generator derived from train seed
and step, leaving model/DataLoader/augmentation RNG untouched. Resume rejects
changed PCGrad settings or seed. The existing loader does not save a batch
permutation cursor, so identical uninterrupted/resumed batches are not promised.
The bounded decoded/reconstruction controller is off; no ratio of 2 is imposed.

Initial Stage 1A latent anchoring and null/conditioned prediction preservation
are all zero. Native flow remains real-data supervision and semantic dropout
still trains the null path. Matching neighborhood KL and both matching
contrasts are off. Generated-code InfoNCE and decoded structure InfoNCE are
also off: their incremental quality benefits are unverified in the supplied
Stage 1B logs. They remain available for controlled ablations.

The trainer skips zero-weight auxiliary computations. Every logged PCGrad
update reports active tasks, projection/conflict counts, correction and sum
norms, and task-local descent fractions. Sampled gradient_conflicts every
200 steps describe pre-PCGrad gradients; post-projection sum metrics are
separate. These measurements are not Adam parameter-update predictions.

One norm-1 clip applies to each combined E, H and G group after surgery.
D retains its separate hinge update and clip, using detached real/fake RGB.
Its weights and spectral-norm buffers are frozen during G's adversarial
forward while image derivatives remain live. Validation never updates D.

### 5.5 Optimizers and schedule

| Group | Learning rate |
| --- | ---: |
| Encoders | `2e-5` |
| Matching heads | `2e-4` |
| AdaLN/token MLP/`z_proj`/final adapter | `5e-6` |
| Attention LoRA | `3.75e-6` (trial; previous control `2.5e-6`) |
| RGB PatchGAN discriminators | `1e-4` |

Native reconstruction uses 64 samples/domain and decoded translation uses 32
queries/direction. Both average their losses, so their counts need not match.
Keep these same batch sizes in the 2.5e-6 LR control. A decoded batch of 64
would require 64 conditional queries and transport batch 160 to retain the
96 fit references; update validation query count to 64 as well. This also
changes the perceptual negative bank and decoded compute budget. The
active trial changes LoRA LR by 1.5x, not every Adam group according to a
single batch multiplier. Research and rationale:
`docs/analysis/stage1b_lora_lr_plateau.md`. Strict resume checks reject changed
generator rates; use a fresh run from the same Stage 1A initialization.

The configured pilot budget is 4,000 updates. Alignment warms up for 1,000 updates;
decoded losses ramp for 2,000. Validation runs every 500 updates and numbered
checkpoints are kept every 500. Logs use 100-update windows every 100 steps;
gradient diagnostics run every 200 steps. Encoder, head, and generator EMA use decay
0.995 after a 500-update warmup. The active Stage 1B config loads both domain
branches on `cuda:0`, which removes repeated Dog-to-Cat device transfers. This
requires enough memory for both SiT/PDAE branches, both VAEs, DINO features,
optimizer state, and the checkpointed decoded-translation graph.

Native flow validation uses 32 fixed real examples/domain. This expands the
earlier eight-example probe and changes its sampled noise/time protocol;
historical eight-example means are not matched comparisons. Standalone image
evaluation uses 32 native images/domain and 16 translations/direction at the
same fit/projection bandwidths. Its protocol identifier separates reports
from the earlier eight-image evaluation. Re-evaluate control checkpoints on
the same larger panel for image-quality comparisons.

Run a meaningful first review at 4,000 updates:

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --max-steps 4000
```

Resume the same experiment with:

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --resume
```

`--smoke` is a 500-update mechanism check. Do not use it as the final run. A
new objective, bandwidth, trainable parameter set, data fingerprint, or Stage
1A initialization requires a new experiment directory rather than resume.

## 6. Checkpoint contract

Experiment D uses checkpoint format 4. Each checkpoint contains:

- current and EMA encoders;
- current and EMA matching heads;
- selected trainable generator state and paired generator EMA;
- optimizer and EMA counters;
- fixed Stage 1A generator teacher snapshots needed for null loss and probes;
- feature discriminators and their optimizer;
- training and translation-noise RNG states;
- configuration, Stage 1A path/step/architecture provenance, and frozen-prior
  content fingerprint.

No batch-local transport plan is saved. The offline Stage 3 plan is built from
the final selected bank. Evaluation must restore raw E/head/G together or EMA
E/head/G together.

Anchor encoders are reloaded from the referenced Stage 1A checkpoints, not
stored as separate anchor-encoder tensors in this payload. Keep those source
artifacts immutable: the Stage 1B provenance check compares path, step, selected
weights and architecture, not a hash of every Stage 1A tensor. This is weaker
than the downstream offline bundle's content/provenance checks.

Stage 1B resume restores optimizer parameter groups from the checkpoint; it
does not reapply a changed `train.lr_encoder` after loading. Do not assume an
edited YAML LR retunes a resumed run. G/head LR changes are covered by stricter
config-block comparisons. Inspect logged actual learning rates. Stage 1A's
trainer separately reapplies its configured optimizer hyperparameters on
resume. The solver-change exception in Stage 1B permits a larger outer budget
when outer convergence is required, and a larger Sinkhorn projection budget
when both marginal and outer convergence are required. Objective, tolerances,
patience and initialization must still match; budgets cannot decrease.
Strict runs may also enable/increase `recovery_iterations` on resume: recovery
only runs after the ordinary cap would otherwise fail. It continues the same
plan and already constructed kernels/cost in FP64, with inner marginal tolerance
`min(projection_tolerance, outer_tolerance / (10 * max(n, m)))`. This controls
accumulated marginal error heuristically; it does not bound the subproblem plan
error. Outer L1 change must satisfy the original tolerance in both internal and
returned dtypes for the configured patience. The output plan keeps its original
dtype and must pass the original marginal checks.
As of 2026-09-25, recovery retains log-domain Sinkhorn column scaling between
updates within the same solve. This prevents repeated cold, capped inner solves
from producing zero outer change while missing the tighter recovery tolerance.
The current cost is solved each time; no plan multiplication or cost accumulation
is introduced. Normal-path initialization, objectives, budgets and tolerances
remain unchanged. Errors distinguish inner accuracy from outer stability.
The plan-gradient density-floor derivative is also corrected to match the MI
loss's explicit floor, including its equality convention. See [the recovery fix and official-source audit](../docs/analysis/infoot_inner_recovery/review.md).
The subsequent [full official-repository audit](../docs/analysis/infoot_full_repository_audit/review.md)
executes upstream commit `352efd202f5b475dc170a8d08a99049689d5ee1a` in test-only
parity checks for fitting and both projection directions. It confirms the
current formulas, documents inherited query-batch RMS dependence, and clarifies
that independent initialization is stationary in symmetric cases but not all
finite KDE problems. The own-feature cost remains detached in neural training;
the alignment loss remains MI-only. No new training protocol is enabled by this audit.

**Reference-EMA RMS (main calibration, 2026-09-25).** The self-supervised
`self_supervised_rms_ema_sit_b2.yaml` and
`self_supervised_patchnce_rms_ema_sit_b2.yaml` recipes in `configs/stage1b_infoot/`
have matched names in `configs/stage1b_eval/`. Their short training outputs are
`outputs/stage1b_self_rms` and `outputs/stage1b_self_patch_rms`. The original
recipes remain controls. The only optimization change is conditional-projection
calibration: per-domain EMA of training-reference covariance trace, decay 0.99,
then square root, replacing query-batch RMS. Initialize on 256 unaugmented train
references and update once per step from fit references only. Raw and model-EMA
encoder/head weight modes have separate checkpointed histories; the model-EMA
history adds a chunked no-gradient reference encoder/head pass. Validation and
standalone evaluation and Stage 2–3 / 4 freeze the selected history. Fit/MI still use current
reference RMS, full denominator derivatives, and bandwidth 0.55; projection
remains 0.10, with a detached running denominator and live distance gradients.
Losses, LRs, variance/covariance protection, PCGrad, and CUT code stay fixed.
Logs add `projection_rms` scale/lag/effective-width statistics; fixed validation
adds `query_batch_dependence_first_query_l1` per direction for both new recipes
and old controls. New runs must start fresh; old Stage 1B checkpoints lack RMS
history. See [design and comparison protocol](../docs/analysis/stage1b_projection_rms_ema/review.md).

Active training, its DINO/PatchGAN controls and matched quick evaluation now use
`projection_iterations: 10000` (previously 2000), with unchanged marginal
tolerance 1e-5, normal outer cap 1200, and up to 4800 additional recovery updates
(6000 total). Easy solves still stop early without recovery. Training windows
log `infoot_recovery_used` as the fraction of updates using recovery; train and
validation also record recovery counts/budgets and effective inner tolerance.
See [the outer recovery fix](../docs/analysis/stage1b_outer_recovery/review.md) and
[the Sinkhorn fix](../docs/analysis/stage1b_sinkhorn_budget/review.md).
Ordinary `train`/`data`
fields are not all compared by that guard; use a fresh controlled experiment
for objective, minibatch or initialization changes even if a field is accepted.

### 6.1 Loss logging and validation weight modes

`loss` sums the six weighted task groups; PCGrad changes gradients without
changing that scalar. `decoded_translation.weighted_loss` is the weighted
perceptual/adversarial/color sum before the ramp; `effective_weighted_loss` includes
the ramp. Code recovery and matching contrasts are disabled in the active
recipe. Their optional legacy metrics preserve the earlier reductions when
enabled. Compare images and individual metrics, not totals across recipes.

Training console/JSONL and fixed validation now use `log_schema_version: 2`
and include the configuration-derived `enabled_losses` list. Disabled objective
fields, including their window means, weights and gradients, are omitted.
Enabled losses that reach zero or are still warming up remain visible. Inactive
structure cosine and the perceptual cosine diagnostic are named
`decoded_translation.diagnostics.structure_cosine_distance` and
`decoded_translation.diagnostics.perceptual_cosine_distance`, including within
the per-direction blocks. Weight-zero histogram controls place their distances
under the same `diagnostics` namespace. Active loss names stay unchanged.

Gradient reports omit disabled pairs and the redundant `dino_vs_adversarial`
alias when only perceptual DINO supervision is enabled. Actual PCGrad metrics,
valid zero cosines and invalid measurements for enabled pairs are retained.
Validation omits Stage 1A reconstruction comparisons when all Stage 1A teacher
objectives are off; current real-data flow/null probes remain. Calculations,
RNG and training updates are unchanged. Old files are not rewritten, and the
summary/color-comparison scripts read both schemas, including resumed mixed
files. See [the logging guide](../docs/analysis/stage1b_log_cleanup/review.md).

Conditional KL now logs its actual weighted contribution, plus per-direction
uniform/query-independent baselines and query-information diagnostics in train
and validation. Gradient probes isolate KL against translation, reconstruction,
InfoOT MI and feature protection on E/H, before PCGrad. These identify teacher
agreement and local tradeoffs; they do not establish causal visual benefit.
The summary script also writes `conditional_kl.md`. See
[the KL diagnostic guide](../docs/analysis/conditional_kl_diagnostics/review.md)
for exact keys, interpretation and the historical 5,000-step report.

Fixed in-training validation uses **raw** paired weights, fixed references,
queries and noise, and does not update D. Its decoded losses report full weight
(`ramp=1`) for comparability, even during the training warmup. Current counts are
32 native flow examples/domain, 96 fixed training references, 32 held-out
conditional queries and 32 decoded queries/direction. Native flow sample IDs
are not logged by that probe; projection/decoded IDs and noise seeds are logged.
Standalone quick evaluation uses paired **EMA** weights by default, 512 fitting
references, all training projection keys, 256 validation queries, 32 native RGB
reconstructions and 16 translations/direction. These are different protocols,
not interchangeable estimates. Re-evaluate old checkpoints on the same panel
and weight mode before making matched claims.

## 7. Quick evaluation

Run:

```bash
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_decoder_sit_b2.yaml \
  --checkpoint outputs/stage1b_patch_color/checkpoints/latest.pt \
  --no-require-stage1a-baseline
```

The quick protocol uses 512 training references per domain, every available
training code as the projection bank, and 256 validation queries per domain.
It fits at bandwidth 0.55 and reads conditional projections at 0.10. The
evaluator warns if its fit bandwidth differs from the alignment config. A CLI
`--projection-bandwidth` override is valid for sensitivity studies only.

For primary checkpoint comparisons, match the checkpoint's actual training
settings: active D uses fit bandwidth 0.55, entropy epsilon 0.02, MI weight
0.10, cross-cost weight 1.0, and projection bandwidth 0.10. Check the saved
run configuration when evaluating an older checkpoint. The bandwidth warning
compares YAML files; it does not automatically reconcile every saved solver
parameter. Keep parameter sweeps separate from the matched comparison.

Matching parameters does not make protocols identical: the main self-supervised
recipe fits 224 references per domain (256 minus 32 disjoint queries), quick
evaluation fits 512, and Stage 2–3 fits all training references. Reference-EMA
projection reuses the same checkpoint history across these protocols; fitting
RMS still reflects each actual reference bank. Matched settings are the starting point for global
fitting, not evidence that they are optimal at every bank size.

The current self-supervised quick-evaluation grids contain three rows of 16
images per direction: VAE-decoded source, original RGB top-1 target reference,
and full InfoOT conditional-mean generation (`readouts: [conditional_mean]`,
`include_source: true`). `translation.include_top1_target` defaults to `true`.
The reference is selected by the maximum conditional probability over the
projection target bank, not by nearest distance to the mean code. Its sample ID,
probability and bank index are recorded under
`decoded_image_diagnostics.top1_target_reference`; it is not paired ground truth.
This adds only original-image loading, with no extra generation or RNG draw.
Set the flag to `false` for the previous layout. Other diagnostic readouts remain
explicitly selectable, and only selected readouts are decoded. The shared
structure-decoder recipe still hides its source row explicitly.

The v6 quick evaluator now selects `readouts: [conditional_mean, z_cfg_2]` and
shows a fourth row after the standard conditional-mean translation. This row
uses the identical InfoOT mean code, initial noise and step count, with semantic
guidance `v_null + 2 * (v_z - v_null)` (`guidance_scale=2`). The code vector is
not multiplied by 2. Image diagnostics run for both generated rows;
`generation_protocol.translation_readout_guidance_scales` and optional saved
generation inputs record each row's guidance. The primary readout remains the
standard conditional mean at guidance 1. Other recipes retain their selected
readouts.

Report checkpoint selection using both translation directions. Inspect:

- same-domain reconstruction error against original RGB images (MSE, PSNR, LPIPS)
  and absolute ground-truth latent flow loss;
- InfoOT convergence, marginal residuals, MI, and effective target count;
- conditional KL and expected frozen-teacher structure cost;
- conditional-mean norm, centered variance, and nearest-target distance;
- decoded DINO structure for every grid readout;
- viewpoint, framing, and coat-color proxy precision and coverage;
- actual target realism and source correspondence in saved grids;
- current and fixed-null reconstruction diagnostics.

Standalone reconstruction grids contain ground truth, VAE reconstruction, and
model reconstruction. Original images use the latent cache's crop/resize.
VAE-to-ground-truth error is reported separately, and missing image metadata
does not silently fall back to a decoded latent. A new evaluation protocol ID
distinguishes these metrics from historical VAE-reference reports.

Stable entropy is not proof of a broken solver, and a visually mixed UMAP is
not proof of useful translation. The full decoded result and independent labels
decide whether to continue.

## 8. Stage 2–3 offline alignment and Stage 4 evaluation

### 8.1 Frozen banks and one global coupling

Select one format-4 checkpoint from the main self-supervised learned-PatchNCE /
reference-RMS-EMA run. Restore its paired encoder, matching-head, adapted generator,
PatchNCE sampler, and RMS history (model-EMA weights by default). Stage 2 encodes
**every training sample** into banks of raw decoder codes and matching features,
with sample IDs, preprocessing metadata, and provenance hashes. Validation
images never enter these fitting banks. Actual domain counts may differ;
there is no hard-coded 5,000-image cap.

Stage 3 fits **one global balanced fused InfoOT coupling** over those complete
banks. Cat-to-Dog uses the coupling and Dog-to-Cat uses its transpose. Uniform
marginals are 1/N and 1/M. Encoding batches and matrix tiles are execution
controls; they do not create independent mini-batch OT problems. The method
uses neither partial transport nor a sparse/top-k approximation.

Active settings are fit bandwidth 0.55, projection bandwidth 0.10, MI weight
0.10, entropy epsilon 0.02, and cross-cost weight 1.0. The outer cap is 1,200
with up to 2,000 Sinkhorn iterations per update. Outer stopping uses L1 change
1e-5, at least five updates, and patience three. Full-bank marginal checks use
absolute tolerance 1e-7 and relative tolerance 1e-3. These tighter marginal
checks account for the much smaller masses in a full bank.

The deployment protocol is `full_bank_calibrated_projection_v2`. Its **fit RMS**
comes from the complete training banks. Its **projection RMS** comes from the
checkpoint's saved reference-EMA history and is never recalculated during export
or held-out evaluation. `--weights raw` selects raw encoder/head history;
`--weights ema` selects the separate history measured with model-EMA weights.
Both source and target density kernels use these frozen projection scales.
The global cross-cost is the same own-feature cosine cost used in Stage 1B;
the main pipeline does not load DINO. Stage 4 loads the fitted bundle and cannot
independently refit, override bandwidth, or update RMS from queries.
Full conditional weights are normalized over all target references and average
raw target codes; the resulting mean is not renormalized.

Both main YAMLs set `require_projection_rms_mode: reference_ema`. Missing history,
mismatched configuration, weight family, or update count is an error. Earlier
non-EMA/DINO controls can be rebuilt with their own `alignment_config`,
`require_projection_rms_mode: full_bank` in both offline YAMLs and separate output
paths. They retain full-training-bank projection calibration. Old v1 bundles
must be rebuilt in a new directory; they cannot be relabeled as EMA results.

### 8.2 Artifacts, cost, and resume

The Stage 2–3 output contains:

- `run.json`, `checks.json`, and a completed `bundle.json`;
- `banks/cat.pt` and `banks/dog.pt`: raw codes, matching features, IDs and metadata;
- `models/paired_state.pt`: paired model/sampler state, raw/model-EMA RMS histories,
  configuration and provenance;
- `transport.pt`: global coupling and marginals;
- `calibration.json`: separate `fit_scales` / `projection_scales`, RMS mode,
  selected weight family, checkpoint step, update count, decay, epsilon and history hash;
- `solver_progress.pt` and `solver_log.jsonl`: resumable numerical progress.

Kernels and density terms are rebuilt from the banks and calibration. Frozen
base-model dependencies remain external and hash-checked; the bundle is not a
self-contained copy of every pretrained weight. Resuming checks model, data,
configuration, and implementation identity before reusing banks or solver state.
Use a new output directory for changed numerical settings or model identities.
An exhausted outer budget returns `iteration_limit` (CLI exit 2), rather than
exporting a completed translation bundle.

The solver uses dense global matrices even with tiled Sinkhorn. One 5,000 by
5,000 float32 matrix occupies about 95.4 MiB; several matrices and model states
are needed, and dense MI products have cubic scaling for equally sized banks.
Measure actual GPU cost with `--max-new-iterations 5` before a long run. This
preflight still uses full banks and preserves progress; it does not produce a
usable bundle unless fitting has converged. Continue with `--resume` and omit
that temporary iteration limit.

### 8.3 Final Stage 4 protocol

Translate **all held-out validation sources** in each direction using the frozen
bundle, full conditional mean, 20 sampling steps, and guidance scale 1.0.
Use all real held-out target-domain images for FID. Report separately:

- **Cat-to-Dog FID** and **Dog-to-Cat FID**, using Clean-FID clean Inception
  2048-dimensional features and the actual held-out target corpus;
- **Cat-to-Dog source SSIM** and **Dog-to-Cat source SSIM**, comparing each
  translation against its original source RGB image.

SSIM uses matching 256-pixel crop/resize, float [0,1], data range 1, Gaussian
sigma 1.5, population covariance, and RGB-channel averaging. Report mean,
standard deviation, sample count, and per-image values. Native reconstruction
SSIM is outside this final protocol. Stage 1A outputs and VAE reconstructions
are not image metric targets.

Save **16 source/translation pairs per direction** (32 total), including
individual source and translation PNGs, pair panels, and labeled contact sheets.
Selection is deterministic; if fewer than 16 sources exist, save all and report
that count. The inspection subset does not limit metric sample counts. Keep
panels and contact sheets outside FID input folders.

The output includes `metrics.json`, `ssim_per_image.jsonl`,
`generation_manifest.jsonl`, `gallery_ids.json`, `protocol.json`, `report.md`,
full translation folders under `images/`, original RGB folders under `real/`,
and inspection folders under `galleries/`. Query banks, projections, and receipts
support validated resume. Stable per-source seeds preserve outputs across chunk
sizes; resume verifies cached images and regenerates damaged ones.
`protocol.json` and `metrics.json` also record fit/projection bandwidths, scales,
RMS provenance and cross-cost source; `report.md` summarizes them.

Source SSIM measures structural similarity, not target realism: copying the
source can score highly. Interpret it jointly with FID and images. FID has
finite-sample effects, so compare identical splits/counts and sampling settings.
The validation split used for tuning is not an untouched final test set.

### 8.4 Linux commands

```bash
cd /data/not_backed_up/yxu209/diffusion-ot
python -m pip install -r requirements-stage4.txt

# Replace with the selected, existing numbered checkpoint.
CHECKPOINT=outputs/stage1b_patch_mlp_rms/checkpoints/step_004000.pt

python scripts/run_offline_alignment.py \
  --config configs/stage23_offline/full_sit_b2.yaml \
  --checkpoint "$CHECKPOINT" --weights ema --device cuda:0

python scripts/evaluate_full_infoot.py \
  --config configs/stage4_eval/fid_ssim_sit_b2.yaml \
  --bundle outputs/s23_rms --device cuda:0
```

For a full-size timing preflight, append `--max-new-iterations 5` to the
Stage 2–3 command. To continue it, rerun with `--resume` and without
`--max-new-iterations`. Stage 4 also accepts `--resume`; reduce
`--generation-batch-size` if needed without changing the evaluation corpus.
Python 3.10 is supported by the Stage 4 pins: Clean-FID 0.1.35,
scikit-image 0.25.2, and SciPy 1.15.3. The first FID run
may download Inception weights; the configured cache is `outputs/fid_cache`.

## 9. Code ownership

| Path | Responsibility |
| --- | --- |
| `src/diffusion_ot/models/pdae_sit.py` | SiT/PDAE branch, checkpoint compatibility, CFG/null behavior |
| `src/diffusion_ot/models/matching_head.py` | Residual normalized matching heads |
| `src/diffusion_ot/models/generator_adaptation.py` | Select, freeze, save, and restore Experiment D generator parameters |
| `src/diffusion_ot/losses/infoot.py` | Kernels, MI derivative, projected Sinkhorn, conditional weights and means |
| `src/diffusion_ot/losses/semantic_prior.py` | Frozen structure descriptors and cache validation |
| `src/diffusion_ot/losses/conditional_structure.py` | Disjoint-query conditional teacher loss |
| `src/diffusion_ot/training/decoded_translation.py` | Differentiable sampler, DINO image features, discriminators, G EMA, image-loss RNG |
| `src/diffusion_ot/training/train_pdae_domain.py` | Stage 1A flow training, warm start, resume, EMA and validation |
| `src/diffusion_ot/training/stage1a_refinement.py` | Native original-RGB losses, G-only native-code InfoNCE and fixed decoded validation |
| `src/diffusion_ot/data/ground_truth.py` | Identity-checked original RGB with cache preprocessing |
| `src/diffusion_ot/losses/contrastive.py` | Detached-key InfoNCE and teacher-positive-set matching losses |
| `src/diffusion_ot/training/pcgrad.py` | Active symmetric task-gradient projection, independently on E/H/G |
| `src/diffusion_ot/training/gradient_balance.py` | Legacy bounded decoded encoder controller; disabled with PCGrad |
| `src/diffusion_ot/training/train_joint_infoot.py` | Self-supervised and Experiment D orchestration, validation, checkpoints, resume |
| `src/diffusion_ot/losses/matching_regularization.py` | Variance floor and modest covariance penalty on normalized OT references |
| `src/diffusion_ot/evaluation/stage1a_eval.py` | Stage 1A reconstruction/CFG evaluation |
| `src/diffusion_ot/evaluation/stage1b_eval.py` | Banks, InfoOT evaluation, projection diagnostics, grids, UMAP, proxies |
| `src/diffusion_ot/evaluation/full_infoot.py` | Global tiled Sinkhorn, exact InfoOT outer updates, frozen-scale conditional projection |
| `src/diffusion_ot/evaluation/offline_pipeline.py` | Stage 2–3 frozen bank construction, paired state, global fitting and validated bundle export |
| `src/diffusion_ot/evaluation/final_eval.py` | Stage 4 full validation corpus, resumable generation, galleries and report |
| `src/diffusion_ot/evaluation/final_image_metrics.py` | Clean-FID, source SSIM, original/generated PNGs and pair panels |
| `scripts/*.py` | Thin command-line entry points |

## 10. Legacy and verification

`legacy/infoot_official_v1` preserves the previous official-based project
implementation byte for byte with a hash manifest. It is a reproducibility
control, not the active source tree. Use `scripts/run_legacy_infoot.py` so its
imports remain isolated.

Historical experiment interpretation lives in:

- `docs/stage1b_guarded_8000_review.md`;
- `docs/stage1b_projection_1000_review.md`;
- `docs/stage1b_fused_projection_review.md`.

Experiment D's design and test record lives in `docs/stage1b_experiment_d.md`.
The main recipe's references are `docs/analysis/stage1b_patchnce_mlp/review.md`,
`docs/analysis/stage1b_projection_rms_ema/review.md`, and section 8 above.

The 2026-09-25 EMA offline integration passed **16 offline/projection tests**
plus **33 shared RMS/sampler/evaluation tests**, with one CUDA-dependent test
skipped. Coverage includes raw/model-EMA history and MLP export/loading,
encoder-only cost (no DINO load), full-bank fit versus saved projection scales,
training/offline readout parity, query/chunk invariance, legacy full-bank mode,
strict history checks, synthetic Stage 2–4 generation and resume. CLI help and
`git diff --check` also passed. These are CPU implementation checks; no full
AFHQ/GPU quality evaluation was run for this change.

Focused verification (commands, not a claim that the entire suite passes):

```bash
pytest -q tests/test_offline_stages.py tests/test_offline_projection_rms.py
pytest -q tests/test_infoot.py tests/test_stage1b_training.py tests/test_stage1b_eval.py
```

Tests cover transport math, target smoothing and density correction,
conditional normalization, disjoint splits, selective G updates, both-direction
image gradients, frozen base/null weights, sampler gradient equivalence,
discriminator isolation, deterministic validation, resume, raw/EMA pairing,
and decoded-grid metrics. GPU AFHQ training remains the empirical validation;
unit tests establish implementation behavior, not translation quality.

The earlier implementation verification record includes 11 passing offline-pipeline
checks and a real Inception CPU feature-extraction smoke check. No complete
AFHQ GPU Stage 2–3/4 run has been validated locally. A broader earlier check
also found the pre-existing legacy quick-evaluation bandwidth mismatch test
(`test_quick_evaluation_uses_the_training_infoot_kernel_and_entropy`); it must
not be described as a passing full suite. Active Experiment D and the new
full-bank configurations use matched fit 0.55 / projection 0.10 settings.


The earlier contrastive test records are historical snapshots. The Stage 1A
refinement implementation subsequently passed 58 targeted tests, including
G-only routing with clipping and validation RNG isolation. Its broad suite
record was 375 passed / 1 skipped / 1 pre-existing legacy plain/quick fit
bandwidth failure (0.55 versus 0.70), before its final RNG regression test was
added. The latent-flip fixture was repaired during that implementation and
is no longer an outstanding failure. See `docs/analysis/stage1a_refinement/review.md`.

This notes audit checks the current code/config values; it does not claim a
new full-suite result or a GPU quality run. Existing shuffled-loader resume
saves generator state but not the live permutation/cursor. No exact batch replay
or AFHQ quality claim follows from CPU checks. For the audit's corrections and
remaining review findings see `docs/analysis/stage1a_stage1b_notes_audit.md`.
