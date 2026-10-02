# diffusion-ot

Cat/Dog PDAE representations with InfoOT conditional projection and diffusion
translation. **Fresh runs default to the residual latent CNN with cosmap flow
weighting in both Stage 1A and Stage 1B (v7).** The encoder has a raw stride-1
convolution stem; internal GroupNorm and encoder/generator LayerNorm remain.
Earlier plain-CNN, RGB, InfoNCE/PatchNCE and external-supervision recipes remain
available through explicit configuration paths.

## Active workflow

The current pipeline has four stages:

1. Train Cat and Dog afresh with the residual-cosmap Stage 1A recipes; evaluate
   their native reconstruction, conditioning controls and N1 appearance probes.
2. Pin the selected Stage 1A step checkpoints and initialize fresh v7 Stage 1B
   heads, optimizer/EMA and training-only calibration. Keep differentiable means
   and the unchanged v6 loss coefficients for this architecture baseline.
3. Evaluate against the saved calibrated Stage 1B step 0 with 224 references/
   targets, 20 integration steps, mean/MAP panels and disjoint train/development
   appearance probes. Use this new baseline before loss/controller ablations.
4. Export the selected encoders, matching heads, adapted generators, kernels,
   target codes, and fitted transport for downstream inference.

The canonical configurations are:

- Stage 1A Cat / Dog default: `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_residual_cosmap.yaml`
- Stage 1A default evaluation: `configs/stage1a_eval/residual_sit_b2_256.yaml`
- Stage 1B default: `configs/stage1b_infoot/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml`
- Stage 1B default evaluation: `configs/stage1b_eval/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml`
- Stage 1A historical plain latent recipe: `configs/stage1a_pdae/{cat,dog}_sit_b2_lora.yaml`
- Stage 1A Cat / Dog optional RGB comparison: `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_rgb.yaml`
- Stage 1B self-supervised PatchNCE: `configs/stage1b_infoot/self_supervised_patchnce_sit_b2.yaml`
- Stage 1B global InfoNCE control: `configs/stage1b_infoot/self_supervised_sit_b2.yaml`
- Stage 1B PatchNCE + reference-EMA RMS experiment: `configs/stage1b_infoot/self_supervised_patchnce_rms_ema_sit_b2.yaml`
- Stage 1B historical v6 source-aware color experiment: `configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml`
- Stage 1B v6 matched evaluation: `configs/stage1b_eval/self_supervised_infonce_v6_sit_b2.yaml`
- Stage 1B v6 bandwidth-only control: `configs/stage1b_infoot/self_supervised_infonce_v6_bandwidth_control_sit_b2.yaml`
- Stage 1B v4.5 spatial-cost control: `configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml`
- Stage 1B v4.5 matched evaluation: `configs/stage1b_eval/self_supervised_infonce_v4_5_sit_b2.yaml`
- Stage 1B v4 image-loss control: `configs/stage1b_infoot/self_supervised_infonce_v4_sit_b2.yaml`
- Stage 1B v4 matched image evaluation: `configs/stage1b_eval/self_supervised_infonce_v4_sit_b2.yaml`
- Stage 1B v3 control, global MLP InfoNCE with 10% conditional-weight gradients: `configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml`
- Stage 1B v3 **100% gradient control**: `configs/stage1b_infoot/self_supervised_infonce_v3_control_sit_b2.yaml`
- Stage 1B v3 matched evaluation: `configs/stage1b_eval/self_supervised_infonce_v3_sit_b2.yaml`
- Stage 1B learned PatchNCE baseline + reference-EMA RMS: `configs/stage1b_infoot/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml`
- Stage 1B full neural InfoOT + relative transport experiment: `configs/stage1b_infoot/self_supervised_patchnce_mlp_full_sit_b2.yaml`
- Stage 1B global MLP InfoNCE + full InfoOT, revised protection/rates: `configs/stage1b_infoot/self_supervised_infonce_full_sit_b2.yaml`
- Stage 1B global InfoNCE + reference-EMA RMS experiment: `configs/stage1b_infoot/self_supervised_rms_ema_sit_b2.yaml`
- Experiment D: `configs/stage1b_infoot/structure_decoder_sit_b2.yaml`
- Experiment D evaluation: `configs/stage1b_eval/structure_decoder_sit_b2.yaml`

The historical self-supervised controls use original flow-only EMA checkpoints with the
unchanged `*_sit_b2_lora.yaml` configs. Experiment D uses its selected
`*_sit_b2_dino.yaml` EMA checkpoints. Neither automatically loads the new RGB
Stage 1A outputs; replacing checkpoint paths alone is insufficient.

### Fresh residual-cosmap workflow (P1a / N1)

The CLI defaults are centralized in `src/diffusion_ot/config_defaults.py`.
Select a domain for Stage 1A; explicit `--config` / `--train-config` still select
historical recipes. Stage 1B train/eval commands default to the paired v7 recipes.
Use process-visible CUDA indices for your machine:

```bash
python scripts/train_pdae_domain.py --domain cat --device cuda:0
python scripts/train_pdae_domain.py --domain dog --device cuda:0
python scripts/evaluate_pdae_domain.py --domain cat --device cuda:0
python scripts/evaluate_pdae_domain.py --domain dog --device cuda:0
```

These Stage 1A recipes start fresh (50,000 updates, batch 64); step checkpoints
are retained. Evaluation uses the exact domain cosmap recipe, original RGB
metrics, correct/shuffled-code controls and the selected raw/EMA state. It runs
the configured **8-image smoke** and separate inferred-noise round trip, plus
N1. The existing `dataset.num_samples: 256` / `metrics.full` fields do not launch
a full generation benchmark; reports contain the actual sample count. Increase
`dataset.smoke_samples` explicitly for a larger native-quality cohort.

After reviewing Stage 1A, replace the v7 recipe's two `latest.pt` initialization
paths with the selected `step_NNNNNN.pt` paths. Both must belong to the named
residual-cosmap runs. Then:

```bash
python scripts/train_joint_infoot.py --quick-eval
# Or evaluate a selected saved Stage 1B checkpoint afterward:
python scripts/evaluate_infoot_alignment.py \
  --checkpoint outputs/stage1b_nce_v7_residual_cosmap/checkpoints/latest.pt \
  --evaluate-step0 --weights ema
```

The new output directory is `outputs/stage1b_nce_v7_residual_cosmap`. Fresh
Stage 1A/1B runs refuse to overwrite an existing latest checkpoint; `--resume`
is for the same architecture and objective. Stage 1B now dispatches its native
loss through the same cosmap/uniform/SNR weighting function as Stage 1A. Both
sample timesteps uniformly; cosmap is not multiplied by SNR. Checkpoints,
`native_flow_objectives.json`, train/validation logs and evaluation protocols
record the resolved objective. Resume/load rejects changes to its weighting,
direction or timestep clamp. Historical SNR checkpoints remain supported;
unrecorded legacy Stage 1B checkpoints cannot be relabeled as cosmap.

The fresh Stage 1B run rebuilds reference banks/plans, projection RMS and frozen
source/spatial calibration from training data. It retains `step_000000.pt` for
matched evaluation. Residual feature stages are explicitly versioned as outputs
after the residual blocks/attention, at 32/16/8/4 pixels. Layers `[0,1]` therefore
use 32/16, unlike the earlier plain encoder. Fit/projection/epsilon remain
**0.55 / 0.25 / 0.02**; separate InfoNCE MLPs, internal norms and loss coefficients
are retained. The v6 inference finalist (0.35 MAP) is a historical comparison,
not a hard-argmax replacement for the differentiable training mean.

Both evaluation recipes enable `input_statistics`. N1 uses up to **1,024 train
and 256 development (`val`) images** without augmentations or extra diffusion
rollouts. It rejects overlapping/duplicate IDs, saves channel means/stds,
original RGB/normalized-Lab summaries, codes before/after encoder LayerNorm,
and actual generator LayerNorm/`z_proj` features. The first-convolution hook
checks that original, offset and scaled latents reach the raw residual stem
unchanged; subsequent feature sensitivity is measured without an invariance
assumption.

Ridge probes compare codes against codes plus the eight latent statistics, with
statistics-only and post-normalization controls. Feature/target scaling and
ridge-strength selection use an internal training holdout only, then refit on
all training samples. Development data is used only for reporting. Probe gains
measure accessibility to a linear probe, not proven loss of image information;
VAE channels are not RGB channels. Check `added_statistics_mse_reduction` for
RGB and Lab separately (positive means adding statistics helped).

Stage 1A's `extra_reports.input_statistics` and Stage 1B's `input_statistics`
link to each `appearance_probe.json`; adjacent `appearance_probe.pt` files save
cohort IDs/features/targets, fitted scalers/coefficients and predictions. Options
include cohort counts, batch size, seed, training tuning fraction and positive
`ridge_alphas`; `enabled: false` disables N1 for a deliberate quick run. The P1
cache reuses N1 within the same checkpoint/options across readout variants.
Native image metrics and explicit unweighted flow diagnostics remain unweighted.

### Historical Stage 1B v6: source-aware selection and color

V6 adds query-to-target spatial/appearance compatibility to the v4.5 readout.
Fixed initial **training-only** cost scales are checkpointed. Added costs are
detached; base conditional probabilities retain their existing gradient route.
The source-color loss is now normalized Lab patch SWD (weight 0.04, scales
64/32/16, patch 5, 128 directions), inspired by the training-free
[MS-SWD implementation](https://github.com/real-hjq/MS-SWD).
Global RGB-uv histogram and target texture **training** weights are zero.
Target patch SWD remains a validation diagnostic, including a real-real baseline.
Projection bandwidth is 0.25 in training/evaluation; fit bandwidth remains 0.55.
Other v4.5 losses, rates, original Stage 1A initialization, and PCGrad are retained.

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v6_sit_b2.yaml \
  --checkpoint outputs/stage1b_nce_v6/checkpoints/latest.pt --weights ema
```

Start this experiment fresh from Stage 1A. Resume its own checkpoints with
`--resume latest`. The paired `v6_bandwidth_control` configs change only projection
bandwidth versus v4.5, apart from output/evaluation paths. See
[v6 implementation, diagnostics and comparison commands](docs/analysis/stage1b_v6/implementation.md).
The changes have CPU integration coverage; image-quality gains need a GPU run.

### Retained Stage 1B v4.5: spatially-correlative transport cost

The separate **v4.5** recipe implements the
[spatial-correlative plan](docs/analysis/stage1b_v3_2500/spatial_correlative_plan.md).
It adds own-encoder spatial relationships to **reference-plan fitting**:
convolution stages 0/1, pooled to 8x8, spatial centering, normalized local
correlations, diagonal removal, and symmetric row-cosine comparison.
Descriptors are detached; conditional projection still uses the existing global
matching kernels. The global InfoNCE MLPs and all v4 losses, weights, learning
rates, PCGrad, projection RMS EMA, and 10% decoded W-gradient routing are retained.

```text
C_mix = g * (0.80 * C_encoder + 0.20 * (s_encoder / s_spatial) * C_spatial)
```

Component spreads use doubly centered cost RMS on a fixed initial training bank.
The gain g matches the initial mixture spread to the encoder-only cost. Scales
and calibration IDs are frozen and checkpointed, separately from RMS EMA.
Constant calibration costs produce an explicit encoder-only fallback. No
external teacher, GAN, extra spatial neural loss, or diffusion rollout is added.

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml --smoke
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml --resume latest
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v4_5_sit_b2.yaml \
  --checkpoint outputs/stage1b_nce_v4_5/checkpoints/latest.pt --weights ema
```

Training outputs are `outputs/stage1b_nce_v4_5`; evaluation outputs are
`outputs/stage1b_eval_nce_v4_5`. Start fresh from the same original Stage 1A
checkpoints as v4. A `step_000000.pt` checkpoint retains the starting weights and
calibration. The evaluator requires a v4.5 checkpoint; evaluate that initial
checkpoint before `latest.pt` for automatic numeric comparison. `--quick-eval`
does both after training. Raw/EMA evaluation uses the corresponding encoder
maps with the same frozen calibration. Incompatible recipes/caches are rejected.

Enabled-only `spatial_correlative` diagnostics report component costs/spreads,
uniform baselines, effective Sinkhorn-cost spread, degenerate rows, fixed-cohort
descriptor drift, and held-out query matching. Evaluation also saves original
source / top two target references / generated image grids with IDs and weights.
These measure the model's own matching prior; judge layout, species, native
reconstruction and artifacts on matched images. CPU integration tests passed;
AFHQ image-quality improvement requires the paired GPU runs. See the
[v4.5 implementation note](docs/analysis/stage1b_v4_5/implementation.md).

### Stage 1B v4 control: four fixed-image losses

The separate v4 recipe adds **coarse RGB 0.02, RGB-uv histogram 0.02,
local layout 0.02, and target patch SWD 0.01** to v3. It retains global MLP
InfoNCE 0.05, native flow, MI/relative alignment, variance/covariance protection,
learning rates, bandwidths, and the original flow-only Stage 1A initialization.
The v3 files and outputs remain available as the control.

The source losses compare generated RGB with **original source RGB**. Texture
SWD compares fixed luminance Laplacian patches with **unpaired original target
training images**, sampled from the disjoint transport-reference pool. No
external feature network, discriminator, own-feature SWD or spectral loss is
added. Each new term can be disabled independently with `weight: 0`.

```text
v4 decoded loss = translation_ramp * (
    0.05 * source_InfoNCE
  + 0.02 * coarse_RGB + 0.02 * RGBuv_histogram
  + 0.02 * local_layout + 0.01 * target_patch_SW1)
```

All terms share the existing decoded images and 2,000-step ramp. The 10% W
backward gate now applies to **all decoded terms**; forward conditions and
direct generator/reference-code derivatives are unscaled. PCGrad treats their
sum as one `decoded_images` task. Component gradient norms and cosines are
measured separately every 100 steps, without changing that task grouping.

Training/validation schema 4 records enabled losses, weights, reductions,
original target IDs and gradient scope. Fixed validation reports per-image
source distances and a disjoint real-versus-real SWD baseline when enough
references are available. Standalone v4 evaluation uses the same definitions
on the displayed images and saves an original-source paired grid. These are
training-related diagnostics, not independent semantic/realism scores.

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v4_sit_b2.yaml
```

Outputs go to `outputs/stage1b_nce_v4`; the quick evaluator is linked in that
training YAML. Start fresh for this loss comparison. Resume within v4 restores
private image-sampling RNG along with diffusion noise; protocol changes are
rejected. The 2,500-step pilot retains v3's duration for a matched comparison.
Weights are initial experimental values; full-model image-quality improvement
must be established with a new GPU run.

### Stage 1B v3 control: protect matching geometry

The v2 log review found matching dimensional collapse despite preserved total
spread. The new v3 pilot uses **MI-only neural alignment**, covariance weight
**0.30** (previously 0.03), and matching-head LR **1e-5** (previously 5e-5).
The solver still fits fused InfoOT with own-encoder cosine cost, MI 0.10,
and entropy 0.02. The relative loss stays at 0.01. Native flow, variance
protection, global InfoNCE with separate cat/dog MLPs, and PCGrad stay enabled.

```text
flow_cat + flow_dog
- 0.02 * alignment_ramp * MI + 0.01 * alignment_ramp * relative
+ 0.10 * variance + 0.30 * covariance
+ 0.05 * translation_ramp * global_source_InfoNCE
```

`decoded_translation.source_contrastive_projection_gradient_scale: 0.10`
applies only to InfoNCE's backward path through conditional matching weights W:

```python
decoded_W = W.detach() + 0.10 * (W - W.detach())
condition = decoded_W @ target_codes
```

The forward weights, conditions, sampled images, scalar loss, and direct
generator/target-code/readout/MLP gradients are unchanged at fixed model state,
inputs and noise. The gradient through W is multiplied by 0.10; encoder gradients
through that path also change. Other loss graphs are untouched. PCGrad then
combines the routed task gradients. Subsequent training trajectories can differ.
The default is 1.0 for existing recipes. The scale is logged in train/validation
and saved in checkpoints; changing it on resume is rejected.

Start both runs fresh from the same original flow-only Stage 1A EMA checkpoints:

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_v3_control_sit_b2.yaml
```

Outputs are `outputs/stage1b_nce_v3` and `outputs/stage1b_nce_v3_g1`. The control
differs only in the output path and gradient scale 1.0. Both run 2,500 steps,
save/validate every 250, and measure gradients every 100. Keep the same Stage 1A
checkpoint files across the pair. Resume each run with its own config and
`--resume latest`; the existing v2 full-objective config remains available.

Evaluate each with its own alignment config/checkpoint and the same evaluator:

```bash
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v3_sit_b2.yaml \
  --checkpoint outputs/stage1b_nce_v3/checkpoints/latest.pt --weights ema
```

The evaluator uses 224 fit references/224 projection targets, fit bandwidth 0.55,
projection bandwidth 0.10 and conditional means. Checkpoint-specific report
subdirectories separate comparisons. Use it for v2 as well before comparing
images. Stage 2–3/4 defaults still point to the established PatchNCE experiment
until the new pilot is evaluated. See the
[v2 analysis](docs/analysis/stage1b_v2_5000/review.md) and
[v3 implementation/tests](docs/analysis/stage1b_v2_5000/implementation.md).
CPU gradient and training/resume tests verify routing; AFHQ quality gains
require the paired GPU runs.

### Stage 1B baseline: learned PatchNCE samplers and reference-EMA RMS

The new `self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml` recipe adds CUT-style
per-layer MLPs with DCLGAN's separate cat/dog sampler routing. Each sampled
feature becomes a 256-dimensional vector through Linear-ReLU-Linear, then L2
normalization and the existing official PatchNCE loss. Source keys are detached;
both samplers learn through the two generated-image query directions.

The sampler learning rate is `2e-4`, with gradient clipping at 1.0. PatchNCE
weight/temperature remain `0.15/0.20`; existing encoder/G learning rates, losses,
PCGrad and reference-EMA RMS settings are unchanged. The samplers have their own
optimizer group and raw/EMA checkpoint state. They are separate from InfoOT's
global matching heads and are not needed to generate images after training.

Start a **fresh** run using the original flow-only Stage 1A checkpoints:

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml
```

Output: `outputs/stage1b_patch_mlp_rms`. Resume it with the same config and
`--resume latest`. Earlier no-MLP Stage 1B checkpoints cannot resume into this
architecture. Evaluate with:

```bash
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml \
  --checkpoint outputs/stage1b_patch_mlp_rms/checkpoints/latest.pt --weights ema
```

The paired evaluator re-encodes generated images and reports PatchNCE using
the checkpoint's selected raw/EMA samplers. Training validation uses raw heads,
as it does for raw E/G. These are learned correspondence scores; use the fixed
image grids and original-RGB reconstruction checks to assess visual benefit.

For an isolated comparison against the older live-RMS PatchNCE control, use
`self_supervised_patchnce_mlp_sit_b2.yaml` in both config directories, output
`outputs/stage1b_patch_mlp`. All original no-MLP/global-InfoNCE recipes remain
available. See [design and comparison protocol](docs/analysis/stage1b_patchnce_mlp/review.md).

### Full neural InfoOT and relative transport experiment

The `self_supervised_patchnce_mlp_full_sit_b2.yaml` recipe adds live encoder-cost
gradients to the neural alignment objective and retains its MI term. Its full
objective is `0.20 * alignment_ramp * (cost - 0.10 * MI - 0.02 * entropy)`.
Entropy is measured on the detached fitted plan and has no neural gradient.
A separate `0.01 * alignment_ramp` term rewards transported centered feature
correlation. Both sides of its independent-matching baseline remain differentiable.

Variance/covariance weights remain `0.05/0.01`; PatchNCE, PCGrad, learning rates,
live fitting RMS, reference-EMA projection RMS, and the original flow-only
Stage 1A checkpoints match the existing MI-only control. This is an additive
experiment: retaining raw cost means the relative term does not guarantee
prevention of feature contraction. Check feature spread/rank and image grids.

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_patchnce_mlp_full_sit_b2.yaml
```

Output: `outputs/stage1b_patch_mlp_full`. Start fresh; after starting this recipe,
resume it with the same config and `--resume latest`. Objective changes across
resume are rejected. Paired evaluation uses
`configs/stage1b_eval/self_supervised_patchnce_mlp_full_sit_b2.yaml`.
See [full objective, gradients, and checks](docs/analysis/stage1b_full_infoot/review.md).

### Global InfoNCE with full InfoOT and revised hyperparameters

`self_supervised_infonce_full_sit_b2.yaml` replaces the full recipe's spatial
PatchNCE with bidirectional global source-code InfoNCE. Each domain has a learned
two-layer MLP (`512 -> 256 -> 256`, ReLU between layers), reusing the CUT sampler's
MLP transform on global codes. The generated dog is read by the dog encoder and
dog MLP, then contrasted against detached cat-MLP embeddings of the original cat
query/reference codes; the reverse direction is symmetric. InfoNCE L2-normalizes
these embeddings. This uses the existing RGB decode/re-encode path and
target-domain readout. The global InfoOT matching heads are separate; spatial
PatchNCE sampling is disabled.

The weighted objective is:

```text
flow_cat + flow_dog
+ alignment_ramp * [0.20 * (cost - 0.10 * MI - 0.02 * entropy) + 0.01 * relative]
+ 0.10 * variance + 0.03 * covariance
+ 0.05 * translation_ramp * global_source_InfoNCE
```

The variance standard-deviation target is **0.80**. Encoder/matching-head LRs
are **1e-5 / 5e-5**; adapter/LoRA LRs remain **5e-6 / 3.75e-6**. InfoNCE temperature
is 0.20, with the existing 0.95 near-duplicate negative filter computed on raw
source codes so a contracting MLP cannot filter away distinct negatives. MLP LR
is **2e-4**, matching the PatchNCE sampler control, with gradient clipping at 1.0.
Each MLP learns through the generated-image query branch; projected source keys
are detached. The heads participate in PCGrad, optimizer state, EMA, and resume.
Full InfoOT,
relative weight 0.01, PCGrad, EMA RMS, original flow-only Stage 1A initializers,
5,000 steps, batches and warmups match the full PatchNCE recipe. The paired
evaluation also retains 512 fit references and all training projection targets.
The fitted plan is detached; entropy has no neural gradient, and PCGrad projects
the weighted task gradients before clipping and Adam.

Start a fresh run:

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_infonce_full_sit_b2.yaml
```

Output: `outputs/stage1b_nce_full`. Resume this experiment using the same config
and `--resume latest`. Use the same fixed Stage 1A checkpoint files for a
comparison; a prior PatchNCE or MI-only Stage 1B checkpoint is not a compatible
resume. A raw-code InfoNCE checkpoint without these MLPs is also incompatible;
start fresh when enabling the heads. Evaluate with:

```bash
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_full_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_full_sit_b2.yaml \
  --checkpoint outputs/stage1b_nce_full/checkpoints/latest.pt --weights ema
```

Global InfoNCE loss/retrieval and original-RGB diagnostics are in training's
`validation.jsonl`. Retrieval is measured after the MLP, while existing raw-code
geometry metrics remain available. Standalone evaluation loads the matching
raw/EMA MLP state and disables the PatchNCE metric. Because
the requested experiment also changes protection and learning rates, comparison
with the old PatchNCE recipe alone does not isolate the loss choice. Quality and
artifact improvement still require a GPU run and matched image panels.
See [experiment details](docs/analysis/stage1b_infonce_full/review.md).

### Reference-EMA RMS projection

The main recipe and the `*_rms_ema_sit_b2.yaml` comparison recipes average reference-feature variances
with decay 0.99 and use their square roots to calibrate conditional projection.
This makes the scale independent of other queries in the batch. InfoOT fitting
and neural MI retain live reference RMS with full gradients; fit/projection
bandwidths remain 0.55/0.10. Losses, weights, learning rates, PCGrad and feature
protection match their controls. The original recipes remain available.

For the **no-MLP comparison**, start the PatchNCE variant fresh from the original flow-only Stage 1A checkpoints:

```bash
python scripts/train_joint_infoot.py --config configs/stage1b_infoot/self_supervised_patchnce_rms_ema_sit_b2.yaml
```

Its output is `outputs/stage1b_self_patch_rms`. Evaluate with the paired recipe:

```bash
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_patchnce_rms_ema_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_patchnce_rms_ema_sit_b2.yaml \
  --checkpoint outputs/stage1b_self_patch_rms/checkpoints/latest.pt --weights ema
```

For the global InfoNCE variant, use `self_supervised_rms_ema_sit_b2.yaml` in both
directories and checkpoint output `outputs/stage1b_self_rms`. Resume each new run
with its own config and `--resume latest`; old Stage 1B checkpoints lack these
statistics and cannot initialize this experiment through resume.

Raw and model-EMA encoder/head weights have separate RMS histories saved in the
checkpoint. Model-EMA tracking costs an additional chunked, no-gradient reference
encoder/head pass per step; it adds no diffusion rollout. Validation/inference
freeze the selected history. Logs expose `projection_rms` scale lag/effective
widths and validation `query_batch_dependence_first_query_l1` in both directions.
EMA projection has detached denominator statistics, so its gradient differs from
live-RMS projection; this is a controlled experiment, not a proven quality gain.
See [design, scope, and comparison protocol](docs/analysis/stage1b_projection_rms_ema/review.md).

Experiment D trains both encoders,
residual matching heads, every added AdaLN/token-MLP conditioning adapter,
`z_proj`, the final adapter, and rank-64 attention LoRA. The pretrained SiT
backbone, VAE, DINOv2 model, and learned null tokens stay frozen. Both Stage 1B
domain branches run on `cuda:0`; the separate Stage 1A training commands may
still use one GPU per domain.

## Installation and data

Install the Python dependencies from the Linux project root. Install a PyTorch
build compatible with the machine's CUDA version when the default wheel is not
appropriate.

```bash
cd /data/not_backed_up/yxu209/diffusion-ot
python3 -m pip install -r requirements.txt
```

Prepare deterministic AFHQ manifests and cache VAE latents before Stage 1A.
The exact data locations are defined in `configs/data/afhq_huggan.yaml`.
Keep the original AFHQ source dataset available: RGB Stage 1A reads the original
images by cached sample metadata, with the same crop/resize and paired flips.
Experiment D also requires the frozen DINOv2 structural descriptor bank:

```bash
python3 scripts/cache_infoot_structure.py --device cuda:0
```

The cache must match the current data split and unflipped preprocessing. The
trainer validates its model revision, descriptor definition, sample IDs, and
content fingerprint.

## Stage 1A

The revised latent-input Stage 1A uses a residual CNN: an unnormalized stride-1
stem, two residual blocks at each of 32/16/8/4 pixels, four-head attention at
16x16, and a 512-dimensional code. Start fresh Cat/Dog runs:

```bash
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml --device cuda:1
```

These flow-only runs retain batch 64, 50,000 steps and encoder/adapter/LoRA
learning rates 1e-4/1e-4/2.5e-5. Outputs are `outputs/stage1a_{cat,dog}_rescnn`.
Evaluate each domain using its exact training config:

```bash
python3 scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml --eval-config configs/stage1a_eval/residual_sit_b2_256.yaml --device cuda:0
python3 scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml --eval-config configs/stage1a_eval/residual_sit_b2_256.yaml --device cuda:1
```

Image metrics use original RGB targets; encoder inputs remain cached VAE
latents. Reports record the encoder specification. Resume with `--resume latest`
only within the corresponding residual run. Old plain-CNN or RGB checkpoints
cannot initialize the new encoder. The original `*_sit_b2_lora.yaml` recipes
remain available for existing checkpoints and current Stage 1B initialization.
Architecture, checkpoint and comparison details are in the
[residual encoder protocol](docs/analysis/stage1a_residual_encoder/review.md).

### Stage 1A time-weighting comparison

The residual-CNN trainer supports `loss_weighting.type: uniform` (velocity
weight 1) and `cosmap` (velocity weight `2 / (pi * (t^2 + (1-t)^2))`). Both
sample time uniformly. These modes accept only the `type` field: remove the
PDAE gamma/normalization/clamping fields when switching a copied config.

Ready-to-run Cat/Dog comparisons retain the same initialization seed, batch
64, 50,000 steps, architecture and learning rates as the existing SNR control:

```bash
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora_residual_uniform.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora_residual_uniform.yaml --device cuda:1
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml --device cuda:1
```

Outputs are `outputs/stage1a_{cat,dog}_rescnn_{uniform,cosmap}`. Resume a run
using its same config and `--resume latest`; changing the flow weighting on
resume is rejected. The original `*_lora_residual.yaml` files retain the SNR
control and their original output directories.

Use the same `configs/stage1a_eval/residual_sit_b2_256.yaml` for evaluation,
passing each experiment's exact training YAML as `--train-config`. Training
logs identify the weighting and include `unweighted_flow_mse`; validation MSE
and time-bin metrics stay unweighted. Compare original-RGB fidelity and code
usefulness, not the differently weighted training losses. Full commands and
metric details: [time-weighting comparison](docs/analysis/stage1a_time_weighting/implementation.md).

### Optional RGB encoder comparison

Start fresh Cat and Dog RGB runs on separate GPUs (no old checkpoint resume):

```bash
python3 scripts/train_pdae_domain.py \
  --config configs/stage1a_pdae/cat_sit_b2_lora_rgb.yaml \
  --device cuda:0

python3 scripts/train_pdae_domain.py \
  --config configs/stage1a_pdae/dog_sit_b2_lora_rgb.yaml \
  --device cuda:1
```

Both configs use original RGB in [-1,1] as the semantic encoder input, six
convolution stages ending at 4x4, and a 512-dimensional code. The diffusion
target remains the cached four-channel VAE latent. They use all 12 SiT-B/2
blocks for AdaLN-Zero/LoRA, rank/alpha 64/64, semantic dropout 0.10, 50,000
updates and batch 64. Encoder/adapter learning rates are 1e-4 and LoRA is 2.5e-5.
No DINO or refinement objective is enabled.

Outputs are `outputs/stage1a_cat_rgb_lora` and `outputs/stage1a_dog_rgb_lora`.
Use `--resume latest` only to continue the corresponding new RGB run. Old
latent-input checkpoints cannot initialize this changed encoder; their configs
remain available unchanged as `*_sit_b2_lora.yaml`, with their original output
directories and checkpoint paths.

For a matched quality comparison, use
`configs/stage1a_eval/rgb_vs_latent_sit_b2_256.yaml` for **both** original and RGB
branches. It scores against original RGB and uses a separate comparison report
directory. The existing `sit_b2_256.yaml` evaluation config stays unchanged.
Match checkpoint steps, samples, noise, guidance and weight mode. The RGB encoder
uses six conv stages versus the original three-stage latent encoder, so this is
a recipe comparison rather than proof of the input representation alone.
See the [RGB encoder protocol](docs/analysis/stage1a_rgb_encoder/review.md)
for commands and comparison criteria.

### Historical Stage 1A image-supervision alternatives

The separate latent-input `*_refine.yaml` recipes retain the earlier
flow + original-RGB DINO + native-code InfoNCE experiment:

```bash
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_refine.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_refine.yaml --device cuda:1
```

Their current configs start fresh for 40,000 updates at the lower historical
learning rates, saving to `outputs/stage1a_{cat,dog}_native40k`.
The `*_dino.yaml` alternatives retain flow + DINO with native-code InfoNCE off,
original learning rates, and `outputs/stage1a_{cat,dog}_dino` outputs.
They require the original AFHQ dataset and the pinned DINO structure cache.
These are separate from the new RGB flow-only experiment. See
[Stage 1A refinement](docs/stage1a_refinement.md) for loss routing,
hyperparameters, validation grids, and checkpoint selection before Stage 1B.

## Experiment D co-training

The active recipe adds teacher-guided matching contrastive losses, decoded
DINO structure InfoNCE, and contrastive generated-code recovery. It retains real-data flow
reconstruction, full-distribution KL, corrected RMS gradients, and matching
variance/covariance protection. The full InfoOT conditional mean still uses
all target references. See the [implementation and pilot record](docs/analysis/stage1b_contrastive_extension/review.md)
for equations, gradient routing, research references, diagnostics, and limitations.

A bounded controller targets a decoded/reconstruction encoder gradient ratio
of **2.0** at full decoded ramp, with scale bounds 0.25–4.0. This is a
translation-first pilot setting, not a demonstrated optimum. The controller
logs the gradient cosine and achieved ratio; it does not project away conflicts.
Generator-code recovery trains G only, using the detached target condition and
current target encoder as a differentiable readout with detached parameters.
It now uses InfoNCE over all 32 current projected query conditions (temperature
0.20, near-duplicate threshold 0.95), with no extra image rollouts. This replaces
the cosine-only recovery objective. See the [code InfoNCE research and implementation note](docs/analysis/stage1b_code_infonce/review.md).

Start fresh from Stage 1A; changed objectives cannot resume old checkpoints:

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --max-steps 4000 --quick-eval
```

Evaluate a saved checkpoint with paired EMA encoder, matching-head, and generator
weights:

```bash
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_decoder_sit_b2.yaml \
  --checkpoint outputs/stage1b_d_nce/checkpoints/latest.pt \
  --no-require-stage1a-baseline
```

Resume only this same objective using `--resume`. The configured pilot is
4,000 updates; `--smoke` uses 500. Review checkpoints every 500 updates and
judge post-ramp translation quality before extending the run. Training writes
`outputs/stage1b_d_nce`; evaluation writes `outputs/stage1b_eval_d_nce`.

| Setting | Value |
| --- | ---: |
| Pilot updates | 4,000 |
| Encoder / matching-head LR | `2e-5` / `2e-4` |
| Adapter / LoRA LR | `5e-6` / `2.5e-6` |
| InfoOT fit / projection bandwidth | `0.55` / `0.10` |
| Entropy regularization | `0.02` |
| Outer OT update budget / tolerance | `1200` / `1e-5`, convergence required |
| Matching variance / covariance weights | `0.05` / `0.01`, from update 1 |
| Matching scaled standard-deviation floor | `0.70`, on `sqrt(dim) * m(z)` |
| Matching neighborhood / conditional contrastive weights | `0.01` / `0.005` |
| Matching contrastive positive neighbors | 8, including boundary ties |
| Decoded structure / adversarial / contrastive weight | `0.10` / `0.01` / `0.01` |
| Generated-code InfoNCE weight, G only | `0.02` |
| Encoder decoded/reconstruction target ratio | 2.0 at full ramp |
| Decoded sampler / ramp | 20 steps / 2,000 updates |
| Validation / checkpoint interval | 500 / 500 updates |

Each update draws 128 samples per domain from the shuffled training split:
96 references for InfoOT/protection and 32 disjoint conditional queries.
Four queries per direction enter decoded losses. Matching contrastive losses
ramp over 1,000 steps; decoded losses and the encoder ratio target ramp over
2,000. Native image quality is evaluated against original dataset RGB, not
Stage 1A outputs. Native conditioned Stage 1A preservation stays disabled.

The prior [5,000-step run](docs/analysis/stage1b_gt_5000/review.md) used entropy
0.05. The active 0.02 user setting is retained; a matched baseline rerun is
needed to isolate the new objectives. Pre-extension config snapshots are saved
with the new report. Earlier detached-RMS, RMS-only, VICReg/cosine, and fit-0.55
control YAMLs remain available for their original experiments.

## Stage 2–3: Frozen Bank Construction and Global InfoOT Fitting

The combined **Stage 2–3: Offline Alignment** workflow first freezes the selected
checkpoint and builds complete feature banks (Stage 2), then fits and exports
**one global fused InfoOT coupling over all training Cats and
Dogs**, with fit bandwidth **0.55** and frozen projection bandwidth **0.10**.
The main config selects **self-supervised PatchNCE + learned MLPs + reference-EMA
RMS**, using the encoder matching-feature cosine cross-cost. No DINO cache is
needed. It includes paired raw/EMA checkpoint restoration, complete-manifest
checks, and resumable global fitting (Stage 3). Matrix tiles are computation chunks, not separate
OT problems. Unequal domain counts are supported.

The `full_bank_calibrated_projection_v2` bundle separates two scales:

- **Fit RMS:** computed from each complete training reference bank.
- **Projection RMS:** copied from the selected checkpoint's reference-EMA
  history, then frozen. `--weights ema` selects the history for model-EMA
  encoder/head weights; `--weights raw` selects the raw history. Neither is
  recalculated from held-out queries or replaced with the full-bank fit RMS.

Stage 4 translates **every held-out validation source** using that exported
coupling. It reports FID against real target-domain validation RGB and SSIM
against each original source RGB, separately in both directions. It saves
**16 source/translation pairs per direction**, including individual PNGs and
labeled contact sheets. The full generated corpus is used for FID.

### Linux commands

Run in the existing training environment (Python 3.10 supported). The configured
Stage 1A checkpoints, pretrained SiT/VAE snapshot,
canonical train/validation manifests, cached latents, and original AFHQ dataset
must be available under the project paths. Inception weights download on the
first Stage 4 metric run and are cached in `outputs/fid_cache`.

```bash
cd /data/not_backed_up/yxu209/diffusion-ot
python -m pip install -r requirements-stage4.txt

# Example only: select an existing numbered checkpoint from the run you want.
CHECKPOINT=outputs/stage1b_patch_mlp_rms/checkpoints/step_004000.pt

# Stage 2-3: frozen banks and global InfoOT fit, paired EMA weights.
python scripts/run_offline_alignment.py \
  --config configs/stage23_offline/full_sit_b2.yaml \
  --checkpoint "$CHECKPOINT" --weights ema --device cuda:0

# Stage 4: use the completed Stage 2-3 bundle; no refitting.
python scripts/evaluate_full_infoot.py \
  --config configs/stage4_eval/fid_ssim_sit_b2.yaml \
  --bundle outputs/s23_rms --device cuda:0
```

Use `--weights raw` on Stage 2–3 if selecting a raw checkpoint evaluation; Stage 4
always inherits the same paired model state. Set Stage 2–3 `alignment_config` to
the configuration matching the checkpoint architecture and Stage 1A provenance.
The final bundle copies learned E/head/G, PatchNCE samplers, and both RMS histories; the large frozen model files
and configuration dependencies remain external and are checked by content hash.
It requires a format-4 Stage 1B checkpoint and never substitutes Stage 1A outputs
as image-quality references.

Both main offline YAMLs require `reference_ema` calibration and reject missing
or mismatched history. The pre-EMA controls remain supported by selecting their
matching `alignment_config` and using `require_projection_rms_mode: full_bank`
in both offline configs, with separate outputs. Their projection uses the old
full-training-bank calibration; the DINO controls still require their cache.
Old v1 bundles must be rebuilt in a new directory. The new defaults
`outputs/s23_rms` and `outputs/s4_rms` keep earlier output directories intact.
`calibration.json`, Stage 4 `protocol.json`/`metrics.json`, and `report.md`
record fit/projection scales and the selected calibration mode for review.

An optional **full-size preflight** measures actual iteration time and GPU
memory without changing the reference count:

```bash
python scripts/run_offline_alignment.py \
  --checkpoint "$CHECKPOINT" --device cuda:0 --max-new-iterations 5
# Continue that same fit; omit --max-new-iterations to finish it.
python scripts/run_offline_alignment.py \
  --checkpoint "$CHECKPOINT" --device cuda:0 --resume

# Resume interrupted evaluation, optionally lowering generation memory use.
python scripts/evaluate_full_infoot.py \
  --bundle outputs/s23_rms --device cuda:0 --resume --generation-batch-size 2
```

Run the preflight instead of the initial Stage 2–3 command, or add `--resume`
when progress already exists. A nonconverged preflight exports no `bundle.json`.
Reaching the outer cap without convergence exits with status 2. Review
`solver_log.jsonl` and `checks.json`; numerical/scientific configuration changes
require a new output directory. Resume validates fingerprints and retains all
global references. Use `--output-dir` for independent checkpoint/configuration
comparisons and point Stage 4 `--bundle` at the corresponding directory.

Memory settings are separate from sample counts. Start with encoding batch 32,
matrix block 512, query projection batch 32, target projection block 512, and
generation batch 4. The exact fit still needs several dense global matrices and
cubic matrix products. At 5,000×5,000, one FP32 matrix is about 95.4 MiB; measure
the full-size preflight on the GPU before estimating runtime. `solver_device`
can explicitly select CPU; there is no hidden subset fallback. `--device`
overrides all devices; omit it when using separate encoding/solver devices.

Outputs:

- `outputs/s23_rms/bundle.json`, `transport.pt`, `banks/`, `models/`,
  `calibration.json`, `solver_progress.pt`, `solver_log.jsonl`, `checks.json`.
- `outputs/s4_rms/metrics.json`, `ssim_per_image.jsonl`,
  `generation_manifest.jsonl`, `gallery_ids.json`, `report.md`.
- `outputs/s4_rms/images/{cat_to_dog,dog_to_cat}/`: complete metric corpus.
- `outputs/s4_rms/real/{cat,dog}/`: original held-out RGB references.
- `outputs/s4_rms/galleries/{cat_to_dog,dog_to_cat}/contact_sheet.png`:
  16 pairs per direction, with source IDs and separate source/translation PNGs.

SSIM measures source preservation, not target-domain fidelity: copying the
source can score highly. FID uses the pinned [Clean-FID implementation](https://github.com/GaParmar/clean-fid)
in clean Inception mode and custom statistics for the actual real images.
SSIM uses the [scikit-image Gaussian convention](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity),
RGB channel averaging and `data_range=1.0`. Metric versions, Inception weight
hash, real/generated counts, and protocol identity accompany the results.
No native reconstruction SSIM or additional quality metrics are computed.

Test the offline mechanism and orchestration with:

```bash
python -m pytest tests/test_offline_stages.py -q
```

## Quick Stage 1B evaluation output

The evaluator fits on 512 training references, projects held-out validation
queries over the full target training bank, and uses `h_proj=0.10` by default.
Use `--projection-bandwidth` only for an explicit sensitivity study; it does
not change the transport-fitting bandwidth.

For the current self-supervised recipes (`readouts: [conditional_mean]`), each
translation grid contains three rows, with corresponding columns:

1. source image (VAE-decoded cached input);
2. original RGB target reference with the highest InfoOT conditional probability;
3. full InfoOT conditional-mean translation.

The v6 evaluator selects `readouts: [conditional_mean, z_cfg_2]` and adds a fourth
row: the same conditional-mean code generated with semantic guidance scale 2.
Both generated rows share their initial noise and sampling steps. `z_cfg_2`
uses `v_null + 2 * (v_z - v_null)` rather than scaling the latent code. Guidance
per readout is recorded in the report and any saved generation inputs.

`translation.include_top1_target` defaults to `true`; set it to `false` to hide
the reference row. The report records its target IDs and probabilities. This
reference is a retrieval diagnostic, not paired ground truth. Additional MAP,
sampled-target, or structure-teacher readouts remain selectable and add generated
rows. The reference row requires original-image metadata in the target bank and
does not perform an additional diffusion rollout.

The report includes reconstruction drift, transport diagnostics, projected-code
geometry, decoded DINO structure errors, and bidirectional proxy precision for
viewpoint, framing, and coat color. Proxy labels are optional JSONL records at
`data/proxy_labels/afhq_viewpoint_framing.jsonl`; missing labels reduce coverage
but do not enter training.

`viewpoint` describes the subject's camera-relative direction, such as front or
side. `framing` describes crop scale, such as close-up or full body. UMAP plots
use translucent target-bank points and larger outlined projection markers so
overlap remains visible. UMAP is a visualization, not the checkpoint-selection
criterion; decoded translation quality and independent proxy metrics are needed.

The v6 evaluation recipes enable the P0 projection audit. Each direction saves
`projections/<direction>.pt` with conditional/base/barycentric codes and weights,
ordered IDs and the effective evaluation protocol. Existing `banks/` files retain
raw and matching codes. `evaluation_protocol.json` records settings, weight
selection and ordered cohorts. Unlabeled plots use different colors for real
targets and projections, with wrapped, uncropped captions.

`audit/<direction>/projection_audit.pt` and its JSON summary include held-out
real-target controls and a same-domain Eq. (7) readout using an identity reference
plan. Its validation queries are excluded from fitting and the target gallery;
cross-domain source-selection calibration is deliberately not reused for this
kernel-only control. Numeric diagnostics cover raw codes, the actual checkpoint
LayerNorm output (including affine parameters and epsilon), and full `z_proj`
output, with norm/rank/variance and Euclidean/cosine nearest-target distributions.
The target bank's own nearest-neighbor diagnostic excludes each point itself.

With visualization enabled, the audit adds target-fitted PCA and UMAP in raw and
decoder-transformed spaces, each with Euclidean and L2-normalized/cosine views.
Shuffled bank subsets and exact selected target codes expose UMAP fit/transform
displacement by ID. Raw joint UMAPs are explicitly **transductive exploratory
plots, not checkpoint-selection scores**. Reducer parameters/version, coordinates,
PCA bases and local reducer pickle files are saved. Only load reducer pickles
you trust. Set `visualization.enabled: false` to keep numeric/PCA tensor controls
without the additional UMAP fits and PNGs; set `projection_audit.enabled: false`
to skip the extra controls entirely.

Recover the matched initial comparison with the saved calibrated step-0 model:

```bash
python scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v6_sit_b2.yaml \
  --checkpoint outputs/stage1b_nce_v6/checkpoints/step_008000.pt \
  --weights ema --evaluate-step0
```

This evaluates the sibling `step_000000.pt` first, then the requested model,
using identical bandwidth overrides, sample limits, IDs, seeds and raw/EMA choice.
Use `--initial-checkpoint PATH` if that saved checkpoint was moved. Explicit
recovery recomputes the initial report. It checks protocol and cohort agreement
before producing deltas; a missing initial checkpoint is reported rather than
replaced by uncalibrated Stage 1A weights. The new audit protocol versions output
directories, so older reports must be rerun for a matched comparison.

### Cached Stage 1B checkpoint/readout screen (P1)

Run the E0 screen with frozen EMA checkpoints at steps 0, 2,500, 5,000 and 8,000:

```bash
python scripts/screen_infoot_checkpoints.py run \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v6_sit_b2.yaml \
  --checkpoint-dir outputs/stage1b_nce_v6/checkpoints \
  --output-dir outputs/stage1b_v6_p1_screen
```

The default screen uses projection bandwidths **0.15, 0.20, 0.25 and 0.35**,
the existing conditional mean/MAP/sample readouts, 20 integration steps, 16
generated images per direction and **three fixed categorical/noise draws**.
Bandwidth 0.10 is excluded because the earlier visual screen found behavior
similar to top-1 selection. Keep 0.25 as the primary existing control and 0.35 as
the broader, blurrier control. Fit bandwidth (0.55), entropy epsilon, checkpoint
RMS scales and source-selection calibration stay fixed. This screen does not
change training or implement tempered means/adaptive controllers. P1a/N1 is
enabled separately by the new residual-cosmap evaluation recipes above.

Use `--checkpoints PATH ...` for an explicit list (saved step 0 first), or
`--checkpoint-steps 0 8000` to narrow a directory run. Missing requested files
fail preflight; checkpoints are never silently substituted. With step 0 first,
every later variant receives its own protocol-matched initial comparison.
Otherwise, the report explicitly records any missing initial comparison.

Models, encoded banks, source appearance descriptors and the fitted transport
plan are cached **within the process for one checkpoint**. Each new checkpoint
gets fresh models/features/fit; changed reference-bank or fit settings also
invalidate the relevant cache entries. Existing `.pt` files are not blindly
reloaded. Query IDs and generation seeds stay fixed across checkpoints and
bandwidths. All readouts in a draw receive the same starting noise; categorical
selection uses a separate private generator. The model's normalization is unchanged.

After image/metric review, explicitly name the finalist bandwidth/readout pairs.
For example, the following is command syntax, **not a measured winner selection**:

```bash
python scripts/screen_infoot_checkpoints.py run \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v6_sit_b2.yaml \
  --checkpoint-dir outputs/stage1b_nce_v6/checkpoints --checkpoint-steps 0 8000 \
  --finalist 0.25:conditional_mean --finalist 0.20:conditional_map \
  --output-dir outputs/stage1b_v6_p1_finalists
```

Finalists default to **64 images, 20 and 40 steps, and three fixed draws**, using
only the named pairs. `--samples`, `--num-steps` and `--draws` override these
settings. Optional `--bank-seeds SEED1 SEED2 SEED3` runs a separate robustness
screen: it changes reference/gallery selection while preserving query and noise
seeds. Bank size remains the evaluation recipe's setting (224 for v6); a larger
bank belongs in a separate evaluation recipe and output directory.

The output contains `screen_manifest.json` (axes, reports, baseline status and
cache hit/miss counts), effective YAML recipes, and `screen_summary.json/.md`.
Each normal evaluation directory retains the P0 tensors and numeric controls;
repeated UMAP rendering is disabled for the sweep. Translation files named
`*_grid_inputs.pt` record the exact starting noise, code inputs, corrected
weights, selected target indices, ordered IDs, seeds, dtype and integration
settings. Reports include target reuse/coverage for MAP/sample. The summary
checks complete saved protocols before computing paired image-loss intervals
for bandwidth or 20/40-step comparisons. Texture discrepancy is an aggregate
diagnostic and does not receive a per-image interval.

To compare copied reports without the checkpoint/data dependencies:

```bash
python scripts/screen_infoot_checkpoints.py summarize \
  --results-dir results/v6.5-p0 \
  --output-dir docs/analysis/stage1b_v6_8000/p1_v65_p0
```

The supplied 0.35 results improve source RGB/Lab losses while worsening texture
and visible sharpness. Choose finalists using source fidelity, target anatomy,
texture and target reuse together; a single scalar loss is insufficient.

## Method boundary

The transport solver and density-ratio conditional readout follow the
[official InfoOT implementation](https://github.com/chingyaoc/InfoOT). The
learned matching heads, frozen DINO structural teacher, differentiable decoder
losses, feature discriminators, and generator adaptation are project-specific
extensions. The conditional readout uses all target samples with target-kernel
smoothing and target-density correction; it is an estimated conditional mean
inside the target-code convex hull, not a retrieved real target or a guaranteed
ground-truth correspondence.

The previous official-based project snapshot remains available at
`legacy/infoot_official_v1`. Verify or test it in an isolated process:

```bash
python3 scripts/run_legacy_infoot.py check
python3 scripts/run_legacy_infoot.py test -q
```

Earlier frozen-generator experiments remain reproducibility controls. Their
results and rationale are archived in `docs/stage1b_guarded_8000_review.md` and
`docs/stage1b_experiment_d.md`; they are not the active training workflow.

## Verification

Run the focused suite with:

```bash
pytest -q tests/test_infoot.py tests/test_stage1b_training.py tests/test_stage1b_eval.py
```

Experiment D has CPU mechanism coverage for selective generator updates,
both-direction decoded gradients, fixed null/base weights, raw/EMA restoration,
sampler gradients, discriminator isolation, deterministic validation, resume,
and decoded-grid metrics. Real AFHQ results still determine whether a checkpoint
is useful for image translation.
