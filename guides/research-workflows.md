# Research workflows and handoff

[Back to the project README](../README.md)

This guide records the operational details behind the concise project overview:
configuration dependencies, checkpoint compatibility, alternative experiments,
evaluation controls, and the artifacts needed to continue the work. The YAML
files and [CLI defaults](../src/diffusion_ot/config_defaults.py) define the actual
recipes; explicit `--config` arguments select the retained alternatives.

## Configuration and required artifacts

Run commands from the repository root. Before using a recipe on a new machine,
set `project_root` in its training, evaluation, data, model, and pretrained
configuration chain to the absolute checkout path. Updating only the training
YAML can leave a nested configuration pointing at the old lab directory.

The default residual-cosmap workflow uses these files:

| Role | Configuration |
| --- | --- |
| AFHQ download, split manifests, and latent locations | [`afhq_huggan.yaml`](../configs/data/afhq_huggan.yaml) |
| SiT-B/2 dimensions and VAE cache settings | [`sit_b2_256.yaml`](../configs/model/sit_b2_256.yaml) |
| Pretrained snapshot location and revision | [`bilisakura_sit_b2_256.yaml`](../configs/pretrained/bilisakura_sit_b2_256.yaml) |
| Stage 1A cat | [`cat_sit_b2_lora_residual_cosmap.yaml`](../configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml) |
| Stage 1A dog | [`dog_sit_b2_lora_residual_cosmap.yaml`](../configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml) |
| Stage 1A evaluation | [`residual_sit_b2_256.yaml`](../configs/stage1a_eval/residual_sit_b2_256.yaml) |
| Stage 1B training | [`self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml`](../configs/stage1b_infoot/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml) |
| Stage 1B evaluation | [`self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml`](../configs/stage1b_eval/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml) |

Use process-visible CUDA indices. With one GPU exposed through
`CUDA_VISIBLE_DEVICES`, use `cuda:0`. Stage 1B has separate `--device-cat` and
`--device-dog` overrides; both branches can use the same device.

Preparation requires the following artifacts:

| Artifact | Preparation or source | Why it is needed |
| --- | --- | --- |
| SiT transformer, scheduler, custom Diffusers code, and VAE | `scripts/verify_sit_snapshot.py --download-if-missing` | Frozen generator backbone and image encoding/decoding |
| Original AFHQ dataset | `scripts/download_afhq_dataset.py` | RGB inputs, original-image metrics, and target references |
| Deterministic cat/dog train and validation manifests | `scripts/prepare_afhq.py` | Stable IDs and disjoint fitting/evaluation populations |
| VAE latents and latent manifests | `scripts/cache_vae_latents.py` | Flow targets and latent-input encoders |
| Frozen SigLIP 2 snapshot, for v2/v2-L and token mapping | `scripts/download_siglip2.py` | Shared patch-token representation |
| DINOv2 structure cache, for teacher-based recipes | `scripts/cache_infoot_structure.py --device cuda:0` | Structural descriptors tied to the original training IDs |

Pin pretrained model revisions before recording results. The default SiT-B/2
config currently specifies `main`; a moving revision is insufficient to identify
a reproducible snapshot. Preserve the downloaded files and identity reports.

AFHQ preparation uses the configured seed and validation fraction, with cats and
dogs active and wild held out. Keep original image metadata and source files
alongside the caches. RGB inputs and original-image metrics cannot be recovered
from the cached VAE latents alone.

## Default learned-vector workflow

### Stage 1A: residual-cosmap autoencoders

The residual encoder takes four-channel, 32×32 VAE latents. It has a raw stride-1
convolution stem, two residual blocks per stage at 32/16/8/4 pixels, attention at
16×16, internal GroupNorm, and a normalized 512-dimensional code. Conditioning
adapters and rank-64 attention LoRA train across all 12 SiT-B/2 blocks; the base
SiT and VAE remain frozen.

Both domains use batch 64, 50,000 optimizer updates, and encoder/adapter/LoRA
learning rates `1e-4 / 1e-4 / 2.5e-5`. The cosmap flow loss samples time uniformly
and weights velocity MSE by:

```text
w(t) = 2 / (pi * (t^2 + (1 - t)^2))
```

This weight is applied directly, without an additional SNR multiplier. Native
validation and explicit unweighted flow diagnostics retain their unweighted
definitions. Do not compare differently weighted training losses as image-quality
scores.

Use the commands in the [README](../README.md#getting-started). For an initial
path/configuration check, or to continue the same run:

```bash
python scripts/train_pdae_domain.py --domain cat --dry-run
python scripts/train_pdae_domain.py --domain cat --device cuda:0 --resume latest
```

The dog command uses `--domain dog`. Outputs are
`outputs/stage1a_{cat,dog}_rescnn_cosmap`; numbered checkpoints are retained.
Old plain-CNN or RGB checkpoints do not match this encoder.

### Stage 1B: v7 alignment and co-training

Select Stage 1A checkpoints using native reconstruction and correct/shuffled
conditioning controls. Pin those numbered files in the v7 training YAML.
The recipe requires residual latent encoders, cosmap weighting, semantic CFG,
and the specified attention LoRA. Replacing paths to an incompatible checkpoint
does not change these requirements.

A fresh run rebuilds transport references, plans, matching heads, optimizer/EMA
state, and calibration from training data. It retains `step_000000.pt` for
protocol-matched initial evaluation. The default budget is 8,000 updates, with
checkpointing and validation every 250.

| Setting | Default v7 value |
| --- | --- |
| Fit / projection bandwidth multipliers | `0.55 / 0.25` |
| Entropy epsilon | `0.02` |
| Conditional training readout | Differentiable conditional mean |
| Projection scale calibration | Reference-variance EMA, decay `0.99` |
| Spatial cost | Own-encoder descriptors, layers 0/1, 8×8 grid |
| Source-aware selection | Spatial and appearance compatibility |
| Source contrastive objective | Separate global InfoNCE MLPs |
| Gradient combination | PCGrad |

Residual layers 0/1 refer to the 32/16-pixel stages after their residual blocks
and attention. They differ from the earlier plain-CNN feature stages.

```bash
python scripts/train_joint_infoot.py --dry-run
python scripts/train_joint_infoot.py --quick-eval
python scripts/train_joint_infoot.py --resume latest
```

For a short training pilot, `--smoke` selects the recipe's 500 updates.
It is still a training run with the same artifact prerequisites. Use a separate
output directory for an independent pilot.

Resume restores the same experiment. Changes to architecture, native flow
weighting/direction/timestep clamp, or guarded objective settings are rejected.
Fresh runs also refuse to overwrite an existing latest checkpoint. Use a new
recipe/output directory when changing the scientific setup.

### Fitting scales and projection scales

The fitting objective uses live reference-feature RMS with its gradient route.
Reference-EMA projection uses saved running variances and detached denominator
statistics. These are different calibration mechanisms.

Raw and model-EMA encoder/matching-head weights have separate projection RMS
histories. Evaluation freezes the history corresponding to `--weights raw` or
`--weights ema`; held-out query batches must not recalibrate it. Check
`projection_rms` diagnostics for scale lag and query-batch dependence.

The spatial and source-selection scales are calibrated once from training-only
references and saved in the checkpoint. Their descriptors/added selection costs
are detached where specified by the recipe; the base conditional probabilities
keep their existing gradient route.

## Evaluation and diagnostics

### Stage 1A conditioning and reconstruction

The default residual evaluator generates an 8-image fixed-noise smoke cohort,
with correct/shuffled conditions and guidance controls. It separately evaluates
an inferred-noise round trip and runs the enabled appearance probe.
`dataset.num_samples: 256`, `metrics.full`, and other planned full-protocol fields
do not make this CLI run a full image benchmark. Read the actual report counts;
increase `dataset.smoke_samples` deliberately for a larger generation cohort.

Use the exact training configuration and a fixed numbered checkpoint when
comparing runs. Image metrics reference original RGB rather than Stage 1A output
or VAE reconstruction. The VAE reconstruction is a separate control.

### Stage 1B matched evaluation

The default v7 evaluator uses 224 fitting references and 224 target projection
codes per domain, held-out validation queries, and 16 generated images per
direction at 20 integration steps. It displays conditional-mean and MAP readouts.

```bash
python scripts/evaluate_infoot_alignment.py \
  --checkpoint outputs/stage1b_nce_v7_residual_cosmap/checkpoints/latest.pt \
  --evaluate-step0 --weights ema
```

`--evaluate-step0` evaluates the sibling calibrated `step_000000.pt` first.
Use `--initial-checkpoint PATH` if it was moved. The evaluator checks protocols,
ordered cohorts, and weight selection before computing comparison deltas.
It does not substitute uncalibrated Stage 1A weights for a missing initial model.

Useful readout distinctions:

- **Conditional mean:** weighted target-code average; the matched training readout.
- **MAP:** the highest-probability target code, generated through the decoder.
- **Sampled target:** a categorical target-code draw; requires fixed draw seeds.
- **Top-1 reference image:** an original training image shown for context, not a
  generated output or paired ground truth.
- **Semantic CFG:** combines conditioned and null velocity predictions. The v6
  `z_cfg_2` panel uses `v_null + 2 * (v_z - v_null)`; it does not scale the code.

Projection bandwidth overrides change the conditional readout, not the fitted
coupling. Compare with fixed model state, IDs, initial noise, integration steps,
and raw/EMA selection.

### Appearance probes and projection audits

The enabled `input_statistics` probe uses up to 1,024 training and 256 development
images, without augmentation or additional diffusion rollouts. It saves latent
channel means/stds, original RGB/Lab summaries, encoder codes before/after
normalization, and the actual generator LayerNorm/`z_proj` features.

Ridge probes compare codes alone, codes plus eight latent statistics, and
statistics-only controls. Scaling and ridge selection use an internal training
holdout; development data is used for reporting. A positive
`added_statistics_mse_reduction` means the extra statistics improved that probe.
It does not establish that image information was lost, and VAE channels are not
RGB channels. The adjacent `appearance_probe.json` and `appearance_probe.pt`
retain metrics, cohort IDs, features, fitted coefficients, and predictions.

The projection audit saves codes and conditional weights in raw and
decoder-transformed spaces. It includes held-out real-target and same-domain
controls, feature spread/rank, distances to target codes, and PCA/UMAP views.
Joint UMAP plots are exploratory and transductive; they are not a checkpoint
ranking criterion. Disable `visualization.enabled` to avoid repeated UMAP fits,
or `projection_audit.enabled` to skip the additional controls.

### Cached checkpoint/readout screening

The P1 screen reuses models, banks, calibration, and a fitted transport plan
within one checkpoint across readout variants. Each different checkpoint gets
fresh features and fitting. Query IDs and generation seeds remain fixed.

Choose checkpoint steps that exist in your run; for example:

```bash
python scripts/screen_infoot_checkpoints.py run \
  --alignment-config configs/stage1b_infoot/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml \
  --eval-config configs/stage1b_eval/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml \
  --checkpoint-dir outputs/stage1b_nce_v7_residual_cosmap/checkpoints \
  --checkpoint-steps 0 8000 \
  --output-dir outputs/stage1b_v7_screen
```

Default screening compares bandwidths 0.15/0.20/0.25/0.35, mean/MAP/sample
readouts, 16 images per direction, 20 steps, and three fixed draws. Put saved
step 0 first for matched initial comparisons. `--finalist BANDWIDTH:READOUT`
selects explicit pairs for a larger 64-image, 20/40-step comparison.
`--bank-seeds` changes reference/gallery populations in a separate robustness
screen; it does not change query/noise seeds.

Outputs include `screen_manifest.json`, `screen_summary.json/.md`, effective
YAMLs, evaluation protocols, and per-grid generation inputs. Saved inputs record
noise, codes, conditional weights, target selections, ordered IDs, and sampling
settings. Review source preservation, target anatomy, texture, and target
reuse together rather than ranking by one scalar loss.

To summarize the committed historical reports without loading checkpoints or
the AFHQ dataset:

```bash
python scripts/screen_infoot_checkpoints.py summarize \
  --results-dir results/v6.5-p0 --output-dir outputs/v6_projection_summary
```

## SigLIP-conditioned autoencoders

PDAE v2 and v2-L are separate Stage 1A experiments. Original RGB is encoded by
frozen `google/siglip2-base-patch16-224` into 196×768 patch tokens. A shared
LayerNorm/MLP feeds independent image cross-attention branches between each
SiT block's self-attention and FFN. New output projections start at zero.

The projector, image-attention branches, learned CFG null token, and rank-64
attention LoRA train; SigLIP, the base SiT, and VAE stay frozen. Training uses
cosmap-weighted flow without refinement losses.

| Setting | v2, SiT-B/2 | v2-L, SiT-L/2 |
| --- | --- | --- |
| Training configs | [Cat](../configs/stage1a_pdae_v2/cat.yaml), [dog](../configs/stage1a_pdae_v2/dog.yaml) | [Cat](../configs/stage1a_pdae_v2_l/cat.yaml), [dog](../configs/stage1a_pdae_v2_l/dog.yaml) |
| Evaluation config | [`pdae_v2.yaml`](../configs/stage1a_eval/pdae_v2.yaml) | [`pdae_v2_l.yaml`](../configs/stage1a_eval/pdae_v2_l.yaml) |
| SiT blocks / hidden width | 12 / 768 | 24 / 1,024 |
| New conditioning-layer LR | `1e-4` | `5e-5` |
| LoRA LR | `2.5e-5` | `1.25e-5` |
| Linear LR warmup | 500 optimizer updates | 1,000 optimizer updates |
| Effective batch | 64 | 64 |
| Current configured budget | 50,000 updates per domain | 30,000 updates per domain |
| Output directories | `outputs/pdae_v2_{cat,dog}_aug` | `outputs/pdae_v2_l_{cat,dog}` |

These learning rates are experiment settings, not an established scaling rule.
Both paths use float32 parameters/optimizer state, AdamW weight decay 0.01,
gradient clipping 1.0, EMA decay 0.999, and matched raw/EMA validation on 32 fixed
images every 1,000 updates. Logs save every 500 updates and checkpoints every
5,000. For less memory on L, batch 8 with accumulation 8 retains effective batch
64; measure actual GPU memory and throughput.

Matched augmentation applies the same transformed RGB to SigLIP and the VAE
flow target: horizontal flip probability 0.5, mild color jitter probability
0.25, and small affine changes with probability 0.2. VAE encoding is chunked at
16 images. This bypasses cached-latent flips; validation remains unaugmented.

### Prepare, train, and evaluate

For v2-B, use the shared SiT-B/2 dataset and latent preparation from the README,
then download the pinned SigLIP snapshot and select explicit configs:

```bash
python scripts/download_siglip2.py
python scripts/train_pdae_domain.py --config configs/stage1a_pdae_v2/cat.yaml
python scripts/train_pdae_domain.py --config configs/stage1a_pdae_v2/dog.yaml
python scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae_v2/cat.yaml --eval-config configs/stage1a_eval/pdae_v2.yaml --weights both
python scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae_v2/dog.yaml --eval-config configs/stage1a_eval/pdae_v2.yaml --weights both
```

For v2-L, configure its separate [data](../configs/data/afhq_pdae_v2_l.yaml),
[model](../configs/model/sit_l2_256.yaml), and
[pretrained snapshot](../configs/pretrained/bilisakura_sit_l2_256.yaml) paths.
The supplied SiT-L/2 snapshot pins commit
`2f02393a3079dfa6210c10302cc927f59d2cb142`.

```bash
python scripts/download_sit_l.py
python scripts/download_siglip2.py
python scripts/prepare_afhq.py --config configs/data/afhq_pdae_v2_l.yaml
python scripts/cache_pdae_v2_l_latents.py --device cuda:0 --batch-size 16
python scripts/train_pdae_domain.py --config configs/stage1a_pdae_v2_l/cat.yaml
python scripts/train_pdae_domain.py --config configs/stage1a_pdae_v2_l/dog.yaml
python scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae_v2_l/cat.yaml --eval-config configs/stage1a_eval/pdae_v2_l.yaml --weights both
python scripts/evaluate_pdae_domain.py --train-config configs/stage1a_pdae_v2_l/dog.yaml --eval-config configs/stage1a_eval/pdae_v2_l.yaml --weights both
```

The L VAE has a separate latent cache. Do not substitute SiT-B weights or overwrite
the B cache. The pretrained downloader includes transformer, custom Diffusers
code, scheduler, and bundled VAE.

Evaluation uses 16 fixed images in batches of eight and correct/shuffled
conditions at CFG 1/1.5/2. The L grid shows one original-image row followed by
six generated rows; VAE reconstruction metrics remain in the report even though
the VAE row is hidden. These token recipes disable the vector-only appearance
ridge probe and default to no inferred-noise round trip.

`--weights both` writes matched raw/EMA reports and
`raw_ema_comparison.json`. Use `--checkpoint` to pin a numbered step;
`--roundtrip` enables the optional inversion diagnostic.
`--solver heun --num-steps 50` selects the predictor/corrector sampler.
Heun uses two velocity evaluations per step; the legacy midpoint-time Euler
convention is still first order. Compare sampling at equal model-evaluation
cost, with fixed images and noise.

### Resume and snapshot identity

Resume the corresponding run using its explicit config and `--resume latest`.
For v2-B, `latest` resolves inside the configured `*_aug` directory; an older
run in `outputs/pdae_v2_cat` or `outputs/pdae_v2_dog` requires an explicit
checkpoint path for the first continuation. Preserve the original LR schedule
when resuming an older constant-LR run. A separate warm-start experiment uses
`train.initialize_from` and a new output directory.

SigLIP checkpoints retain the frozen encoder/processor identity. Subsequent
training/evaluation uses the local snapshot. A missing `pdae_v2_snapshot.json`
can be restored from a trained checkpoint only when the local files match its
saved hashes. To recover missing files at the checkpoint's exact revision:

```bash
python scripts/download_siglip2.py --checkpoint outputs/pdae_v2_l_cat/checkpoints/latest.pt
```

This checks matching local files first and otherwise downloads the saved revision.
It does not select the latest Hub revision. Changing EMA decay affects future
updates without replacing existing saved EMA history.
`--reset-ema-on-resume` explicitly resets the shadow average and is not a normal
continuation step.

### Standalone token transport

Joint neural Stage 1B token-transport training is not implemented. The separate
[`infoot_vit/` workflow](../infoot_vit/README.md) fits offline mappings over frozen
SigLIP tokens and generates with an already trained v2/v2-L target decoder.

Bank extraction uses original unaugmented RGB and retains the complete
196×768 feature map, stable IDs, preprocessing identity, and encoder hashes.
Both domains must share the same snapshot and feature definition. A trained
domain decoder is needed for generation, but not for frozen bank extraction.

```bash
python -m pip install -r infoot_vit/requirements.txt
python infoot_vit/bank/bank_cat.py --split train
python infoot_vit/bank/bank_dog.py --split train
python infoot_vit/bank/bank_cat.py --split val
python infoot_vit/bank/bank_dog.py --split val
```

Follow the component guide for whole-map, global patch, grouped, and partial
mapping commands. The [low-rank guide](../infoot_vit/lowrank/README.md) records the
factorized transport/kernel objectives, approximation checks, memory settings,
and checkpoint formats. Low-rank modes are research approximations requiring
comparison against dense controls.

Projection bandwidth and top-k overrides operate on a saved mapper. They do not
refit the coupling or add image pairs absent from the saved pair inventory.
Top-k 1 is a retrieval/decoder control, not evidence of source-preserving
translation. Use the saved query protocol and generation report when comparing.

## Retained experimental recipes

Historical configs remain explicit alternatives. Use the exact recipe belonging
to a checkpoint, rather than renaming it or replacing checkpoint paths in a
different architecture. Keep a new output directory for an independent objective
or representation comparison.

### Stage 1A alternatives

Each cat recipe has a matching dog file in the same directory.

| Recipe | Purpose |
| --- | --- |
| [Original latent CNN](../configs/stage1a_pdae/cat_sit_b2_lora.yaml) | Flow-only initialization used by older Stage 1B controls |
| [Residual SNR control](../configs/stage1a_pdae/cat_sit_b2_lora_residual.yaml) | Same residual architecture with historical SNR weighting |
| [Residual uniform control](../configs/stage1a_pdae/cat_sit_b2_lora_residual_uniform.yaml) | Uniform velocity-MSE weight; compare against cosmap with matched settings |
| [RGB encoder](../configs/stage1a_pdae/cat_sit_b2_lora_rgb.yaml) | Original RGB input and a separate six-stage CNN; target remains a VAE latent |
| [DINO-supervised branch](../configs/stage1a_pdae/cat_sit_b2_dino.yaml) | Flow plus structural supervision |
| [Refinement branch](../configs/stage1a_pdae/cat_sit_b2_refine.yaml) | Flow, original-RGB DINO, and native-code InfoNCE |

The RGB-versus-latent comparison uses
[`rgb_vs_latent_sit_b2_256.yaml`](../configs/stage1a_eval/rgb_vs_latent_sit_b2_256.yaml).
Both input space and encoder architecture differ, so it is a recipe comparison,
not an isolated test of RGB versus latent input. Teacher-based recipes require
the DINO cache and original images.

### Stage 1B alternatives

| Training recipe | Main experimental change |
| --- | --- |
| [Global InfoNCE control](../configs/stage1b_infoot/self_supervised_sit_b2.yaml) | Self-supervised source-code contrastive objective |
| [PatchNCE control](../configs/stage1b_infoot/self_supervised_patchnce_sit_b2.yaml) | Spatial source correspondence loss |
| [Global InfoNCE + RMS EMA](../configs/stage1b_infoot/self_supervised_rms_ema_sit_b2.yaml) | Running reference calibration for projection |
| [PatchNCE + RMS EMA](../configs/stage1b_infoot/self_supervised_patchnce_rms_ema_sit_b2.yaml) | Same calibration with no learned patch MLPs |
| [Learned PatchNCE baseline](../configs/stage1b_infoot/self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml) | Domain-specific CUT-style MLP samplers plus RMS EMA; supplied offline-export recipe |
| [Learned PatchNCE, live RMS](../configs/stage1b_infoot/self_supervised_patchnce_mlp_sit_b2.yaml) | Calibration control for the learned-sampler baseline |
| [Full neural InfoOT + PatchNCE](../configs/stage1b_infoot/self_supervised_patchnce_mlp_full_sit_b2.yaml) | Live encoder-cost gradients and relative alignment |
| [Full neural InfoOT + global MLP InfoNCE](../configs/stage1b_infoot/self_supervised_infonce_full_sit_b2.yaml) | Global contrastive readout with revised protection weights/rates |
| [v3](../configs/stage1b_infoot/self_supervised_infonce_v3_sit_b2.yaml) / [gradient control](../configs/stage1b_infoot/self_supervised_infonce_v3_control_sit_b2.yaml) | MI-only neural alignment, stronger covariance protection, and conditional-weight gradient routing |
| [v4](../configs/stage1b_infoot/self_supervised_infonce_v4_sit_b2.yaml) | Coarse RGB, RGB-uv histogram, local layout, and target texture losses |
| [v4.5](../configs/stage1b_infoot/self_supervised_infonce_v4_5_sit_b2.yaml) | Own-encoder spatial relationships in transport fitting |
| [v6](../configs/stage1b_infoot/self_supervised_infonce_v6_sit_b2.yaml) / [bandwidth control](../configs/stage1b_infoot/self_supervised_infonce_v6_bandwidth_control_sit_b2.yaml) | Source-aware spatial/appearance selection and normalized Lab patch SWD |
| [Current Experiment D](../configs/stage1b_infoot/structure_decoder_sit_b2.yaml) | DINO-guided conditional matching, perceptual InfoNCE, RGB PatchGAN, and source color histogram |
| [Earlier full Experiment D control](../configs/stage1b_infoot/structure_decoder_full_sit_b2.yaml) | Additional matching contrasts, gradient-ratio controller, and generated-code recovery |

Older self-supervised controls initialize from the original flow-only
`*_sit_b2_lora.yaml` EMA checkpoints. Experiment D uses its selected
`*_sit_b2_dino.yaml` checkpoints. Neither automatically adopts the residual or
SigLIP outputs.

Train a selected recipe with `scripts/train_joint_infoot.py --config PATH`.
For evaluation, pass its training YAML as `--alignment-config` and the matching
file from [`configs/stage1b_eval/`](../configs/stage1b_eval/) as `--eval-config`.
Where supplied, `quick_evaluation.config` identifies the paired evaluator.

Important mechanism distinctions retained from these experiments:

- Learned PatchNCE uses separate domain samplers and detached source keys. Its
  learned scores measure correspondence under the trained readout; samplers are
  not needed for image generation.
- Full neural InfoOT adds live feature-cost gradients and MI. Plan entropy is
  measured on a detached fitted coupling and has no neural gradient. A relative
  term alone does not guarantee protection against feature contraction.
- v3 routes 10% of decoded InfoNCE gradients through conditional weights while
  retaining the forward weights/conditions. Its control routes 100%.
  Covariance weight is 0.30 and matching-head LR is `1e-5`.
- v4 groups the decoded image terms into one PCGrad task and applies the existing
  conditional-weight gradient gate to all decoded terms.
- v4.5 mixes encoder and own-feature spatial costs at 0.80/0.20, with fixed
  training-only RMS calibration. Spatial descriptors are detached.
- v6 adds detached source spatial/appearance costs and Lab patch SWD weight 0.04.
  RGB-uv histogram and target texture training weights are zero; target texture
  remains a diagnostic. Fit/projection bandwidths are 0.55/0.25.
- Current Experiment D uses 5,000 updates, a 256-image transport batch per domain
  split into 224 references and 32 disjoint queries, and 32 decoded queries per
  direction with a 20-step sampler. Perceptual InfoNCE, adversarial, and source
  histogram weights are 0.10/0.01/0.02. PCGrad is enabled; the gradient-ratio
  controller, extra matching contrasts, and generated-code recovery are disabled.
- The earlier full Experiment D control uses 4,000 updates, 96 references and
  32 disjoint queries, and the bounded encoder gradient-ratio target 2.0. This
  is a pilot setting, not a measured optimum. Its generated-code recovery trains
  the generator with detached conditions and a frozen-parameter encoder readout.

## Full-bank export and final evaluation

This is a separate, retained workflow for compatible learned-vector checkpoints.
The supplied [Stage 2–3 config](../configs/stage23_offline/full_sit_b2.yaml)
selects the learned PatchNCE + reference-EMA RMS recipe, with fit bandwidth 0.55
and projection bandwidth 0.10.

**The exporter rejects spatial-cost recipes**, including v4.5, v6, and v7.
Changing `alignment_config` to v7 does not make this path supported. Extending
full-bank spatial calibration/cost construction and its evaluation is remaining
implementation work.

For a supported recipe, Stage 2 builds complete frozen training banks and Stage 3
fits one global fused InfoOT coupling. Matrix tiles are computation chunks,
not independent transport problems. Domain counts may differ. The workflow
requires a format-4 paired Stage 1B checkpoint and validates dependency hashes
and full manifest coverage.

Choose an existing numbered checkpoint from the configured PatchNCE run:

```bash
python -m pip install -r requirements-stage4.txt
CHECKPOINT=outputs/stage1b_patch_mlp_rms/checkpoints/step_NNNNNN.pt

python scripts/run_offline_alignment.py \
  --config configs/stage23_offline/full_sit_b2.yaml \
  --checkpoint "$CHECKPOINT" --weights ema --device cuda:0

python scripts/evaluate_full_infoot.py \
  --config configs/stage4_eval/fid_ssim_sit_b2.yaml \
  --bundle outputs/s23_rms --device cuda:0
```

Replace `step_NNNNNN.pt` with a file that actually exists. `--weights raw` selects
the paired raw model and its calibration; Stage 4 inherits the bundle's state
and does not refit it.

Full-bank **fitting RMS** is computed from complete training reference banks.
**Projection RMS** comes from the selected checkpoint's matching raw/EMA history
and remains frozen. The supplied offline configs require `reference_ema`.
Pre-EMA controls need their own matching recipe, `require_projection_rms_mode:
full_bank` in both configs, and distinct outputs. Old bundle protocols require
rebuilding in a new directory.

A full-size preflight can measure iteration cost without reducing references:

```bash
python scripts/run_offline_alignment.py --checkpoint "$CHECKPOINT" --device cuda:0 --max-new-iterations 5
python scripts/run_offline_alignment.py --checkpoint "$CHECKPOINT" --device cuda:0 --resume
python scripts/evaluate_full_infoot.py --bundle outputs/s23_rms --device cuda:0 --resume --generation-batch-size 2
```

Run the preflight instead of the first full fitting command, or add `--resume`
when progress exists. No completed `bundle.json` is exported before convergence.
Reaching the outer iteration cap exits with status 2. Inspect `solver_log.jsonl`
and `checks.json`; configuration/population changes require a new output directory.

The exact fit retains dense global matrices and cubic matrix products.
One 5,000×5,000 FP32 matrix is about 95.4 MiB, but total memory includes several
matrices and workspace. Measure the full preflight rather than estimating total
memory from one array. Encoding, fitting, projection, and generation batch/block
sizes are separate from reference counts; there is no automatic subset fallback.

Stage 4 generates every held-out validation source, computes FID against original
target-domain validation images and SSIM against original source images, and
saves 16 source/translation pairs per direction. The initial FID run downloads
Inception weights into its configured cache.

| Output | Contents |
| --- | --- |
| `outputs/s23_rms/` | `bundle.json`, `transport.pt`, banks, paired learned model state, `calibration.json`, fitting progress/logs/checks |
| `outputs/s4_rms/` | `metrics.json`, per-image SSIM, generation manifest, gallery IDs, protocol, and `report.md` |
| `outputs/s4_rms/images/` | Complete generated metric corpus in both directions |
| `outputs/s4_rms/real/` | Original held-out real-domain references |
| `outputs/s4_rms/galleries/` | Contact sheets and individual source/translation pairs |

FID uses [Clean-FID](https://github.com/GaParmar/clean-fid) in clean Inception mode.
Source SSIM uses [scikit-image's Gaussian convention](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity),
RGB averaging, and `data_range=1.0`. Reports record versions, weight identity,
counts, and protocol. Source SSIM measures preservation and can reward copying
the source; it does not measure target-domain fidelity.

## Handoff inventory

A complete handoff includes source code plus the external artifacts required by
the chosen workflow. Keep the original experiment configurations with the model;
record path adaptations separately when moving to another machine.

| Item to preserve | What a successor needs |
| --- | --- |
| Source revision and configs | Git commit; exact training/evaluation YAMLs; launch commands; Python/CUDA/package environment |
| Pretrained snapshots | SiT transformer, scheduler, custom code, VAE; SigLIP or DINO files when used; exact revisions and hashes |
| Data | Original AFHQ source, canonical train/validation manifests, stable IDs, latent files and metadata |
| Stage 1A initialization | Selected numbered checkpoints and their exact domain recipes |
| Stage 1B training state | Numbered selected checkpoint, saved step 0, optimizer/EMA, matching heads, trainable adapters/LoRA, samplers, and calibration histories |
| Offline mapping | Completed bundle or mapper directory, banks, plans/factors, pair inventory, masks, manifests, and external dependencies |
| Evaluation evidence | Effective protocol, IDs/seeds, raw/EMA selection, grids, JSON reports, logs, and comparison summaries |

Bundles copy learned state but still depend on the frozen models and data/config
files checked by identity. Result directories contain evidence of a run, not all
its training or inference prerequisites.

Priorities for continuing the research:

1. Reproduce one selected baseline with the same checkpoint, cohort, and noise.
2. Separate conditional-mean blur from decoder quality using matched native
   target, MAP, and sampled-target controls.
3. Assess source layout/appearance, target anatomy/texture, and target reuse
   together; learned losses and dimensionality plots are supporting diagnostics.
4. Validate low-rank kernel/transport approximations against dense controls
   before drawing conclusions about image quality or scalability.
5. Implement and verify full-bank export for spatial-cost recipes if a v7
   end-to-end benchmark is the next objective.

Use the focused and full checks documented in the [README](../README.md#verification)
after implementation changes. CPU mechanism tests verify numerical and workflow
behavior; scientific claims still require matched AFHQ GPU evaluations.
