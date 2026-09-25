# diffusion-ot

Cat/Dog PDAE representations with InfoOT conditional projection and diffusion
translation. **Original latent-input, flow-only LoRA remains the main Stage 1A
experiment**, with its configs/checkpoints unchanged. A separate optional RGB
encoder experiment supports a quality comparison. SiT diffusion still operates
on VAE latents in both. Stage 1B includes self-supervised global InfoNCE/PatchNCE controls and
the external-supervision **Experiment D**; each retains its explicitly selected
earlier Stage 1A architecture/checkpoints pending a separate RGB migration.

## Active workflow

The current pipeline has four stages:

1. Train separate Cat and Dog encoders/conditioning paths with semantic CFG,
   rank-64 LoRA and latent flow loss only using the original latent-input main
   recipe. The optional RGB experiment runs separately for comparison.
2. Run the selected compatible Stage 1B experiment with full InfoOT conditional
   means. Existing recipes remain on their earlier latent-input checkpoints.
3. Evaluate held-out queries with the complete target projection bank.
4. Export the selected encoders, matching heads, adapted generators, kernels,
   target codes, and fitted transport for downstream inference.

The canonical configurations are:

- Stage 1A Cat / Dog main latent-input recipe: `configs/stage1a_pdae/{cat,dog}_sit_b2_lora.yaml`
- Stage 1A Cat / Dog optional RGB comparison: `configs/stage1a_pdae/{cat,dog}_sit_b2_lora_rgb.yaml`
- Stage 1B self-supervised PatchNCE: `configs/stage1b_infoot/self_supervised_patchnce_sit_b2.yaml`
- Stage 1B global InfoNCE control: `configs/stage1b_infoot/self_supervised_sit_b2.yaml`
- Stage 1B PatchNCE + reference-EMA RMS experiment: `configs/stage1b_infoot/self_supervised_patchnce_rms_ema_sit_b2.yaml`
- Stage 1B global InfoNCE + reference-EMA RMS experiment: `configs/stage1b_infoot/self_supervised_rms_ema_sit_b2.yaml`
- Experiment D: `configs/stage1b_infoot/structure_decoder_sit_b2.yaml`
- Experiment D evaluation: `configs/stage1b_eval/structure_decoder_sit_b2.yaml`

The self-supervised controls use original flow-only EMA checkpoints with the
unchanged `*_sit_b2_lora.yaml` configs. Experiment D uses its selected
`*_sit_b2_dino.yaml` EMA checkpoints. Neither automatically loads the new RGB
Stage 1A outputs; replacing checkpoint paths alone is insufficient.

### Reference-EMA RMS projection experiment

The optional `*_rms_ema_sit_b2.yaml` recipes average reference-feature variances
with decay 0.99 and use their square roots to calibrate conditional projection.
This makes the scale independent of other queries in the batch. InfoOT fitting
and neural MI retain live reference RMS with full gradients; fit/projection
bandwidths remain 0.55/0.10. Losses, weights, learning rates, PCGrad and feature
protection match their controls. The original recipes remain available.

Start the PatchNCE variant fresh from the original flow-only Stage 1A checkpoints:

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

The original latent-input Stage 1A remains main. Its training commands, configs,
checkpoint paths and evaluation config are unchanged:

```bash
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/cat_sit_b2_lora.yaml --device cuda:0
python3 scripts/train_pdae_domain.py --config configs/stage1a_pdae/dog_sit_b2_lora.yaml --device cuda:1
```

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
It includes paired raw/EMA checkpoint
restoration, complete-manifest checks, fixed training-reference RMS calibration,
and resumable global fitting (Stage 3). Matrix tiles are computation chunks, not separate
OT problems. Unequal domain counts are supported.

Stage 4 translates **every held-out validation source** using that exported
coupling. It reports FID against real target-domain validation RGB and SSIM
against each original source RGB, separately in both directions. It saves
**16 source/translation pairs per direction**, including individual PNGs and
labeled contact sheets. The full generated corpus is used for FID.

### Linux commands

Run in the existing training environment (Python 3.10 supported). The configured
Stage 1A checkpoints, pretrained SiT/VAE snapshot, DINO structure cache,
canonical train/validation manifests, cached latents, and original AFHQ dataset
must be available under the project paths. Inception weights download on the
first Stage 4 metric run and are cached in `outputs/fid_cache`.

```bash
cd /data/not_backed_up/yxu209/diffusion-ot
python -m pip install -r requirements-stage4.txt

# Example only: select an existing numbered checkpoint from the run you want.
CHECKPOINT=outputs/stage1b_d_nce/checkpoints/step_004000.pt

# Stage 2-3: frozen banks and global InfoOT fit, paired EMA weights.
python scripts/run_offline_alignment.py \
  --config configs/stage23_offline/full_sit_b2.yaml \
  --checkpoint "$CHECKPOINT" --weights ema --device cuda:0

# Stage 4: use the completed Stage 2-3 bundle; no refitting.
python scripts/evaluate_full_infoot.py \
  --config configs/stage4_eval/fid_ssim_sit_b2.yaml \
  --bundle outputs/s23_full --device cuda:0
```

Use `--weights raw` on Stage 2–3 if selecting a raw checkpoint evaluation; Stage 4
always inherits the same paired model state. Set Stage 2–3 `alignment_config` to
the configuration matching the checkpoint architecture and Stage 1A provenance.
The final bundle copies learned E/head/G state; the large frozen model files
and configuration dependencies remain external and are checked by content hash.
It requires a format-4 Stage 1B checkpoint and never substitutes Stage 1A outputs
as image-quality references.

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
  --bundle outputs/s23_full --device cuda:0 --resume --generation-batch-size 2
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

- `outputs/s23_full/bundle.json`, `transport.pt`, `banks/`, `models/`,
  `calibration.json`, `solver_progress.pt`, `solver_log.jsonl`, `checks.json`.
- `outputs/s4_eval/metrics.json`, `ssim_per_image.jsonl`,
  `generation_manifest.jsonl`, `gallery_ids.json`, `report.md`.
- `outputs/s4_eval/images/{cat_to_dog,dog_to_cat}/`: complete metric corpus.
- `outputs/s4_eval/real/{cat,dog}/`: original held-out RGB references.
- `outputs/s4_eval/galleries/{cat_to_dog,dog_to_cat}/contact_sheet.png`:
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

Each translation grid contains:

1. source image;
2. full InfoOT conditional mean, the proposed final readout;
3. conditional MAP target, an ablation;
4. conditional sampled target, an ablation;
5. direct structure-teacher mean, a diagnostic that bypasses InfoOT.

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
