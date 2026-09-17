# diffusion-ot

Cat/Dog PDAE representations with InfoOT conditional projection and diffusion
translation. The active Stage 1B experiment is **Experiment D**. It jointly
adapts the semantic encoders, InfoOT matching heads, and the added conditioning
components of each domain generator, then trains decoded conditional means with
source-structure and target-realism losses.

## Active workflow

The current pipeline has four stages:

1. Train separate Cat and Dog PDAE branches with semantic CFG and rank-64 LoRA.
2. Run Experiment D co-training with full InfoOT conditional means.
3. Evaluate held-out queries with the complete target projection bank.
4. Export the selected encoders, matching heads, adapted generators, kernels,
   target codes, and fitted transport for downstream inference.

The canonical configurations are:

- Stage 1A Cat: `configs/stage1a_pdae/cat_sit_b2_lora.yaml`
- Stage 1A Dog: `configs/stage1a_pdae/dog_sit_b2_lora.yaml`
- Experiment D: `configs/stage1b_infoot/structure_decoder_sit_b2.yaml`
- Experiment D evaluation: `configs/stage1b_eval/structure_decoder_sit_b2.yaml`

Experiment D starts from the Stage 1A EMA checkpoints. It trains both encoders,
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
Experiment D also requires the frozen DINOv2 structural descriptor bank:

```bash
python3 scripts/cache_infoot_structure.py --device cuda:0
```

The cache must match the current data split and unflipped preprocessing. The
trainer validates its model revision, descriptor definition, sample IDs, and
content fingerprint.

## Stage 1A

Train Cat and Dog on separate GPUs:

```bash
python3 scripts/train_pdae_domain.py \
  --config configs/stage1a_pdae/cat_sit_b2_lora.yaml \
  --device cuda:0

python3 scripts/train_pdae_domain.py \
  --config configs/stage1a_pdae/dog_sit_b2_lora.yaml \
  --device cuda:1
```

Both configs use all 12 SiT-B/2 blocks for AdaLN-Zero injection and attention
LoRA, LoRA rank/alpha 64/64, semantic dropout 0.10, and 50,000 updates. Stage 1B
requires checkpoints with learned null tokens and cannot load the older non-CFG
format.

## Experiment D co-training

The active config now differentiates the feature-dependent InfoOT RMS kernel
scale (`matching.distance_scale_gradient: full`). The step-2,600 review found
a reproducible spurious shrinkage gradient when that scale was detached.
The full validation also exposed an unfinished outer OT solve. Active D now
requires outer convergence, with a 1,200-iteration cap and early stopping.
The cap was raised after a batch exhausted 300 updates while its plan change
was still `2.01e-5` versus the `1e-5` tolerance. A budget-only increase can
resume an existing run that already required outer convergence; objective and
tolerance changes still require a fresh run. See the
[convergence fix and resume command](docs/analysis/stage1b_experiment_d_outer_budget_fix.md).
The RMS-only follow-up still loses about 90% of matching-head spread by step
1,500, while raw encoder spread stays nearly unchanged. The active revision
protects matching-feature variance and adds a modest covariance penalty from
update 1, plus distance-normalized neighborhood teaching to remove a remaining
cosine-logit contraction incentive. Start from Stage 1A in the new
`outputs/stage1b_d_fit050` output directory. The protected cosine control is
`structure_decoder_vicreg_cosine_sit_b2.yaml`; its `_rmsgrad_vicreg` outputs remain
separate. `structure_decoder_rmsgrad_sit_b2.yaml` preserves RMS only, and
`structure_decoder_detached_sit_b2.yaml` preserves the original Experiment D.
Each has a corresponding evaluation YAML. See the
[new analysis, paper/GitHub audit, and reproduction](docs/analysis/stage1b_rmsgrad_1500/review.md).
Changing neighborhood geometry or temperature requires a fresh run. CPU checks
verify the targeted mechanism; improved AFHQ training remains to be tested.
The active fit bandwidth is now **0.50**, with projection still **0.10**.
`structure_decoder_relational_fit070_sit_b2.yaml` preserves the otherwise
identical 0.70 relational control in both train/eval folders. Start the 0.50
run from Stage 1A; changing fit bandwidth is incompatible with resume.

Review an initial 1,500-update run, with particular attention to spread at
200–500 steps. If this comparison succeeds, continue to 5,000 updates to check
behavior beyond the 2,000-update decoded-loss ramp and review image quality.

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --max-steps 1500 --quick-eval
```

Evaluate that checkpoint with paired EMA encoder, matching-head, and generator
weights:

```bash
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_decoder_sit_b2.yaml \
  --checkpoint outputs/stage1b_d_fit050/checkpoints/latest.pt \
  --no-require-stage1a-baseline
```

Continue the accepted run to its configured 30,000 updates:

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --resume
```

Omit `--max-steps` for a fresh full run. `--smoke` runs only 500 updates and is
an implementation check. Do not resume a frozen-generator or matching-head-only
checkpoint into Experiment D; checkpoint validation rejects incompatible
objectives and trainability.

The active defaults are:

| Setting | Value |
| --- | ---: |
| Updates | 30,000 |
| Encoder / matching-head LR | `2e-5` / `2e-4` |
| Adapter / LoRA LR | `1e-5` / `5e-6` |
| InfoOT fit / projection bandwidth | `0.50` / `0.10` |
| Entropy regularization | `0.05` |
| Outer OT update budget / tolerance | `1200` / `1e-5`, convergence required |
| Matching variance / covariance weights | `0.02` / `0.001`, from update 1 |
| Matching scaled standard-deviation floor | `0.70`, on `sqrt(dim) * m(z)` |
| Same-domain semantic dropout | `0.10` |
| Decoded sampler steps | `20` |
| Decoded structure / adversarial weight | `0.10` / `0.01` |
| Decoded-loss ramp | 2,000 updates |
| Validation / checkpoint interval | 500 / 1,000 updates |

Matching protection uses the 96 normalized OT references independently per
domain. Covariance is the mean squared off-diagonal population covariance
of the scaled features. The floor is a soft penalty, and these initial weights
still need AFHQ evaluation. Raw decoder codes and projected means are not its
targets. Training and fixed validation log weighted/unweighted terms, spread,
and the fraction of coordinates below the floor; training also measures the
regularizer's encoder/head gradients. Changing these settings requires a fresh run.

These are experiment settings, not demonstrated optima. The training batch uses
the full shuffled training split over time. Each update draws 128 samples per
domain: 96 references for the InfoOT fit and 32 disjoint conditional queries.

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

Run in the existing training environment (Python >=3.11). The configured
Stage 1A checkpoints, pretrained SiT/VAE snapshot, DINO structure cache,
canonical train/validation manifests, cached latents, and original AFHQ dataset
must be available under the project paths. Inception weights download on the
first Stage 4 metric run and are cached in `outputs/fid_cache`.

```bash
cd /data/not_backed_up/yxu209/diffusion-ot
python -m pip install -r requirements-stage4.txt

# Example only: select an existing numbered checkpoint from the run you want.
CHECKPOINT=outputs/stage1b_d_gt/checkpoints/step_004000.pt

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
