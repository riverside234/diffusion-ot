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
requires outer convergence, with a 300-iteration budget and early stopping.
The active revision also protects matching-feature variance and adds a modest
covariance penalty from update 1. Start it from Stage 1A in the new
`_rmsgrad_vicreg` output directory. `structure_decoder_rmsgrad_sit_b2.yaml`
preserves the corrected-RMS-only comparison; `structure_decoder_detached_sit_b2.yaml`
preserves the original Experiment D settings and checkpoint compatibility.
See the [log analysis](docs/analysis/stage1b_experiment_d_2600/review.md) and
[protection definition and comparison protocol](docs/analysis/stage1b_experiment_d_vicreg.md).

Run an initial 5,000-update checkpoint review. This reaches well beyond the
2,000-update decoded-loss ramp while avoiding the cost of the full run before
the first image-quality check.

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --max-steps 5000
```

Evaluate that checkpoint with paired EMA encoder, matching-head, and generator
weights:

```bash
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_decoder_sit_b2.yaml \
  --checkpoint outputs/stage1b_cat_dog_structure_decoder_infoot_sit_b2_cfg_adaln_all_lora_r64_steps20_rmsgrad_vicreg/checkpoints/latest.pt \
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
| InfoOT fit / projection bandwidth | `0.70` / `0.10` |
| Entropy regularization | `0.05` |
| Outer OT update budget / tolerance | `300` / `1e-5`, convergence required |
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

## Evaluation output

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
