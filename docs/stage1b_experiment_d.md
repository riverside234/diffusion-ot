# Experiment D: trainable conditioning network and decoded translation losses

This experiment implements the selected extension to Stage 1B: train both
encoders and matching heads, and adapt both generators' added adaLN branches,
token MLPs, semantic projection MLP, final adapter, and existing rank-64
attention LoRA. The original SiT parameters, VAE, DINOv2 feature extractor,
and learned null vectors remain frozen.

The 8,000-step guarded run motivated this experiment, but did not establish
that frozen G caused the plateau. Its analysis remains in
[the previous review](stage1b_guarded_8000_review.md). Experiment D is an
unvalidated AFHQ training candidate, not a measured improvement over that run.
The official-based legacy snapshot and previous configs are unchanged.

## Training path

```text
source image latent -> E_source -> m_source -> full InfoOT conditional weights
target image latents -> E_target -> m_target -----------^
                             |
                             +-> raw target codes

projected code = sum_j conditional_weight_j * raw_target_code_j
projected code + fresh noise -> 50-step target flow -> frozen VAE -> image
image -> frozen DINOv2 -> spatial structure loss + target feature discriminator
```

The transport solve is still detached and uses the official projected-Sinkhorn
mathematics. The projection retains target smoothing and density correction.
All 96 target references participate in each minibatch conditional mean, with
no top-k truncation, sampled-code substitution, or projected-code normalization.
The standalone evaluator uses its complete declared target projection bank.
Minibatch and full-bank projection remain different estimators.

Both domains use the full training split through independently shuffled
loaders. Each update encodes 128 samples per domain: 96 OT references and 32
disjoint queries. Four queries per direction are decoded for image losses;
16 samples per domain enter the same-domain flow loss. Reference/query and
decoded-query selections therefore rotate with the shuffled batches.

### Losses

The existing same-domain PDAE flow reconstruction, latent anchoring, InfoOT
feature loss, neighborhood distillation, and conditional KL remain active.
The new image objective is averaged over the two translation directions:

```text
L_image = ramp(step) * [0.10 L_structure + 0.01 L_adversarial]
ramp(step) = min(step / 2000, 1)
```

`L_structure` is one minus cosine similarity between the decoded source and
translated image's pooled DINOv2 patch self-similarity descriptors. Source
features are detached; generated-image features retain input gradients. The
source is decoded from its cached VAE latent, so the online image loss does
not assume bitwise equality with descriptors cached from original pixels.

Target realism uses two small **feature discriminators**, one per domain.
Each applies LayerNorm and a spectral-normalized two-layer MLP to frozen
DINO CLS and patch tokens. Global and mean patch scores have equal weight.
Discriminators learn a hinge objective on real target images and detached
generated images. The generator uses the negative fake score. This is an
experimental DINO feature GAN, not a reproduction of CycleGAN-Turbo's CLIP
discriminator, and not a guarantee of pixel-level realism. No paired target
image or exact image cycle is assumed.

Each discriminator update finishes before its generator adversarial forward.
The generator forward freezes discriminator parameters and spectral-norm
updates while preserving the gradient into generated features. Validation
never steps the discriminator or changes its buffers. Training discriminator
scores should not be compared across checkpoints as calibrated quality scores.

### Differentiability and CFG

Training uses the same midpoint-time Euler integration rule as evaluation,
including learned-null CFG. The new sampler supports non-reentrant activation
checkpointing of complete SiT predictions. It does not call the inference-only
evaluation sampler. VAE decoding and DINO extraction also support activation
checkpointing. Images pass through torch resize/crop/normalization without
PIL/NumPy conversions or detached intermediate tensors.

Semantic dropout is explicitly applied with probability **0.1** on same-domain
flow training. Image-loss sampling uses the full condition and CFG scale 1.
The learned null vector stays fixed, but shared adapters and LoRA also change
`G(null)`. A separate null-velocity distillation loss, weight 0.10, compares
the current null branch with a fixed Stage 1A null branch on the same noisy
real-image state and time. The teacher uses stateless parameter overrides,
so it does not mutate weights in a live autograd graph or copy the full SiT.

The former 0.25 reconstruction-relative encoder gradient guard is disabled
in the new config. With an adapting G, reconstruction-gradient magnitude is
an unreliable budget for alignment. Instead E, matching heads, and G receive
weighted-sum gradients and are separately clipped at norm 1.0. This changes
the update policy as well as trainability; the previous guarded experiment
remains a control. The optional guarded code path also supports trainable G
and is covered by tests.

## Defaults

| Setting | Value |
| --- | ---: |
| Training updates | 30,000 |
| Encoder LR | 2e-5 |
| Matching-head LR | 2e-4 |
| Adapter / token MLP / z projection LR | 1e-5 |
| LoRA LR | 5e-6 |
| Feature discriminator LR | 1e-4 |
| Fit / projection bandwidth | 0.70 / 0.10 |
| Flow steps, training / fixed validation / quick evaluation | 50 / 50 / 50 |
| Semantic dropout | 0.10 |
| LoRA rank / alpha | 64 / 64 |
| Image-loss ramp | 2,000 updates, starting immediately |
| Gradient norm clips, E / head / G / discriminator | 1 / 1 / 1 / 1 |
| Fixed validation / checkpoint interval | 500 / 1,000 |
| EMA decay / warmup | 0.995 / 500 |

These learning rates and image-loss weights are starting settings. More compute
does not make an incorrect correspondence or information-losing conditional
mean invertible. The model still has a global semantic conditioning code and
fresh target noise, rather than a direct spatial source-conditioning branch.

## Linux workflow

Use the existing Stage 1A rank-64 checkpoints and descriptor cache. The cache
must include `metadata.resolved_revision`, `model`, `processor`, `grid_size`,
and `descriptor: pooled_patch_self_similarity_v1`. The online DINO model is
loaded at that exact recorded revision into `artifacts/pretrained/dinov2`.
If an old cache lacks provenance, regenerate it with the existing cache script
to a new path and set `semantic_prior.path` in the new training config.

Run from `/data/not_backed_up/yxu209/diffusion-ot`:

```bash
# Full experiment D from Stage 1A; 30,000 updates.
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml

# Quick evaluation of a saved D checkpoint, using paired EMA E/head/G.
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_decoder_sit_b2.yaml \
  --checkpoint outputs/stage1b_cat_dog_structure_decoder_infoot_sit_b2_cfg_adaln_all_lora_r64/checkpoints/latest.pt

# Continue the same experiment from its latest complete checkpoint.
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_decoder_sit_b2.yaml \
  --resume
```

To inspect an initial 2,000 updates, add `--max-steps 2000` to the first command,
then use the resume command to continue toward 30,000. `--smoke` uses 500
updates and includes active decoded losses; it is not the full training run.
Adding `--quick-eval` preserves a Stage 1A baseline before training and evaluates
the final checkpoint afterward. A matching baseline is optional for standalone
evaluation. Do not resume an older frozen-G checkpoint as experiment D.

Train output:
`outputs/stage1b_cat_dog_structure_decoder_infoot_sit_b2_cfg_adaln_all_lora_r64`.
Eval output:
`outputs/stage1b_eval_structure_decoder_infoot_cfg_adaln_all_lora_r64`.

## Checkpoints and evaluation

Format 4 adds current generators, generator EMA and EMA counters, fixed Stage
1A generator states/teacher parameter snapshots, both discriminators and their
optimizer, and dedicated per-domain translation-noise RNG states. Encoder and
matching-head state remain paired with the same update's generator state.
No batch-local transport plan is saved. Resume rejects changed image objectives
or trainability. Older formats remain available with their original configs.

Evaluation loads raw E/head/G together or EMA E/head/G together; missing adapted
G or EMA state is an error. The learned null vector is fixed throughout the
run and is included in the saved generator state. Original SiT/VAE/DINO weights
are supplied by their existing pretrained artifacts rather than duplicated in
every checkpoint. Stage 2-4 export bundles must now carry the selected G state
as well as E, matching heads, kernels, raw target codes, and fitted transport.

Fixed validation uses raw weights and fixed train references, held-out queries,
noise, and times. The Stage 1A reconstruction baseline evaluates **E0 with G0**;
it does not accidentally use the changing G with the anchor encoder. Null
reconstruction is reported for both current and fixed Stage 1A generators.
Decoded validation uses four queries per direction and a fixed 50-step sampler;
its IDs, counts, seed, structure scores, and discriminator update count are logged.

Quick evaluation uses eight decoded examples per direction. Its five grid rows
remain source, full InfoOT mean, selected target, sampled target, and direct
structural-teacher mean. `projections.<direction>.decoded_image_diagnostics`
contains mean and per-image structural errors for each generated row. This
uses the training DINO prior; it is not an independent semantic metric.
Continue inspecting actual grids, target realism, source viewpoint/framing/color,
and semantic-CFG null samples. More labeled held-out examples are needed for
strong quantitative claims; neither discriminator loss nor teacher similarity
alone determines checkpoint quality.

## Research basis and verification

- [PDAE, Section 3.2](https://proceedings.neurips.cc/paper_files/paper/2022/file/8aff4ffcf2a9d41692a805b3987e29ea-Paper-Conference.pdf)
  jointly trains its encoder and correction network with a frozen pretrained DPM.
- [ControlNet](https://arxiv.org/html/2302.05543v3) motivates adapting conditioning
  components while preserving a pretrained generative backbone.
- [Splice/SpliceNet](https://arxiv.org/html/2311.12193v1) motivates generated-image
  structure supervision with a frozen ViT; our pooled DINOv2 descriptor differs.
- [CycleGAN-Turbo](https://arxiv.org/html/2403.12036v1) motivates combining trainable
  diffusion adaptation weights with target-domain image supervision. This
  experiment retains multi-step SiT, InfoOT means, and a different discriminator.
- [Official InfoOT](https://github.com/chingyaoc/InfoOT/blob/main/infoot.py) remains
  the basis of the kernel solver and density-ratio conditional readout, not
  the neural adaptation/image objectives introduced here.

CPU tests exercise actual experiment orchestration using tiny SiT/VAE/DINO
fixtures: both-direction image gradients, selective G updates, frozen base/null
weights, sampler value/gradient equivalence with and without checkpointing,
null dropout, fixed teacher stability, discriminator isolation, validation
determinism, resume, raw/EMA evaluator restoration, and decoded grid metrics.
No real AFHQ GPU training or DINO-based translation-quality experiment has been
run locally. The focused InfoOT/training/evaluation suite passes 103 tests;
the preserved legacy suite passes 77. The repository-wide suite passes 142
tests with one pre-existing failure: the latent horizontal-flip fixture
supplies `[1,4,4]` while its loader defaults to `[4,32,32]`. That loader and
fixture are unchanged. Config, all-layer rank-64 LoRA/CFG, sampler/evaluator
consistency, and the training CLI help entry point were also checked.
