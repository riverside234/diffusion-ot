# diffusion-ot

Cat/Dog PDAE representations with InfoOT conditional projection.

## Current experiment: learned InfoOT matching heads

The guarded 8,000-update run still has weak structural matching. Validation
KL improves about 5.1%, but expected structure cost improves only 0.57%/0.12%
Cat-to-Dog/Dog-to-Cat. Reconstruction drifts 2.13%/2.69%, the support divergence
increases, and Cat-to-Dog projected variance decreases. All logged InfoOT
solves converge. See the [8,000-update review, plots, and research rationale](docs/stage1b_guarded_8000_review.md).

The new opt-in `structure_metric_sit_b2.yaml` trains residual matching heads
alongside the encoders. Heads start at normalized identity and learn InfoOT
geometry outside the encoder's gradient cap. Decoder inputs remain the raw
semantic codes. Encoder/head learning rates are `2e-5`/`2e-4`; the budget is
8,000 updates. The previous support loss is disabled: a conditional mean
generally has less variance than the target distribution, and the prior
experiment did not demonstrate benefit from forcing those distributions
together. The original guarded config and official-based legacy snapshot
remain available as controls.

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_metric_sit_b2.yaml

python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_metric_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_metric_sit_b2.yaml \
  --checkpoint outputs/stage1b_cat_dog_structure_metric_infoot_sit_b2_cfg_adaln_all_lora_r64/checkpoints/latest.pt
```

Start from Stage 1A in the new output directory, then use `--resume` only for
that experiment. To inspect an initial 2,000-update run, add `--max-steps 2000`.
Evaluation preserves head/encoder raw-or-EMA pairing and adds a fifth grid row
for the direct structural-teacher mean when descriptors are available. New
variance diagnostics separate variability of means from conditional
uncertainty. Decoder-code nearest-neighbor diagnostics remain in raw-code
coordinates. Independent proxy labels and generated grids must still establish
translation quality; the new architecture is an experiment, not a proven fix.

## Earlier guarded conditional projection experiment

The preceding, unguarded logs stop at update 1,000, exactly the end of alignment warmup.
Every logged InfoOT solve converges. Fixed-validation conditional KL falls
6.3%, but expected structure cost improves only about 0.3-0.5%, while Cat/Dog
reconstruction losses increase 2.0%/2.8%. This is measurable learning with
limited correspondence gains; it is too early to establish a post-warmup
plateau. The conditional-loss gradient reaches 2.15 times the reconstruction
gradient. See [the new log review](docs/stage1b_projection_1000_review.md) for
measurements, research sources, and limits of this diagnosis. The
[earlier 1,500-update review](docs/stage1b_fused_projection_review.md) describes
a different run before the solver revision.

`configs/stage1b_infoot/structure_fused_projection_sit_b2.yaml` introduced:

- A per-encoder gradient guard: reconstruction plus latent anchoring is the
  primary objective. Conflicting auxiliary components are projected away,
  then their combined norm is capped at 25% of the primary norm. This is an
  asymmetric adaptation inspired by PCGrad, not its complete algorithm or a
  guarantee that Adam steps preserve held-out reconstruction.
- A debiased Sinkhorn divergence between full conditional means and frozen
  Stage 1A target codes. Target masses follow the frozen structural teacher's
  mixture for the current queries, so a small batch need not cover every
  target mode. Gradients teach the Eq. (7) weights; target values and anchors
  are detached for this term. Output codes are still unmodified conditional
  means, with target smoothing, density correction, and no top-k truncation.
- Fixed-probe and full-evaluator variance diagnostics alongside norms. A
  larger norm alone cannot establish recovery from mean contraction.

Conditional-structure KL weight changes from 0.05 to 0.02 and the new
projection-support weight is 0.02. LR stays `2e-5`, fit/projection bandwidths
stay 0.70/0.10, inner MI weight stays 0.10, and entropy regularization stays
0.05. The original budget was 5,000 updates and was extended to 8,000 for the
reviewed guarded run, with 1,000-update warmup. Each
domain still supplies 96 OT references and 32 disjoint training queries.
These are experimental settings; the new losses are project extensions to
official InfoOT and need validation on generated images.

Reuse `data/semantic_priors/afhq_dinov2_structure.pt`. Start from Stage 1A in
the new `...projection_guarded...` output directory. Changed objectives are
rejected on resume; existing checkpoints and old output directories are kept.
Both generators, attention LoRA, and learned null tokens remain frozen.

```bash
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_fused_projection_sit_b2.yaml

python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_fused_projection_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_fused_projection_sit_b2.yaml \
  --checkpoint outputs/stage1b_cat_dog_structure_fused_projection_guarded_infoot_sit_b2_cfg_adaln_all_lora_r64/checkpoints/latest.pt
```

The current guarded-control config runs 8,000 real updates. To inspect an earlier checkpoint,
use `--max-steps 1500`, then use the same config with `--resume` to continue.
`logs/validation.jsonl` records a fixed train-reference/validation-query
projection probe at step 0 and every 500 updates, alongside the fixed
reconstruction probe. Monitor validation KL, expected structure cost,
reconstruction drift, `projection_support_loss`, and
`projected_to_target_variance_ratio`. In training logs,
`gradient_guard.cat.auxiliary_ratio_after` and the Dog equivalent describe
the applied auxiliary contributions. The older weighted-gradient diagnostics
measure gradients before this guard. With this update rule, the logged total
loss is descriptive; it is not a scalar objective whose gradient Adam uses
unchanged. Warmup and random training batches also prevent a monotonic-loss
expectation.

The probe uses 96 target references; the quick evaluator projects over the
full target training bank and supplies the required decoded-image check.
Teacher agreement is not independent semantic evidence. Use viewpoint,
framing, and color precision plus both translation directions before accepting
a checkpoint. A matching Stage 1A report is optional for standalone evaluation.

## Preserved official-based version

The prior InfoOT core, training/evaluation modules, CLIs, configs, tests, and
documentation are preserved byte for byte in
[`legacy/infoot_official_v1`](legacy/infoot_official_v1/README.md), with SHA-256
verification and a launcher that isolates their imports from the active code.
The snapshot includes the previous project extensions and shares unchanged
Stage 1A model/data dependencies with the main tree.

```bash
python3 scripts/run_legacy_infoot.py check
python3 scripts/run_legacy_infoot.py train \
  --config configs/stage1b_infoot/structure_fused_projection_sit_b2.yaml
```

The legacy command resolves the preserved config and its original output
directory. Use `--resume` to continue that old experiment, or a copied legacy
config with a different output directory for a separate reproduction. Plain
official-based InfoOT and the first fused configs are also preserved. See the
legacy README for evaluation and test commands.

## Earlier structure-guided co-training experiment

The 5,000-update plain-InfoOT run produced realistic targets with weak source
faithfulness. Every logged 64-by-64 plan had entropy `log(64)`, while MI stayed
near 1.789. This indicates almost permutation-like transport, not successful
semantic correspondence. Raw training loss also mixes changing minibatches,
flow times/noise, and alignment warmup, so it need not decrease monotonically.

`structure_fused_sit_b2.yaml` is the earlier Stage 1B-2 control. It uses
the official Fused InfoOT update `C - lambda * grad(MI)` as a fresh Sinkhorn
cost, where `C` compares frozen DINOv2 patch self-similarity descriptors in a
shared space. A separate within-domain neighborhood-distillation loss teaches
each PDAE encoder to retain the descriptor's structural relationships. No
Cat/Dog raw-code coordinate distances are used. The target decoder still
receives the full Eq. (7) conditional mean, including target smoothing and
target-density correction.

This follows [official InfoOT](https://github.com/chingyaoc/InfoOT/blob/main/infoot.py)
for the transport update. The structural prior is a project adaptation motivated
by [Splice](https://arxiv.org/abs/2311.12193) and
[DINO visual descriptors](https://arxiv.org/abs/2112.05814).
[EGSDE](https://arxiv.org/abs/2207.06635) motivates evaluating source faithfulness
separately from target realism; its image-level energy guidance is not implemented
by this co-training change. The new method requires empirical validation on AFHQ.

Run from the Linux project root, starting fresh from the configured Stage 1A
EMA checkpoints:

```bash
# 1. Cache the shared structural descriptors once (downloads DINOv2-small).
python3 scripts/cache_infoot_structure.py --device cuda:0

# 2. Train the new 5,000-update pilot in a separate output directory.
python3 scripts/train_joint_infoot.py \
  --config configs/stage1b_infoot/structure_fused_sit_b2.yaml

# 3. Evaluate the full conditional mean; a Stage 1A report is optional here.
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/structure_fused_sit_b2.yaml \
  --eval-config configs/stage1b_eval/structure_fused_sit_b2.yaml \
  --checkpoint outputs/stage1b_cat_dog_structure_fused_infoot_sit_b2_cfg_adaln_all_lora_r64/checkpoints/latest.pt
```

The new pilot uses transport/reconstruction batches 128/16, `h_fit=0.70`,
Sinkhorn regularization 0.05, `h_proj=0.10`, and encoder LR `2e-5`. The
reconstruction/MI/anchor weights remain 1/0.02/0.01, with a new neighborhood
weight of 0.05. These are pilot settings, not demonstrated optima. EMA decay
0.995 gives about 200 updates of history, versus roughly 10,000 at 0.9999.
Both domain generators, attention LoRA, and learned null tokens stay frozen.
The static structure cache requires unflipped training inputs, enforced in code.

Training logs now contain `window_mean`, row/column transport effective counts,
prior transport cost versus independent cost, and weighted gradient diagnostics.
`logs/validation.jsonl` compares raw-encoder reconstruction against the fixed
Stage 1A encoder on the same held-out images, noise, and times, without changing
the training RNG. Numbered checkpoints are retained every 500 updates. Use
`--max-steps 1500` for an initial check after the 1,000-step alignment warmup;
resume that same experiment with `--resume` to reach 5,000.

Descriptor banks include sample IDs, split/domain membership, model revision,
preprocessing, a training-only cost calibration, and a content fingerprint.
Training refuses missing/mismatched descriptors or a changed prior on resume.
Evaluation includes the prior fingerprint in its protocol hash. A new descriptor
bank requires a new output file. Descriptor agreement is training-prior evidence;
independent proxy labels and decoded grids still determine correspondence quality.
For a matched Stage 1A + Fused InfoOT control, repeat command 3 without
`--checkpoint`; existing plain-InfoOT reports remain separate controls.

## InfoOT evaluation bandwidths

In `configs/stage1b_eval/quick_sit_b2.yaml`,
`matching.bandwidth_multiplier` controls fitting the transport plan.
`matching.projection_bandwidth_multiplier` controls conditional retrieval and
Eq. (7) projection, including decoded means and UMAP. The Stage 1A cat/dog
sweep selected `0.10`; override it from the CLI for sensitivity checks:

```bash
python3 scripts/evaluate_infoot_alignment.py \
  --alignment-config configs/stage1b_infoot/plain_sit_b2.yaml \
  --eval-config configs/stage1b_eval/quick_sit_b2.yaml \
  --projection-bandwidth 0.40
```

For Stage 1B evaluation, add `--checkpoint <joint-checkpoint.pt>` and use the
same projection bandwidth as its Stage 1A baseline. Reports and output
protocol IDs record the effective bandwidth. Each full CLI invocation still
fits a transport plan; this option changes its conditional readout, not the
fitting settings, and does not add cross-run coupling reuse.

## UMAP proxy labels

The evaluation UMAP uses color for a proxy attribute and marker shape for point
type. When an entire panel lacks labels, it falls back to distinct point-type
colors and prints a coverage warning. `viewpoint` describes the camera-relative
direction of the subject (for example, front, side, three-quarter, or back).
`framing` describes the crop or shot scale (for example, face close-up,
head-and-torso, or full body). A sample can therefore be both `side` viewpoint
and `close` framing.

`Target bank` means the real target-domain latent codes used by the projection;
for dog-to-cat evaluation these are cat codes. `Projected source` means dog query
codes mapped into the cat code space. `Unlabeled` means that the requested
attribute was absent from both dataset metadata and the JSONL file configured at
`proxy_labels.path`; it is not a learned class or an InfoOT result.

`proxy_labels.path` is resolved relative to `project_root`. With the Linux root
in the configs, this setting:

```yaml
proxy_labels:
  path: data/proxy_labels/afhq_viewpoint_framing.jsonl
  attributes: [viewpoint, framing, coat_color]
  sample_id_key: sample_id
```

loads
`/data/not_backed_up/yxu209/diffusion-ot/data/proxy_labels/afhq_viewpoint_framing.jsonl`.
The file uses one JSON object per line, with no duplicate sample IDs:

```json
{"sample_id":"afhq_cat_<hf_index>","viewpoint":"<viewpoint>","framing":"<framing>","coat_color":"<coat_color>"}
{"sample_id":"afhq_dog_<hf_index>","viewpoint":"<viewpoint>","framing":"<framing>","coat_color":"<coat_color>"}
```

Use the same vocabulary for both domains. Recommended values are `front`,
`three_quarter`, `side`, and `back` for viewpoint, and `close_up`, `medium`, and
`full_body` for framing. Use `black`, `white`, `gray`, `brown`, `orange`, and
`mixed` for coat color. Omit an attribute or set it to `null` when it has not
been labeled; do not guess a label merely to increase coverage. Coat color is
an evaluation readout: it does not add color supervision to InfoOT training.
