# Review of the 1,000-update conditional co-training run

Reviewed 2026-09-13. This is the new run supplied as Desktop `train.jsonl`
and `validation.jsonl`, after the previous solver/conditional-loss changes.
It is distinct from [the earlier 1,500-update fused run](stage1b_fused_projection_review.md).

## What the logs establish

There are 51 training records through update 1,000 and five validation rows.
Three step-0 validation rows are identical; treat them as one baseline, not
independent repetitions. Steps 0, 500, and 1,000 use the same train references
and held-out validation queries, with raw encoder weights.

| Fixed validation diagnostic | Step 0 | Step 500 | Step 1,000 |
| --- | ---: | ---: | ---: |
| Cat reconstruction loss | 0.672565 | 0.684171 | 0.685853 |
| Dog reconstruction loss | 0.609793 | 0.623742 | 0.626907 |
| Bidirectional conditional KL | 3.072929 | 3.017688 | 2.880258 |
| Cat-to-Dog expected structure cost | 0.941100 | 0.939537 | 0.936419 |
| Dog-to-Cat expected structure cost | 0.991073 | 0.990572 | 0.987828 |
| Cat-to-Dog effective targets | 29.04 | 30.00 | 30.93 |
| Dog-to-Cat effective targets | 24.70 | 25.63 | 25.26 |
| Cat-to-Dog projected/target norm | 0.360 | 0.393 | 0.420 |
| Dog-to-Cat projected/target norm | 0.423 | 0.462 | 0.518 |

Every logged training and validation InfoOT solve satisfies the configured
marginal tolerance and reports outer convergence. Most finish in 11-12
outer iterations. The 96-by-96 fit plan averages roughly 26 effective
targets per source, versus about 1.02 in the earlier failed run. Its fairly
stable entropy is therefore not evidence that this solver is stuck.
Decreasing plan entropy is not a training objective in itself.

Validation KL decreases 6.27%, while structure cost improves about 0.50%
Cat-to-Dog and 0.33% Dog-to-Cat. Cat/Dog reconstruction increases 1.98%/2.81%.
The teacher costs remain 0.6958/0.8170, so the student is still far from the
structural teacher. These are small fixed probes (32 queries/domain and eight
reconstruction samples/domain), not population estimates of image quality.

The weighted conditional-KL gradient/reconstruction-gradient ratios at steps
200, 400, 600, 800, and 1,000 are 0.39, 0.76, 1.40, 2.15, and 1.85. The MI
ratio at step 1,000 is only 0.0058. This supports testing a limit on auxiliary
gradients; it does not by itself prove that conditional KL caused the drift.
No gradient angles were logged in this run, so conflict remains a hypothesis
that the new diagnostics will measure.

The run ends exactly at the 1,000-update alignment warmup. Its 100-update
average total rises from 0.9553 to 1.0698 while the loss weights are increasing.
Over those windows, reconstruction decreases from 0.4586 to 0.4472 for Cat
and 0.4877 to 0.4716 for Dog; conditional KL decreases from 2.6191 to 2.4515.
Neither a constant-weight plateau nor generalization from these minibatch
losses can be inferred. The norm ratios suggest contraction, but centering
and diversity cannot be reconstructed from scalar norms; the revised code
logs variance as well.

Input SHA-256 values for traceability:

```text
train.jsonl       cb3c6da6b970e33810c08ecdcccf0ac2db2ac402777d53c2db7c256837a99c9d
validation.jsonl  6a5fdcfc4003a3816537aba26487a78960955e262581e1b5cf62569f076d9c74
```

## Research and implementation choices

The [official InfoOT implementation](https://github.com/chingyaoc/InfoOT/blob/main/infoot.py)
optimizes a plan on fixed feature sets, using a fresh entropic OT solve with
cost `C - lambda * grad(MI)` for Fused InfoOT. It does not define a neural
encoder co-training objective or guarantee image translation. We retain that
update and the complete Eq. (7) readout. The following additions are project
extensions rather than claims of official-repository parity.

### Per-encoder gradient guard

[Yu et al., Gradient Surgery for Multi-Task Learning (PCGrad, 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
identify negative gradient dot products as conflicting task updates and
project away conflicting components. Here reconstruction plus anchoring is
privileged over alignment. For each encoder, let `g_p` be that primary
gradient and `g_a` the combined, warmed-up auxiliary gradient:

```text
if <g_a, g_p> < 0:
    g_a <- g_a - <g_a,g_p> / ||g_p||^2 * g_p
g_a <- g_a * min(1, 0.25 * ||g_p|| / ||g_a||)
gradient <- g_p + g_a
```

Zero primary gradient suppresses the auxiliary update. The implementation
checks finite gradients and logs each encoder's angle, conflict decision,
scaling factor, and norms before/after adjustment. This is asymmetric
projection with a norm cap, not PCGrad's randomized symmetric procedure.
The first-order statement applies to raw gradients; Adam's preconditioning,
momentum, and finite steps mean it cannot guarantee held-out reconstruction.
The weighted total is now a monitoring scalar, not an unchanged gradient
descent objective. Global gradient clipping still applies afterwards.

### Distribution loss on full projected means

[Feydy et al., Interpolating between Optimal Transport and MMD using Sinkhorn Divergences (2019)](https://proceedings.mlr.press/v89/feydy19a.html)
define the debiased divergence

```text
OT_eps(a,b) = min_P <P,C> + eps * KL(P || a outer b)
S_eps(x,y) = OT_eps(x,y) - 0.5*OT_eps(x,x) - 0.5*OT_eps(y,y)
```

The self terms remove entropic self-attraction. Our new shared implementation
includes all three regularized values and differentiates costs using detached,
converged optimal plans. Both source arguments in the self-cost receive
gradients. Uniform and nonuniform-mass values are checked independently
against POT, and cost gradients against finite differences. Plans outside
tolerance are rejected rather than used for envelope gradients.

For Cat-to-Dog, the source distribution consists of each query's full Eq. (7)
mean in raw Dog-code coordinates. The target atoms are fixed Stage 1A Dog
codes for the same target-reference IDs. The reverse direction is analogous.
Costs are squared distances divided by dimension times the fixed Stage 1A
calibration variance. No raw Cat/Dog coordinates are directly compared.

Target masses are the average frozen structural-teacher probability across
the current queries. This asks projected codes to cover the target mixture
relevant to those queries, rather than every target mode in a small batch.
[Nguyen et al., Improving Mini-batch Optimal Transport via Partial Transportation (2022)](https://proceedings.mlr.press/v162/nguyen22e.html)
show why minibatch OT can suffer misspecified matches; their partial-OT
algorithm is not implemented here. Teacher-conditioned masses are our
heuristic and do not remove all minibatch bias or teacher errors.

The new loss detaches the current raw target value bank and Stage 1A anchors;
gradients pass through the full conditional weights into both KDE encoders.
This removes the direct path that could reduce this loss by simply inflating
target prototypes. Existing reconstruction and anchoring control changes to
the value bank. There is no output renormalization, nearest-target replacement,
or top-k truncation. This loss tests code-distribution compatibility, not
pixel-level preservation; its benefit on generated images is unproven.

## Settings and workflow

| Setting | Supplied run | New experiment |
| --- | ---: | ---: |
| Conditional-structure KL weight | 0.05 | 0.02 |
| Projection-support divergence weight | 0 | 0.02 |
| Auxiliary/primary gradient cap, per encoder | none | 0.25 |
| Projection-support regularization | none | 0.10 |
| Encoder learning rate | 2e-5 | 2e-5 |
| Fit / projection bandwidths | 0.70 / 0.10 | 0.70 / 0.10 |
| Inner MI / outer alignment weights | 0.10 / 0.20 | 0.10 / 0.20 |
| InfoOT entropy regularization | 0.05 | 0.05 |
| Budget / alignment warmup | 5,000 / 1,000 | 5,000 / 1,000 |

The active train/eval configs retain the name `structure_fused_projection_sit_b2.yaml`
and now write separate `...projection_guarded...` outputs. Start them fresh
from Stage 1A; use `--resume` only for continuation of that same experiment.
Reuse the existing frozen DINO descriptor cache. Linux commands are in
[README](../README.md#current-conditional-projection-co-training-experiment).
An existing output checkpoint prevents accidental fresh-run overwrite.
Fresh runs reset both log streams together, avoiding repeated baseline rows
left by incomplete initialization.

Both generators, all LoRA/adapters, and learned null tokens stay frozen.
Encoder EMA, checkpoint provenance, split isolation, detached batch-local
plans, and the full conditional readout remain in use. Training-probe
diagnostics use raw encoders; evaluate the saved EMA checkpoint with the
full-bank quick evaluator before judging decoded image quality.

Inspect 500, 1,000, 1,500, and later checkpoints using the same fixed protocol:

- Solver feasibility and outer convergence, then auxiliary ratios **after**
  the guard and conflict frequency. A permanently saturated cap is evidence
  that alignment is being constrained, not necessarily that it is improving.
- Fixed reconstruction drift, expected structural cost and KL relative to
  step 0. Inspect individual examples because the probe is small.
- Support divergence, centered variance ratio, mean shift, effective target
  count, and nearest-target distance. Do not select a model on norm recovery
  or declining entropy alone.
- Full-bank Cat-to-Dog and Dog-to-Cat decoded conditional means, source
  faithfulness, target realism, and independent proxy precision with nonzero
  label coverage. Teacher agreement is not independent semantic validation.

For an ablation, copy the new config to a new output directory and disable
`projection_support.enabled` with its loss weight set to zero. This isolates
the gradient guard/weight change from the distribution term. The preserved
workflow also permits continuing the supplied old objective beyond warmup;
its experiment should remain separate from the revised one.

## Preserved version and checks

[`legacy/infoot_official_v1`](../legacy/infoot_official_v1/README.md) preserves
27 source/config/test/documentation files byte for byte from the pre-revision
working tree at commit `9274a66b082ef0d46cab563429cbe95e7b269a4e`. Its launcher
verifies hashes and that protected modules resolve inside the snapshot.
Newline conversion is disabled for snapshot files so verification works
after Windows-to-Linux checkout. The snapshot includes prior project
extensions, not a vendored copy of unmodified upstream InfoOT. Shared Stage
1A model/data dependencies remain in the main repository.

Local CPU validation covers independent Sinkhorn-divergence values and
gradients, contraction sensitivity, frozen anchors/value banks, per-encoder
gradient bounds, both encoders updating with frozen generators, resume
compatibility, train/validation isolation, evaluator integration, and legacy
runtime isolation. The active focused suite has 87 passing tests; the
preserved suite has 77 passing tests. The full repository suite has 126
passing tests and the existing latent-flip fixture failure: it supplies
`[1,4,4]` while the loader expects the default `[4,32,32]`. That fixture and
the data loader are unchanged. No real AFHQ GPU training or decoded
image-quality comparison was run on this machine.
