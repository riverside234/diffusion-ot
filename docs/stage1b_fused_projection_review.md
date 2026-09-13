# Fused InfoOT co-training: 1,500-update log review

Reviewed on 2026-09-13. Input: the user's Desktop `train.jsonl`, 76 logged
records covering updates 1 through 1,500. No new AFHQ checkpoints, descriptor
bank, decoded grids or `validation.jsonl` from this run are available locally.

## What the log establishes

The following loss entries are the stored 100-update moving averages, not
averages over the sparse 20-update logging interval.

| Diagnostic | Update 100 | Update 1,500 |
| --- | ---: | ---: |
| Total loss | 0.947304 | 0.927609 |
| Cat reconstruction | 0.458646 | 0.442775 |
| Dog reconstruction | 0.487676 | 0.469389 |
| KDE mutual information | 0.265513 | 0.266668 |
| Within-domain structure KL | 0.453184 | 0.403282 |
| Plan effective targets per source (instantaneous) | 1.01931 | 1.02060 |
| Mean maximum row probability (instantaneous) | 0.99805 | 0.99791 |

Learning is slow, rather than absent: reconstruction decreases and structure
KL decreases about 11%. The total also includes a changing alignment warmup,
so comparing its values alone is not a fixed-objective convergence test.

For a feasible 128-by-128 plan with uniform marginals, joint entropy ranges
from `log(128) = 4.85203` (permutation) to `2 log(128) = 9.70406`
(independent). Logged entropy ranges from 4.86711 to 4.88646, close to the
lower bound. The explicit row diagnostics confirm saturation. Decreasing
this entropy further is not the training goal, and making it uniform would
also destroy useful correspondence. Conditional row entropy, normalized by
`log(target_count)`, is a better comparison across transport batch sizes.

After warmup, the weighted MI gradient is only 2.91%, 3.19% and 3.72% of the
reconstruction gradient at updates 1,000, 1,200 and 1,400. Every logged maximum
row residual exceeds `projection_tolerance=1e-5`: range `2.89e-5` to
`4.99e-5`. These are small marginal errors, not evidence of a broken or
non-finite solve, but the old code silently exhausted its 200-iteration cap.

The frozen-prior cost already helps choose matches: mean transported cost
is 0.77056 versus 1.00657 for the independent control across logged batches.
That benefit does not prove query generalization or image faithfulness.

## Official code and related research

[InfoOT's official implementation](https://github.com/chingyaoc/InfoOT/blob/main/infoot.py)
fits fixed feature matrices using fresh entropic solves with cost
`C - lambda * grad(MI)`, independent initialization and RMS-scaled Gaussian
kernels. Its FusedInfoOT defaults are `lambda=100` and `reg=1`, with raw
cross-domain Euclidean costs. Our training-calibrated descriptor costs are
near one, so copying those coefficients would not copy the effective balance.
The shared recurrence remains tested against an independent NumPy/POT version.

[InfoOT Section 5](https://proceedings.mlr.press/v202/chuang23a/chuang23a.pdf)
defines conditional projection for unseen samples and explains bandwidth's
averaging effect. The official examples do not supply a PDAE encoder-training
objective. Our earlier frozen descriptor cost has no direct encoder gradient;
the within-domain KL alone does not teach the cross-domain conditional readout.

[Splice](https://splice-vit.github.io/) motivates a frozen visual structure
prior. Our pooled DINOv2 patch descriptor is an adaptation; it can retain
background information and does not guarantee an anatomical match.
[SwAV Section 3](https://arxiv.org/html/2006.09882v5#S3.SS1)
uses probability prediction as a representation-learning objective, with
detached assignments. Its soft-assignment ablation also cautions against
equating rapid assignment saturation with better features. This motivates a
direct prediction loss here; we do not implement SwAV or claim its results
transfer to unpaired Cat/Dog translation.

## Implemented extension

For each domain, randomly ordered training batches contain 128 encoded
examples: the first 96 fit Gamma, the other 32 are disjoint queries.
Only training samples enter encoder updates. Fixed validation probes use
training references and separate validation queries and never update encoders.

For query x and all target references y_j, define the frozen teacher

`q_j(x) = softmax_j(-C_structure(x, y_j) / 0.05)`.

Let `p_j(x)` be the full normalized InfoOT Eq. (7) density-ratio weights,
including target kernel smoothing, division by target density and target
masses. Train the bidirectional average `KL(q || p)`, keeping Gamma and the
teacher detached. Query features and both reference encoders remain
differentiable. There is no top-k target selection. A log-sum-exp version of
Eq. (7) avoids clipping small probabilities into dead gradients. It is tested
against the existing probability-space helper with unequal, nonuniform masses,
and its derivatives are checked numerically.

This teaches probabilities over structurally similar target codes. It is not
a loss on generated pixels or a claim that the decoder of their weighted mean
has the same structure as the source. The target's raw conditional mean is
still the primary evaluation readout. No independent Cat/Dog code axes are
directly compared, and no generators, null tokens or LoRA adapters are updated.

| Setting | Earlier fused | Projection-trained fused |
| --- | ---: | ---: |
| Inner MI coefficient | 1.0 | 0.10 |
| Inner entropy regularization | 0.05 | 0.05 |
| Outer MI weight | 0.02 | 0.20 |
| Effective coefficient on -MI | 0.02 | 0.02 |
| Neighborhood KL weight | 0.05 | 0.10 |
| Conditional structure KL weight | 0 | 0.05 |
| Encoded samples / fit references / queries | 128 / 128 / 0 | 128 / 96 / 32 |
| Maximum Sinkhorn iterations | 200 | 2,000 |
| Fit / projection bandwidth | 0.70 / 0.10 | 0.70 / 0.10 |
| Encoder LR / update budget | 2e-5 / 5,000 | 2e-5 / 5,000 |

All structural and MI outer weights ramp over 1,000 updates. The inner cost
balance stays fixed. Reducing inner MI feedback addresses the saturated plan;
the outer adjustment preserves the MI multiplier while the new KL adds a
separate mapping signal. The solver reports marginal feasibility and outer
plan change, can stop after three stable feasible iterations (minimum five),
and refuses a final infeasible plan in the new config. An outer stability flag
does not certify a global optimum. Original configs retain the original
iteration policy so previous experiments remain reproducible.

The new config and Linux commands are in [README](../README.md). The descriptor
cache can be reused. Start a new output from Stage 1A; the resume guard rejects
switching an existing fused run to the new objective. The old checkpoints can
still be evaluated under either fixed evaluation protocol as controlled tests.

## Validation and next acceptance check

The focused InfoOT/training/evaluation suite passes 77 tests. The full repository
suite has 116 passing tests and one existing failure in the latent-flip fixture,
which supplies shape `[1,4,4]` while its manifest defaults to `[4,32,32]`.
The dataset implementation and that fixture are unchanged by this revision.

CPU tests cover official solver parity, Eq. (7) values and gradients, sparse
plans at small bandwidths, both-domain encoder gradients with detached
teachers/plans, a synthetic query-learning problem, saturation sensitivity,
solver convergence/failure reporting, train/validation isolation, frozen
generators, resume and evaluator integration. Synthetic improvements establish
that the new signal can learn; they do not establish improvement on AFHQ.

At updates 500, 1,000 and 1,500 compare the fixed projection-probe KL and
expected structure cost to their step-0 values, while checking reconstruction
drift, projected-code norm ratio and effective targets. The probe uses 96
target references; the final quick evaluator uses the full target training
bank. Its `structure_prior_diagnostics` explicitly measures the training prior,
so independent viewpoint/framing/color labels and decoded Cat-to-Dog and
Dog-to-Cat grids remain necessary to assess translation. No exact optimum or
real-image quality gain is claimed before that experiment.
