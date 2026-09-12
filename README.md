# diffusion-ot

initial test

## InfoOT evaluation bandwidths

In `configs/stage1b_eval/quick_sit_b2.yaml`,
`matching.bandwidth_multiplier` controls fitting the transport plan.
`matching.projection_bandwidth_multiplier` controls conditional retrieval and
Eq. (7) projection, including decoded means and UMAP. Set it to `null` to
inherit the fitting bandwidth, or override it from the CLI:

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
