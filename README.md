# diffusion-ot

initial test

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
