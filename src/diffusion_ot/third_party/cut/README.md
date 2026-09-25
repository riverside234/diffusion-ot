# CUT PatchNCE source

Upstream: https://github.com/taesungp/contrastive-unpaired-translation

Pinned commit: `b3ac297708dfb6f7589d04662277e53c0d579c27` (retrieved 2026-09-24).

- `patchnce.py` is the **unmodified** upstream `models/patchnce.py`.
- `LICENSE` is the **unmodified** upstream license (BSD 2-Clause).
- `sampling_reference.txt` contains verbatim `Normalize` and `PatchSampleF`
  excerpts from upstream `models/networks.py` at the same revision. It is a
  reference, not imported training code.

Source links:

- https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/patchnce.py
- https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/networks.py
- https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/LICENSE

Downloaded SHA256 checksums:

```
patchnce.py  2bd50b34679786219a4d5c5858a3a277b0fd4467cc49b05d37a2041cb5f7f678
LICENSE     4136d26774a0a0fdcf3f4c0e1140f19279565d41df764c64cc75c0283e6f5249
```

The project wrapper in `diffusion_ot.losses.patchnce` calls this exact loss with
within-image negatives and the actual runtime batch size. Its no-MLP sampling
matches `PatchSampleF(use_mlp=False)`: identical sampled spatial IDs for paired
maps and all images, batch-major flattening, and division by `(L2 norm + 1e-7)`.
It replaces NumPy's global permutation with a caller-owned Torch generator,
computes in FP32, transfers detached keys to the query device, and uses the
vector-norm operator to give a finite derivative at an exactly zero patch.
These are reproducibility/numerical adaptations; there are no learned projectors.

Copyright (c) 2020, Taesung Park and Jun-Yan Zhu. All rights reserved.
