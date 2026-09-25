"""Spatial source correspondence using CUT's official within-image PatchNCE.

The loss itself is vendored unmodified. Sampling adapts official PatchSampleF
with optional learned MLPs to private Torch RNG and FP32; see third_party/cut/README.md.
PatchNCE is still an InfoNCE objective, now over locations within each image.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace

import torch

from diffusion_ot.third_party.cut.patchnce import PatchNCELoss


CUT_COMMIT = "b3ac297708dfb6f7589d04662277e53c0d579c27"
PATCHNCE_PROTOCOL = "cut_within_image_no_mlp_private_rng_v1"
PATCHNCE_MLP_PROTOCOL = "cut_within_image_domain_mlp_private_rng_v1"


def _normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    # CUT Normalize uses division by (norm + epsilon), not clamp(norm, epsilon).
    # vector_norm has the same value and gives a finite gradient at exact zero.
    return patches / (torch.linalg.vector_norm(patches, dim=-1, keepdim=True) + 1e-7)


@torch.no_grad()
def _layer_metrics(query, key, query_raw, key_raw, per_patch, patch_ids, shape):
    batch, channels, height, width = shape
    count = len(patch_ids)
    embedding_dim = query.shape[-1]
    query, key = query.reshape(batch, count, embedding_dim), key.reshape(batch, count, embedding_dim)
    similarities = torch.bmm(query, key.transpose(2, 1))
    matched = similarities.diagonal(dim1=1, dim2=2)
    best = similarities.max(-1, keepdim=True).values
    tied = (similarities - best).abs() <= 1e-6
    retrieval = tied.diagonal(dim1=1, dim2=2).float() / tied.sum(-1)
    diagonal = torch.eye(count, device=query.device, dtype=torch.bool)[None]
    hardest = similarities.masked_fill(diagonal, -torch.inf).max(-1).values
    query_spread = (query - query.mean(1, keepdim=True)).square().sum(-1).mean(1)
    key_spread = (key - key.mean(1, keepdim=True)).square().sum(-1).mean(1)
    return {
        "feature_shape": [batch, channels, height, width],
        "embedding_dim": embedding_dim,
        "patches_per_image": count,
        "patch_ids": patch_ids.detach().cpu().tolist(),
        "samples": batch * count,
        "negatives_per_patch": count - 1,
        "loss": float(per_patch.mean()),
        "retrieval_top1": float(retrieval.mean()),
        "retrieval_chance": 1.0 / count,
        "uniform_loss": math.log(count),
        "positive_probability": float((-per_patch).exp().mean()),
        "matched_cosine": float(matched.mean()),
        "positive_minus_hardest_negative_cosine": float((matched - hardest).mean()),
        "query_zero_norm_fraction": float((query_raw.norm(dim=-1) <= 1e-6).float().mean()),
        "key_zero_norm_fraction": float((key_raw.norm(dim=-1) <= 1e-6).float().mean()),
        "query_spatial_variance": float(query_spread.mean()),
        "key_spatial_variance": float(key_spread.mean()),
        "query_collapsed_image_fraction": float((query_spread <= 1e-6).float().mean()),
        "key_collapsed_image_fraction": float((key_spread <= 1e-6).float().mean()),
    }


def patchnce_loss(
    query_features: Sequence[torch.Tensor],
    key_features: Sequence[torch.Tensor],
    *,
    temperature: float = .2,
    num_patches: int = 64,
    generator: torch.Generator,
    query_projector=None,
    key_projector=None,
):
    """Return mean PatchNCE and JSON-safe diagnostics for paired BCHW maps.

    Query features must remain live; source keys are detached before transfer
    and sampling. The same random locations are selected for both maps and all
    images within each layer. Negatives are other locations of the same source
    image only. Optional domain-specific MLPs transform sampled patches before
    normalization, as in CUT/DCLGAN. The entire key path (including its MLP) is
    detached; bidirectional training updates each MLP through its query path.
    Layer means receive equal weight even when resolutions differ.

    The caller owns/checkpoints ``generator``. This function does not consume
    global RNG. A private CPU generator also avoids GPU randperm variability.
    Exactly zero or collapsed maps retain upstream loss behavior and are logged,
    never silently dropped or counted as perfect retrieval.
    """
    if (not isinstance(query_features, (list, tuple)) or not query_features
            or not isinstance(key_features, (list, tuple))
            or len(query_features) != len(key_features)):
        raise ValueError("PatchNCE requires equally sized nonempty sequences of BCHW feature maps.")
    if isinstance(num_patches, bool) or not isinstance(num_patches, int) or num_patches < 2:
        raise ValueError("PatchNCE num_patches must be an integer >= 2.")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("PatchNCE temperature must be finite and positive.")
    if not isinstance(generator, torch.Generator):
        raise ValueError("PatchNCE requires an explicit private torch.Generator.")
    first = query_features[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("PatchNCE requires BCHW tensors.")
    batch, device = first.shape[0], first.device
    if (query_projector is None) != (key_projector is None):
        raise ValueError("PatchNCE needs both query and key projectors, or neither.")
    if query_projector is not None:
        if (len(query_projector.channels) != len(query_features)
                or len(key_projector.channels) != len(key_features)
                or query_projector.projection_dim != key_projector.projection_dim):
            raise ValueError("PatchNCE projectors must match the layers and output dimension.")
    # Validate before sampling so malformed inputs cannot advance private RNG.
    for layer, (query, key) in enumerate(zip(query_features, key_features)):
        if (not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor)
                or query.ndim != 4 or key.shape != query.shape
                or min(query.shape) < 1 or query.shape[0] != batch
                or query.shape[2] * query.shape[3] < 2 or query.device != device
                or not query.is_floating_point() or not key.is_floating_point()):
            raise ValueError("PatchNCE maps must have paired BCHW shapes, a common query batch/device and >= 2 locations.")
        if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(key).all()):
            raise FloatingPointError("Non-finite PatchNCE feature maps.")
        if query_projector is not None:
            for projector, feature in ((query_projector, query), (key_projector, key)):
                if (projector.channels[layer] != feature.shape[1]
                        or next(projector.parameters()).device != feature.device):
                    raise ValueError("PatchNCE projector channels/device must match its encoder maps.")

    criterion = PatchNCELoss(SimpleNamespace(
        batch_size=batch, nce_T=float(temperature),
        nce_includes_all_negatives_from_minibatch=False,
    ))
    losses, layer_metrics = [], []
    with torch.autocast(device_type=device.type, enabled=False):
        for layer, (query_map, key_map) in enumerate(zip(query_features, key_features)):
            query_map = query_map.float()
            key_map = key_map.detach().float()
            locations = query_map.shape[2] * query_map.shape[3]
            patch_ids = torch.randperm(locations, generator=generator, device=generator.device)[:min(num_patches, locations)]
            patch_ids = patch_ids.to(device=device)
            # The original sampler uses BCHW -> B(HW)C -> (BP)C in this order.
            query_raw = query_map.permute(0, 2, 3, 1).flatten(1, 2)[:, patch_ids].flatten(0, 1)
            key_raw = key_map.permute(0, 2, 3, 1).flatten(1, 2)[:, patch_ids.to(key_map.device)].flatten(0, 1)
            if query_projector is not None:
                query_raw = query_projector(query_raw, layer)
                # Keep the key MLP on the source encoder's device. Detaching
                # only its input would incorrectly train the key-side MLP.
                with torch.no_grad():
                    key_raw = key_projector(key_raw, layer)
            key_raw = key_raw.detach().to(device)
            if not torch.isfinite(query_raw).all() or not torch.isfinite(key_raw).all():
                raise FloatingPointError("Non-finite PatchNCE sampled embeddings.")
            query, key = _normalize_patches(query_raw), _normalize_patches(key_raw)
            per_patch = criterion(query, key)
            losses.append(per_patch.mean())
            metrics = _layer_metrics(query, key, query_raw, key_raw, per_patch, patch_ids, query_map.shape)
            metrics["layer_index"] = layer
            layer_metrics.append(metrics)
        loss = torch.stack(losses).mean()
    mean_fields = (
        "retrieval_top1", "retrieval_chance", "uniform_loss", "positive_probability",
        "matched_cosine", "positive_minus_hardest_negative_cosine",
        "query_zero_norm_fraction", "key_zero_norm_fraction", "query_spatial_variance",
        "key_spatial_variance", "query_collapsed_image_fraction", "key_collapsed_image_fraction",
    )
    metrics = {
        "protocol": PATCHNCE_MLP_PROTOCOL if query_projector is not None else PATCHNCE_PROTOCOL,
        "upstream_commit": CUT_COMMIT,
        "loss": float(loss.detach()),
        "temperature": float(temperature),
        "requested_patches_per_image": num_patches,
        "batch_size": batch,
        "num_layers": len(layer_metrics),
        "samples": sum(item["samples"] for item in layer_metrics),
        "negative_scope": "within_source_image",
        "projector": "domain_mlp" if query_projector is not None else "none",
        "keys_detached": True,
        "layers": layer_metrics,
        "interpretation": (
            "Same-location retrieval in learned patch-projection space; not independent image quality or target realism. Spread is measured after the MLP; collapsed embeddings receive chance-level tie credit."
            if query_projector is not None else
            "Same-location retrieval in own encoder maps; not an independent image-quality or target-realism metric. Collapsed maps receive chance-level tie credit."),
    }
    metrics.update({name: sum(item[name] for item in layer_metrics) / len(layer_metrics) for name in mean_fields})
    return loss, metrics
