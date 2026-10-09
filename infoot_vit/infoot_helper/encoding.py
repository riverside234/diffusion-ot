import torch
from torch.utils.checkpoint import checkpoint


def encode_batches(domains, latents, query_count=8, encode_batch_size=16):
    if query_count < 1:
        raise ValueError("query_count must be positive.")
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive.")

    encoded = {}

    for name in ("cat", "dog"):
        context = domains[name]
        x0 = latents[name].to(
            device=context.device,
            dtype=context.model_dtype,
        )

        v = torch.cat([
            checkpoint(context.branch.encode, chunk, use_reentrant=False)
            for chunk in x0.split(encode_batch_size)
        ]).float()
        references = v[query_count:]
        if len(references) < 2:
            raise ValueError("Need at least two reference images per domain.")

        encoded[name] = {
            "x0": x0,
            "v": v,
            "references": {
                "v": references,
            },
            "queries": {
                "x0": x0[:query_count],
                "v": v[:query_count],
            },
        }

    return encoded
