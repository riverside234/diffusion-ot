def feature_stats(v, eps=1e-6):
    v = v.float()
    mean = v.mean(dim=0, keepdim=True)
    std = (v.var(dim=0, unbiased=False, keepdim=True) + eps).sqrt()
    return mean, std


def standardize(v, stats):
    mean, std = stats
    return (v.float() - mean) / std


def encode_batches(domains, latents, query_count=8):
    if query_count < 1:
        raise ValueError("query_count must be positive.")

    encoded = {}

    for name in ("cat", "dog"):
        context = domains[name]
        x0 = latents[name].to(
            device=context.device,
            dtype=context.model_dtype,
        )

        v = context.branch.encode(x0)
        references = v[query_count:]
        if len(references) < 2:
            raise ValueError("Need at least two reference images per domain.")
        stats = feature_stats(references)

        encoded[name] = {
            "x0": x0,
            "v": v,
            "references": {
                "v": references,
                "m": standardize(references, stats),
            },
            "queries": {
                "x0": x0[:query_count],
                "v": v[:query_count],
                "m": standardize(v[:query_count], stats),
            },
        }

    return encoded
