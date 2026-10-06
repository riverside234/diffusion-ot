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

        v = context.branch.encode(x0).float()
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
