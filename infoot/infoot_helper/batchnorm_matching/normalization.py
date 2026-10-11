import torch


def normalize_features(v, norm, reference_v):
    if not norm.training:
        return norm(v)
    variance, mean = torch.var_mean(reference_v, dim=0, unbiased=False)
    m = (v - mean) * torch.rsqrt(variance + norm.eps)
    return m * norm.weight + norm.bias if norm.affine else m


def add_matching_features(encoded, batch_norms):
    for name, batch in encoded.items():
        norm = batch_norms[name]
        references, queries = batch["references"], batch["queries"]
        if not norm.track_running_stats:
            raise ValueError("Matching BatchNorm must track running statistics for evaluation.")
        references["m"] = norm(references["v"])
        queries["m"] = normalize_features(queries["v"], norm, references["v"])
        batch["m"] = torch.cat([queries["m"], references["m"]])
