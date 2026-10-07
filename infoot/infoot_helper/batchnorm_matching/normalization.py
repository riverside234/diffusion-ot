import torch


def add_matching_features(encoded, batch_norms):

    for name, batch in encoded.items():
        norm = batch_norms[name]
        references, queries = batch["references"], batch["queries"]
        if not norm.track_running_stats:
            raise ValueError("Matching BatchNorm must track running statistics for evaluation.")
        references["m"] = norm(references["v"])
        if norm.training:
            variance, mean = torch.var_mean(references["v"], dim=0, unbiased=False)
            matching = (queries["v"] - mean) * torch.rsqrt(variance + norm.eps)
            if norm.affine:
                matching = matching * norm.weight + norm.bias
            queries["m"] = matching
        else:
            queries["m"] = norm(queries["v"])
