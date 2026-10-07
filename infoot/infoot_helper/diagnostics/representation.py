from diffusion_ot.models.matching_head import matching_geometry_diagnostics


def representation_log(references):
    fields = []
    for name, v in references.items():
        stats = matching_geometry_diagnostics(v, v)
        rank = stats["matching_covariance_effective_rank"]
        variance = stats["raw_code_variance"]
        maximum = min(len(v) - 1, v.shape[1])
        fields.append(f"rank_{name}={rank:.2f}/{maximum} var_{name}={variance:.3e}")
    return " ".join(fields)
