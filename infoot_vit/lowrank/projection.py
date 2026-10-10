"""Stream conditional patch scores from small kernel-rank products."""
import torch


def project_scores(left, target_factors, target, density, chunk_size):
    """Normalize inside ONE target image, including original-density correction.

    Score blocks are [query patches, <=chunk_size target patches]. Never return
    a full plan/kernel/score matrix. Row rescaling cancels on normalization.
    """
    if (density <= 0).any() or not torch.isfinite(left).all() or (left < 0).any():
        raise ValueError("Invalid saved-factor conditional density/product.")
    scale = left.max(1,keepdim=True).values
    left = left / scale.masked_fill(scale == 0,1.)
    target_count = target.shape[-2]
    total = left.new_zeros(*target.shape[:-2],len(left))
    energy = torch.zeros_like(total)
    numerator = left.new_zeros(*total.shape,target.shape[-1])
    chunk_size = min(chunk_size,max(1,target_count-1))
    for start in range(0,target_count,chunk_size):
        stop = start+chunk_size
        scores = (left@target_factors[...,start:stop,:].transpose(-1,-2))/density[...,None,start:stop]
        total += scores.sum(-1)
        energy += (scores*scores.clamp_min(1e-300).log()).sum(-1)
        numerator += scores@target[...,start:stop,:]
    positive = total > 0
    mapped, entropy = torch.zeros_like(numerator),torch.zeros_like(total)
    mapped[positive] = numerator[positive]/total[positive].unsqueeze(-1)
    entropy[positive] = total[positive].log()-energy[positive]/total[positive]
    if not torch.isfinite(mapped).all() or not torch.isfinite(entropy).all():
        raise ValueError("Nonfinite low-rank conditional projection.")
    return mapped,dict(weight_row_sums=positive.to(left.dtype),patch_entropy=entropy)


def partial_project(query_features, fx, fy, target, factors, support, chunk_size):
    q,r,g = factors
    original_density = query_features@fx.mean(0)
    retained = q@(r.sum(0)/g)
    kept_density = query_features@(fx.T@retained)
    has_support = original_density > 0
    raw = torch.zeros_like(original_density)
    raw[has_support] = kept_density[has_support]/original_density[has_support]
    log_density = original_density.log()
    valid = has_support if support["threshold"] is None else has_support & (log_density >= support["threshold"])
    left = (query_features@(fx.T@q)/g)@(fy.T@r).T
    mapped,diag = project_scores(left,fy,target,fy@fy.mean(0),chunk_size)
    # Original b is uniform, so its scalar cancels in the conditional row.
    lost = valid & (raw > 0) & (diag["weight_row_sums"] == 0)
    if lost.any():
        raise ValueError("Positive retained mass but zero low-rank conditional scores; inspect kernel underflow/rank.")
    underflow = has_support & (kept_density == 0) & (left.sum(1) > 0)
    if (underflow & valid).any():
        raise ValueError("Retained confidence underflow for a support-valid query.")
    diag.update(raw_confidence=raw,support_valid=valid,log_density=log_density,
                confidence_underflow=underflow,score_log_retry=torch.zeros_like(valid))
    return mapped,raw*valid,diag
