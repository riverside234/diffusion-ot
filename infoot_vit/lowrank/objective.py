"""Streamed sampled FusedInfoOT on Gamma=Q diag(1/g) R^T.

Exact low-rank KDE algebra; sampled cost, MI outer sum and PLAN entropy.
No factor-entropy substitution, dense plan, dense distance, or smoothing matrix.
"""
import math
import torch


def sample_pairs(n, m, *, seed, per_row=2, count=None, exact=False, device="cpu"):
    rng = torch.Generator(device=device).manual_seed(seed)
    if exact:
        if n*m > 1000000:
            raise ValueError("Exact enumeration is restricted to small numerical references.")
        i, j = torch.arange(n,device=device).repeat_interleave(m), torch.arange(m,device=device).repeat(n)
        kind = "exact_enumeration"
    elif count is not None:
        i, j = torch.randint(n, (count,), generator=rng,device=device), torch.randint(m, (count,), generator=rng,device=device)
        kind = "independent_uniform_with_replacement"
    else:
        i = torch.cat((torch.arange(n,device=device).repeat_interleave(per_row), torch.randint(n, (m*per_row,), generator=rng,device=device)))
        j = torch.cat((torch.randint(m, (n*per_row,), generator=rng,device=device), torch.arange(m,device=device).repeat_interleave(per_row)))
        kind = "two_sided_stratified_with_replacement"
    return dict(i=i, j=j, weight=n*m/len(i), seed=seed, kind=kind, count=len(i),rng_device=torch.device(device).type)


def with_cost(pairs, x, y, scale=None, chunk_size=4096):
    costs = []
    for start in range(0, pairs["count"], chunk_size):
        i, j = pairs["i"][start:start+chunk_size], pairs["j"][start:start+chunk_size]
        costs.append((x[i.to(x.device)]-y[j.to(y.device)]).norm(dim=1))
    costs = torch.cat(costs)
    scale = float(costs.mean()) if scale is None else scale
    if not scale > 0 or not torch.isfinite(costs).all():
        raise ValueError("Invalid sampled Euclidean cost scale.")
    return dict(pairs, cost=costs/scale, cost_scale=scale)


def validate_samples(samples, n, m, config):
    """Validate persisted estimator support, precision, seeds and normalization."""
    ec = config["estimator"]
    for name in ("training", "audit"):
        pairs = samples[name]
        is_train = name == "training"
        count = (n*m if ec["exact"] else ec["samples_per_row"]*(n+m)) if is_train else ec["audit_samples"]
        kind = ("exact_enumeration" if ec["exact"] else "two_sided_stratified_with_replacement") if is_train else "independent_uniform_with_replacement"
        if (pairs["count"] != count or pairs["kind"] != kind
                or pairs["seed"] != ec["seed"] + int(not is_train)
                or pairs["weight"] != n*m/count or not math.isfinite(pairs["cost_scale"])
                or pairs["cost_scale"] <= 0):
            raise ValueError("Sampled objective metadata changed.")
        for key, size in (("i", n), ("j", m)):
            value = pairs[key]
            if value.dtype != torch.int64 or value.shape != (count,) or (value < 0).any() or (value >= size).any():
                raise ValueError("Sampled objective index out of bounds or wrong dtype/shape.")
        cost = pairs["cost"]
        if cost.dtype != torch.float64 or cost.shape != (count,) or not torch.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Invalid saved float64 sampled costs.")
        # Stored indices + the artifact hash preserve support across CPU/GPU.
        # Different device RNG implementations are not assumed to be identical.
        if is_train and pairs.get("rng_device", "cpu") == pairs["i"].device.type:
            regenerated = sample_pairs(n, m, seed=ec["seed"], per_row=ec["samples_per_row"], exact=ec["exact"],device=pairs["i"].device)
            if not all(torch.equal(pairs[key], regenerated[key]) for key in ("i", "j")):
                raise ValueError("Saved estimator differs from deterministic training samples.")
    if samples["training"]["cost_scale"] != samples["audit"]["cost_scale"]:
        raise ValueError("Audit must use the training-derived cost scale.")


def terms(q, r, g, u, v, fx, fy, density_x, density_y, cost, log_floor, log_size, sums=None):
    gamma = (q*r/g).sum(1)
    joint = ((fx@u)*(fy@v)/g).sum(1)
    if sums is None:
        log_ratio = (joint/(density_x*density_y)).clamp_min(log_floor).log()
    else:
        # Exactly the dense partial objective: KDE on Gamma/M and its own
        # transported marginals, then M * E_{Gamma/M}[log density ratio].
        # Sums are differentiable inputs, including M, not stop-grad constants.
        sq, sr = sums
        mass = (sq*sr/g).sum()
        px, py = fx@(u@(sr/g)), fy@(v@(sq/g))
        log_ratio = ((joint/mass).clamp_min(log_floor).log()
                     -(px/mass).clamp_min(log_floor).log()-(py/mass).clamp_min(log_floor).log())
    if (gamma < 0).any() or (joint < 0).any() or not torch.isfinite(log_ratio).all():
        raise ValueError("Negative/nonfinite low-rank plan or KDE density; reassess the factors/kernel.")
    mi = gamma*log_ratio
    # Exact-mass control variates remove the large constant entropy/cost part
    # from sampling noise. evaluate() restores it analytically, with gradients.
    entropy = gamma*(gamma.clamp_min(log_floor).log()+log_size)
    return torch.stack((gamma*(cost-1), mi, entropy, gamma), 1)


def evaluate(q, r, g, fx, fy, pairs, *, lam, reg, log_floor=1e-300, chunk_size=4096, gradient=False, partial=False):
    """Differentiate small sample blocks, then chain through U=Fx.T Q,V=Fy.T R.

    Explicit scatter/reduction avoids an autograd graph over all samples or
    allocating an N*r gather backward tensor once per sample block.
    """
    u, v = fx.T@q, fy.T@r
    sq, sr = q.sum(0), r.sum(0)
    dx, dy = fx@fx.mean(0), fy@fy.mean(0)
    if (dx <= 0).any() or (dy <= 0).any():
        raise ValueError("Nonpositive approximate KDE marginal density.")
    total, sum_squares = q.new_zeros(4), q.new_zeros(4)
    log_size = math.log(len(q)*len(r))
    objective_squares = q.new_zeros(())
    if gradient:
        dq, dr, dg = torch.zeros_like(q), torch.zeros_like(r), torch.zeros_like(g)
        du, dv = torch.zeros_like(u), torch.zeros_like(v)
        dsq, dsr = torch.zeros_like(sq), torch.zeros_like(sr)
    for start in range(0, pairs["count"], chunk_size):
        i = pairs["i"][start:start+chunk_size].to(q.device)
        j = pairs["j"][start:start+chunk_size].to(q.device)
        cost = pairs["cost"][start:start+chunk_size].to(q)
        with torch.set_grad_enabled(gradient):
            inputs = [t.detach().requires_grad_(gradient) for t in ((q[i], r[j], g, u, v, sq, sr) if partial else (q[i], r[j], g, u, v))]
            values = terms(*inputs[:5], fx[i], fy[j], dx[i], dy[j], cost, log_floor, log_size,
                           sums=inputs[5:] if partial else None)
            if gradient:
                loss = (values[:,0]-lam*values[:,1]+reg*values[:,2]).sum()*pairs["weight"]
                gradients = torch.autograd.grad(loss, inputs)
                dq.index_add_(0, i, gradients[0]); dr.index_add_(0, j, gradients[1])
                dg += gradients[2]; du += gradients[3]; dv += gradients[4]
                if partial:
                    dsq += gradients[5]; dsr += gradients[6]
        values = values.detach()
        total += values.sum(0)*pairs["weight"]
        sum_squares += values.square().sum(0)
        objective_squares += (values[:,0]-lam*values[:,1]+reg*values[:,2]).square().sum()
    result = dict(zip(("cost", "mi", "entropy", "estimated_mass"), total.cpu().tolist()))
    sq, sr = q.sum(0), r.sum(0)
    mass = float((sq*sr/g).sum())
    result["cost"] += mass
    result["entropy"] -= (log_size+1)*mass
    result.update(objective=result["cost"]-lam*result["mi"]+reg*result["entropy"], exact_mass=mass)
    # Under balanced marginals: mean row entropy = H(Gamma) - log(n).
    # This inherits sampling error; do not clamp it to a plausible range.
    rows = (q@(sr/g))/mass
    row_entropy = -(result["entropy"]+mass)/mass+math.log(mass)+float((rows*rows.clamp_min(log_floor).log()).sum())
    result["estimated_mean_row_entropy"] = row_entropy
    result["estimated_normalized_row_entropy"] = row_entropy/math.log(len(r)) if len(r)>1 else 0.
    result["latent_component_entropy"] = float(-(g*g.log()).sum())
    # IID standard errors only apply to independent audit samples, not training strata.
    if pairs["kind"] == "independent_uniform_with_replacement":
        n = pairs["count"]
        mean = total/(pairs["weight"]*n)
        variance = (sum_squares/n-mean.square()).clamp_min(0)*n/max(1,n-1)
        result["standard_error"] = dict(zip(("cost", "mi", "entropy", "estimated_mass"),
            ((variance/n).sqrt()*pairs["weight"]*n).cpu().tolist()))
        om = mean[0]-lam*mean[1]+reg*mean[2]
        ov = (objective_squares/n-om.square()).clamp_min(0)*n/max(1,n-1)
        result["standard_error"]["objective"] = float((ov/n).sqrt()*pairs["weight"]*n)
    if gradient:
        dq += fx@du+dsq; dr += fy@dv+dsr
        constant = 1-reg*(log_size+1)
        dq += constant*sr/g; dr += constant*sq/g; dg -= constant*sq*sr/g.square()
        if not all(torch.isfinite(t).all() for t in (dq,dr,dg)):
            raise ValueError("Nonfinite sampled InfoOT factor gradient.")
        return result, (dq,dr,dg)
    return result
