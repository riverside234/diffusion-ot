"""KL-Dykstra projection of partial nonnegative coupling factors.

Q1<=a, R1<=b, Q.T1=R.T1=g, sum(g)=mass, g>=min_g.
The three convex sets have closed-form KL projections. Corrections live in
row/column vectors; no coupling or patch kernel is constructed. This is our
factor-space extension of Bregman/Dykstra projection, not a POT partial solver.
"""
import torch


def log_simplex_floor(values, mass, floor):
    """KL projection onto a simplex with a lower bound (GPU water filling)."""
    k = len(values)
    if not 0 < k*floor < mass:
        raise ValueError("Partial mass must exceed rank * min_g.")
    ordered, _ = values.sort(descending=True)
    # For each possible number of free coordinates, compute its multiplier.
    counts = torch.arange(1, k+1, device=values.device, dtype=values.dtype)
    remaining = mass - (k-counts)*floor
    shifts = remaining.log() - ordered.logcumsumexp(0)
    valid = ordered + shifts >= values.new_tensor(floor).log()
    free = valid.sum().clamp_min(1)-1
    return torch.maximum(values + shifts[free], values.new_tensor(floor).log())


@torch.no_grad()
def partial_project(q, r, g, config):
    if any((t <= 0).any() or not torch.isfinite(t).all() for t in (q, r, g)):
        raise ValueError("Mirror proposals must be finite and strictly positive for log-KL projection.")
    mass = config["transported_mass"]
    lq, lr, lg = q.log(), r.log(), g.log()
    aq, ar = q.new_zeros(len(q)), r.new_zeros(len(r))
    bq, br, bg, cg = (torch.zeros_like(g) for _ in range(4))
    loga, logb = q.new_tensor(1/len(q)).log(), r.new_tensor(1/len(r)).log()
    for iteration in range(config["projection_iterations"]):
        old = (lq, lr, lg)
        # Capacity halfspaces. Dykstra corrections need only one value per row.
        iq, ir = lq+aq[:, None], lr+ar[:, None]
        aq = (iq.logsumexp(1)-loga).clamp_min(0)
        ar = (ir.logsumexp(1)-logb).clamp_min(0)
        lq, lr = iq-aq[:, None], ir-ar[:, None]
        # Common column sums: geometric mean of the three incoming marginals.
        iq, ir, ig = lq+bq, lr+br, lg+bg
        sq, sr = iq.logsumexp(0), ir.logsumexp(0)
        lg = (sq+sr+ig)/3
        bq, br, bg = sq-lg, sr-lg, ig-lg
        lq, lr = iq-bq, ir-br
        # Fixed transported mass and positive latent-component floor.
        ig = lg+cg
        lg = log_simplex_floor(ig, mass, config["min_g"])
        cg = ig-lg
        if iteration % 10 == 0 or iteration+1 == config["projection_iterations"]:
            delta = max(float((a-b).abs().max()) for a, b in zip(old, (lq, lr, lg)))
            if delta <= config["projection_tolerance"]:
                return (lq.exp(), lr.exp(), lg.exp()), iteration+1
    raise ValueError(f"Partial factor KL projection did not converge in {config['projection_iterations']} iterations.")
