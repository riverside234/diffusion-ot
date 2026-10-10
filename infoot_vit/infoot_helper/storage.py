"""Float32 disk artifacts, float64 arithmetic, explicit roundoff budgets.

For nonnegative values RN32 has |q-x| <= u*x + eta, u=2^-24,
eta=2^-150 (half the smallest subnormal). Marginal/mass storage bounds
add this to the ORIGINAL solver tolerance; solver checks never use them.
"""
from __future__ import annotations

import torch
from .device import move

VERSION = "float32_storage_v1"
U = 2. ** -24
ETA = 2. ** -150


def quantize(tensor):
    x = tensor.detach().cpu().double()
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError("Plans/kernels must be finite and nonnegative before storage.")
    q = x.float()
    error = (q.double() - x).abs()
    if not torch.isfinite(q).all() or (error > U * x + ETA).any():
        raise ValueError("Float32 storage exceeds the round-to-nearest error bound.")
    return q, dict(max_abs_error=float(error.max()), l1_error=float(error.sum()),
        relative_l1_error=float(error.sum() / x.sum()) if x.sum() else 0.,
        underflow_entries=int(((x > 0) & (q == 0)).sum()), entries=x.numel())


def store_plan_state(state):
    state = move(state, "cpu")  # Serialization boundary; original/quantized sums use the same arithmetic.
    original = state["plan"].double()
    q, errors = quantize(original)
    plan = q.double()
    errors.update(row_sum_error_max=float((plan.sum(1) - original.sum(1)).abs().max()),
                  column_sum_error_max=float((plan.sum(0) - original.sum(0)).abs().max()),
                  mass_error=float((plan.sum() - original.sum()).abs()))
    state.update(plan=q, storage=dict(version=VERSION, plan_dtype="float32", marginal_dtype="float64",
                                      roundoff_u=U, underflow_eta=ETA, quantization=errors))
    # Sums are recomputed from the serialized values using the loading arithmetic.
    state.update(r=plan.sum(1), c=plan.sum(0))
    return state


def plan_tolerances(state, a, b, mass, config):
    plan = state["plan"]
    storage = state.get("storage")
    if storage is None:
        if plan.dtype != torch.float64:
            raise ValueError("Float32 plans require versioned storage metadata.")
        return (torch.full_like(a, config["feasibility_tolerance"]),
                torch.full_like(b, config["feasibility_tolerance"]), config.get("mass_tolerance", 1e-8))
    if (storage.get("version") != VERSION or storage.get("plan_dtype") != "float32"
            or storage.get("marginal_dtype") != "float64" or plan.dtype != torch.float32
            or storage.get("roundoff_u") != U or storage.get("underflow_eta") != ETA):
        raise ValueError("Plan storage dtype/version mismatch.")
    p = plan.double()
    if (state.get("r") is None or state.get("c") is None
            or state["r"].dtype != torch.float64 or state["c"].dtype != torch.float64
            or not torch.equal(state["r"], p.sum(1)) or not torch.equal(state["c"], p.sum(0))):
        raise ValueError("Stored marginals must be float64 sums of the quantized plan.")
    row_solver = config["feasibility_tolerance"]
    mass_solver = config.get("mass_tolerance", 1e-8)
    def bound(limit, tolerance, count):
        # gamma_n bounds any order of n-1 nonnegative double additions.
        # Account for reductions on both the original and stored side.
        nu = count * 2. ** -53
        if nu >= 1:
            raise ValueError("Plan is too large for a finite reduction bound.")
        gamma = nu / (1 - nu)
        original_upper = (limit + tolerance) / (1 - gamma)
        quantization = U * original_upper + count * ETA
        reduction = gamma * (2 * original_upper + quantization)
        return tolerance + quantization + reduction
    return (bound(a, row_solver, len(b)), bound(b, row_solver, len(a)),
            bound(mass, mass_solver, p.numel()))


def validate_plan(state, a, b, mass, config, *, balanced=False):
    p = state["plan"].double()
    if p.shape != (len(a), len(b)) or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid stored plan shape/values.")
    row_tol, col_tol, mass_tol = plan_tolerances(state, a, b, mass, config)
    rows, cols = p.sum(1) - a, p.sum(0) - b
    row_error = rows.abs() if balanced else rows.clamp_min(0)
    col_error = cols.abs() if balanced else cols.clamp_min(0)
    if (row_error > row_tol).any() or (col_error > col_tol).any() or abs(float(p.sum()) - mass) > mass_tol:
        raise ValueError("Stored plan violates its solver-plus-storage tolerance; no renormalization applied.")
    return dict(mass=float(p.sum()), mass_error=abs(float(p.sum()) - mass),
                row_error_max=float(row_error.max()), column_error_max=float(col_error.max()),
                mass_tolerance=mass_tol, row_tolerance_max=float(row_tol.max()),
                column_tolerance_max=float(col_tol.max()),
                confidence_roundoff_tolerance=float((row_tol / a).max()))


def store_kernels(shared):
    result, errors = dict(shared), {}
    for name in ("kx", "ky"):
        result[name], errors[name] = quantize(shared[name])
    result["storage"] = dict(version=VERSION, kernel_dtype="float32", quantization=errors)
    return result


def load_kernels(state):
    storage = state.get("storage")
    for name in ("kx", "ky"):
        k = state[name]
        if storage is not None and (storage.get("version") != VERSION
                or storage.get("kernel_dtype") != "float32" or k.dtype != torch.float32):
            raise ValueError("Kernel storage dtype/version mismatch.")
        if storage is None and k.dtype != torch.float64:
            raise ValueError("Float32 kernels require storage metadata.")
        if (k.ndim != 3 or k.shape[1] != k.shape[2] or not torch.isfinite(k).all()
                or (k < 0).any() or (k > 1).any() or not (k.diagonal(dim1=1, dim2=2) == 1).all()
                or not torch.equal(k, k.transpose(1, 2))):
            raise ValueError("Invalid saved training kernels.")
    return dict(state, kx=state["kx"].double(), ky=state["ky"].double())
