"""Checked float32 factors; quantization never repairs the saved coupling."""
import hashlib
import torch

from ..infoot_helper.storage import quantize, U, ETA
from ..infoot_helper.feature_bank import digest
from .solver import check, constraint_args

VERSION = "nonnegative_lr_float32_v1"


def checkpoint_digest(state):
    """Self-contained integrity check so latest.pt needs no non-atomic sidecar."""
    h = hashlib.sha256()
    for key in ("q", "r", "g"):
        value = state[key].cpu().contiguous()
        h.update(digest(dict(key=key, shape=list(value.shape), dtype=str(value.dtype))).encode())
        data = memoryview(value.numpy()).cast("B")
        for start in range(0, len(data), 8*1024*1024):
            h.update(data[start:start+8*1024*1024])
    h.update(digest({key:state.get(key) for key in ("step", "fit_fingerprint", "record")}).encode())
    return h.hexdigest()


def pack_factors(factors,config,**metadata):
    check(*factors,config["constraint_tolerance"],config["min_g"],**constraint_args(config))
    state = dict(metadata,storage_version=VERSION,dtype="float32",roundoff_u=U)
    errors = {}
    for key,tensor in zip(("q","r","g"),factors):
        state[key],errors[key] = quantize(tensor)
    state["quantization"] = errors
    q,r,g = (state[key].double() for key in ("q","r","g"))
    state["q_rows"],state["r_rows"] = q.sum(1),r.sum(1)
    state["q_columns"],state["r_columns"] = q.sum(0),r.sum(0)
    state["storage_constraints"] = validate_factors(state,config)
    return state


def storage_tolerance(q,r,g,config):
    # Product/quotient error of Q*R/g is <= (1+u)^2/(1-u)-1.
    # 8u covers that, shared-column differences and double reduction rounding;
    # include a conservative gamma_n reduction and subnormal allowance too.
    count = max(q.numel(),r.numel())
    gamma = count*2.**-53/(1-count*2.**-53)
    subnormal = (q.numel()+r.numel())*ETA/max(float(g.min()),1e-300)*max(len(q),len(r))
    return config["constraint_tolerance"]+8*U+4*gamma+subnormal


def validate_factors(state,config,shape=None):
    if (state.get("storage_version") != VERSION or state.get("dtype") != "float32"
            or state.get("roundoff_u") != U or any(state[key].dtype != torch.float32 for key in ("q","r","g"))):
        raise ValueError("Low-rank factor storage identity/dtype mismatch.")
    q,r,g = (state[key].double() for key in ("q","r","g"))
    if shape is not None and (q.shape != (shape[0],shape[2]) or r.shape != (shape[1],shape[2])):
        raise ValueError("Low-rank support/rank shape mismatch.")
    for key,value in (("q_rows",q.sum(1)),("r_rows",r.sum(1)),("q_columns",q.sum(0)),("r_columns",r.sum(0))):
        if state[key].dtype != torch.float64 or not torch.equal(state[key],value):
            raise ValueError("Stored factor marginals disagree with quantized factors.")
    tolerance = storage_tolerance(q,r,g,config)
    return dict(check(q,r,g,tolerance,config["min_g"],**constraint_args(config)),storage_relative_tolerance=tolerance)
