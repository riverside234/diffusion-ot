"""Shared per-image KDE factors for partial pairs (no P x P kernels)."""
import torch

from . import kernels
from .storage import VERSION
from ..infoot_helper.device import move
from ..infoot_helper.storage import quantize


def parameters(state, index):
    return dict(method="normalized_positive_gaussian_v1",mean=state["means"][index],omega=state["omega"],
        scale=state["scales"][index],sigma=state["sigmas"][index],h=state["h"],rank=state["rank"],seed=state["seed"])


def support_threshold(f, confidence):
    policy = confidence["support_calibration"]
    if policy == "disabled":
        return dict(policy=policy,threshold=None)
    if policy == "fixed":
        return dict(policy=policy,threshold=confidence["support_log_threshold"])
    if len(f) < 2:
        raise ValueError("Leave-one-out support calibration needs at least two patches.")
    # Exclusive prefix/suffix sums avoid cancellation from total-minus-self.
    prefix = torch.cat((f.new_zeros(1,f.shape[1]),f[:-1].cumsum(0)))
    suffix = torch.cat((f[1:].flip(0).cumsum(0).flip(0),f.new_zeros(1,f.shape[1])))
    density = (f*(prefix+suffix)).sum(1)/(len(f)-1)
    if (density <= 0).any():
        raise ValueError("Zero low-rank leave-one-out density; increase kernel rank/bandwidth or select an explicit fixed support threshold.")
    return dict(policy=policy,threshold=float(torch.quantile(density.log(),confidence["support_quantile"])),
        quantile=confidence["support_quantile"],normalization="mean over P-1 nonself approximate kernels",heuristic=True)


def fit_collection(images, rank, kernel, confidence, seed, log, name):
    omega = torch.randn(images.shape[-1],rank,device=images.device,dtype=torch.float64,
                        generator=torch.Generator(device=images.device).manual_seed(seed))
    factors,states,support,checks = [],[],[],[]
    for i,x in enumerate(images):
        f,state = kernels.fit_features(x,rank,kernel["h"],seed,omega=omega)
        factors.append(f); states.append(state)
        support.append(support_threshold(f,confidence))
        check = kernels.error_report(x,f,state,seed=seed+100+i,count=kernel["check_pairs"],
            density_queries=min(kernel["density_queries"],max(1,len(x)-1)),chunk_size=max(1,min(64,len(x)-1)))
        checks.append(check)
        log.append_jsonl("kernel_checks.jsonl",dict(domain=name,image_index=i,**check))
    exact = torch.stack(factors)
    saved,error = quantize(exact)
    state = dict(method="normalized_positive_gaussian_v1",factors=saved,means=torch.stack([s["mean"] for s in states]).cpu(),omega=omega.cpu(),
        scales=[s["scale"] for s in states],sigmas=[s["sigma"] for s in states],h=kernel["h"],rank=rank,seed=seed,
        support=support,storage_version=VERSION,quantization=error,
        approximation=dict(relative_rmse_max=max(s["relative_rmse"] for s in checks),
            relative_rmse_mean=sum(s["relative_rmse"] for s in checks)/len(checks),
            density_relative_error_max=max(s["density_relative_error_max"] for s in checks)))
    return exact,state


def validate_collection(state, shape, rank):
    n,p,d = shape
    f = state["factors"]
    if (state["storage_version"] != VERSION or state.get("method") != "normalized_positive_gaussian_v1"
            or state["rank"] != rank or f.dtype != torch.float32
            or f.shape != (n,p,rank) or not torch.isfinite(f).all() or (f < 0).any()
            or (f.double().square().sum(-1)-1).abs().max() > 2e-6):
        raise ValueError("Invalid per-image kernel factor storage.")
    for key,size in (("means",(n,d)),("omega",(d,rank))):
        t = state[key]
        if t.shape != size or t.dtype != torch.float64 or not torch.isfinite(t).all():
            raise ValueError("Invalid per-image kernel preprocessing parameters.")
    if len(state["scales"]) != n or len(state["sigmas"]) != n or len(state["support"]) != n:
        raise ValueError("Per-image kernel/support count changed.")
    if not 0 < state["h"] < float("inf") or type(state["seed"]) is not int:
        raise ValueError("Invalid per-image kernel bandwidth/seed.")
    for scale,sigma,support in zip(state["scales"],state["sigmas"],state["support"]):
        if not 0 < scale < float("inf") or not 0 < sigma < float("inf") or sigma != scale*state["h"]:
            raise ValueError("Invalid training-derived bandwidth.")
        policy,threshold = support["policy"],support["threshold"]
        if (policy not in {"fixed","disabled","fit_leave_one_out_log_density"}
                or (threshold is None) != (policy == "disabled")
                or (threshold is not None and not -float("inf") < threshold < float("inf"))):
            raise ValueError("Invalid saved per-image support calibration.")


def restore_collection(images,state):
    state = move(state,images.device)
    factors = torch.stack([kernels.features(x,parameters(state,i)) for i,x in enumerate(images)])
    if not torch.equal(factors.float(),state["factors"]):
        raise ValueError("Float64 rebuilt pair kernels differ from their saved snapshot.")
    return factors
