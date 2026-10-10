"""Strict configuration and resource accounting for the separate experiment."""
import math
from copy import deepcopy

from ..infoot_helper.fit_mapping import validate_config, CONFIDENCE
from ..infoot_helper.partial import solver_config
from ..infoot_helper.device import resolve_device

MODE = "grouped_patch_lowrank"
SCHEMA = "siglip_lowrank_grouped_patch_v1"
PARTIAL_MODE = "grouped_partial_lowrank"
PARTIAL_SCHEMA = "siglip_lowrank_grouped_partial_v1"
DEFAULT = dict(mode=MODE,source_bank="data/infoot_vit/cat_train",target_bank="data/infoot_vit/dog_train",
    sampling=dict(images_per_domain=2000,seed=42),transport_rank=256,kernel_rank=256,
    kernel=dict(h=.4,seed=4201,check_pairs=4096,density_queries=32,error_warn_relative_rmse=.5),
    estimator=dict(seed=4202,samples_per_row=2,audit_samples=32768,exact=False),
    optimizer=dict(seed=4203,lam=.1,reg=.05,step_size=1.,max_steps=300,max_backtracks=12,
        projection_iterations=10000,projection_tolerance=1e-12,constraint_tolerance=1e-8,min_g=1e-10,
        stationarity_tolerance=1e-6,objective_tolerance=1e-12,log_floor=1e-300,chunk_size=4096,
        audit_every=10,checkpoint_every=10,log_every=1),
    image_solver=dict(h=.4,lam=.1,reg=.05,cost_scale="mean",max_outer_steps=300,max_inner_steps=10000),
    projection=dict(bandwidth_multiplier=1.,query_chunk_size=1,target_chunk_size=8,
                    selection="mean",top_k_images=None,seed=42,patch_chunk_size=64),
    partial=dict(keep_mass=.8),fit_pair_top_k=None,
    resources=dict(max_artifact_bytes=4000000000,max_working_gib=64),device="cuda",threads=4)
DEFAULT["image_solver"] = solver_config(DEFAULT["image_solver"])
DEFAULT["projection"]["confidence"] = deepcopy(CONFIDENCE)


def canonical(raw):
    c = deepcopy(DEFAULT)
    if not isinstance(raw,dict) or raw.keys()-c.keys():
        raise ValueError("Unknown low-rank configuration fields.")
    if raw.get("mode") == PARTIAL_MODE:
        c.update(transport_rank=64,kernel_rank=64,fit_pair_top_k=8)
        c["optimizer"]["log_every"] = 25
        c["estimator"]["audit_samples"] = 2048
    for key,value in raw.items():
        if isinstance(c[key],dict):
            if not isinstance(value,dict) or value.keys()-c[key].keys():
                raise ValueError(f"Unknown low-rank {key} fields.")
            c[key].update(value)
        else: c[key] = value
    if c["mode"] not in {MODE,PARTIAL_MODE}:
        raise ValueError(f"This entry point requires {MODE} or {PARTIAL_MODE}.")
    resolve_device(c["device"],check=False)
    if not 0 < c["partial"]["keep_mass"] <= 1:
        raise ValueError("partial.keep_mass must be in (0,1].")
    if c["fit_pair_top_k"] is not None and (type(c["fit_pair_top_k"]) is not int or c["fit_pair_top_k"] < 1):
        raise ValueError("fit_pair_top_k must be positive or null (all pairs).")
    if c["mode"] == MODE and c["fit_pair_top_k"] is not None:
        raise ValueError("Pair selection belongs only to grouped_partial_lowrank.")
    for value in (c["transport_rank"],c["kernel_rank"],c["threads"],c["sampling"]["images_per_domain"],
                  c["kernel"]["check_pairs"],c["kernel"]["density_queries"],c["estimator"]["samples_per_row"],
                  c["estimator"]["audit_samples"],c["projection"]["target_chunk_size"],c["projection"]["patch_chunk_size"]):
        if type(value) is not int or value < 1: raise ValueError("Counts and ranks must be positive integers.")
    for section in ("sampling","kernel","estimator","optimizer"):
        if type(c[section]["seed"]) is not int: raise ValueError("Seeds must be integers.")
    if type(c["estimator"]["exact"]) is not bool: raise ValueError("estimator.exact must be boolean.")
    for key,value in c["optimizer"].items():
        if key == "seed": continue
        if key in {"max_steps","max_backtracks","projection_iterations","chunk_size","audit_every","checkpoint_every","log_every"}:
            if type(value) is not int or value < 1: raise ValueError(f"Invalid {key}.")
        elif not math.isfinite(value) or value < 0 or (value == 0 and key not in {"lam","reg","objective_tolerance"}):
            raise ValueError(f"Invalid {key}.")
    mass = c["partial"]["keep_mass"] if c["mode"] == PARTIAL_MODE else 1.
    if c["optimizer"]["min_g"] >= mass/c["transport_rank"] or c["optimizer"]["log_floor"] >= 1:
        raise ValueError("min_g must be below mass/rank and log_floor below 1.")
    for value in (c["kernel"]["h"],c["kernel"]["error_warn_relative_rmse"],*c["resources"].values()):
        if not math.isfinite(value) or value <= 0: raise ValueError("Invalid bandwidth/resource/error threshold.")
    if c["projection"]["bandwidth_multiplier"] != 1:
        raise ValueError("This experiment fixes KDE bandwidths at fit time; multiplier must be 1.")
    extra_chunk = c["projection"].pop("target_chunk_size")
    patch_chunk = c["projection"].pop("patch_chunk_size")
    base = validate_config(dict(mode="grouped_patch",source_bank=c["source_bank"],target_bank=c["target_bank"],
                               solver=c["image_solver"],projection=c["projection"]))
    c["image_solver"],c["projection"] = base["solver"],base["projection"]
    c["projection"]["target_chunk_size"] = extra_chunk
    c["projection"]["patch_chunk_size"] = patch_chunk
    return c


def optimizer_config(config):
    partial = config["mode"] == PARTIAL_MODE
    return dict(config["optimizer"],constraint="partial" if partial else "balanced",
                transported_mass=config["partial"]["keep_mass"] if partial else 1.)


def resources(ns,nt,p,d,c):
    n,m,r,k = ns*p,nt*p,c["transport_rank"],c["kernel_rank"]
    samples = n*m if c["estimator"]["exact"] else c["estimator"]["samples_per_row"]*(n+m)
    factors = (n+m)*(r+k)*4
    parameters = 2*d*(k+1)*8
    estimator = (samples+c["estimator"]["audit_samples"])*24  # two int64 IDs + double cost
    router = ns*nt*4
    if c["mode"] == PARTIAL_MODE:
        pairs = ns*(nt if c["fit_pair_top_k"] is None else min(nt,c["fit_pair_top_k"]))
        factors = pairs*(2*p*r+r)*4 + (n+m)*k*4
        parameters = (ns+nt)*d*8+2*d*k*8
        samples = p*p if c["estimator"]["exact"] else 2*p*c["estimator"]["samples_per_row"]
        estimator = (samples+c["estimator"]["audit_samples"])*16  # Shared indices; costs recomputed per pair.
        logs = pairs*(32768+4096*(math.ceil(c["optimizer"]["max_steps"]/c["optimizer"]["log_every"])+1))
        final = factors+parameters+estimator+router+logs+256*2**20
        working = 8*((n+m)*(d+k)+12*ns*nt+24*p*r+8*c["optimizer"]["chunk_size"]*(d+k+r))
        return dict(source_patches=n,target_patches=m,transport_rank=r,kernel_rank=k,selected_pairs=pairs,
            saved_factor_bytes=factors,estimator_bytes=estimator,router_bytes=router,
            estimated_final_artifact_bytes=final,estimated_final_artifact_GB=final/1e9,
            estimated_working_gib=working/2**30,active_checkpoint_and_atomic_temp_bytes=2*(2*p*r+r)*4,
            objective_sample_count_per_pair=samples,full_dense_patch_matrix_bytes=n*m*8,
            note="Per selected image pair: capacity-constrained factors. Shared per-image KDE factors/indices; no dense patch matrices. Log/metadata reserve included, banks excluded.")
    # Include conservative metadata/log reserve in the final-artifact guard.
    final = factors+parameters+estimator+router+256*2**20
    working = 8*((n+m)*d + (n+m)*k + 10*(n+m)*r + 12*ns*nt + 8*c["optimizer"]["chunk_size"]*(d+k+r))
    return dict(source_patches=n,target_patches=m,transport_rank=r,kernel_rank=k,
        saved_factor_bytes=factors,estimator_bytes=estimator,router_bytes=router,
        estimated_final_artifact_bytes=final,estimated_final_artifact_GB=final/1e9,
        active_checkpoint_and_atomic_temp_bytes=2*(n+m)*r*4,
        estimated_working_gib=working/2**30,full_dense_patch_matrix_bytes=n*m*8,
        objective_sample_count=samples,
        note="No Npatch*Mpatch or Npatch^2 arrays. Float64 compute, float32 disk factors; existing banks excluded. Working-memory estimate is conservative, not a measured peak.")
