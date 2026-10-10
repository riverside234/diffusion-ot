"""Low-rank partial InfoOT on the router's saved image pairs."""
import json
import os
import warnings
import torch

from ..infoot_helper.feature_bank import digest,checked_file,write_json
from ..infoot_helper.fit_mapping import _verified_save,_cleanup_latest,_file_entry,_read_pair_inventory,_sync_directory
from ..infoot_helper.pair_selection import select_pairs,selection_edges
from ..infoot_helper.device import move
from .config import optimizer_config
from . import objective,pair_kernels
from .units import fit_unit


def fit_pairs(directory,manifest,source,target,x,y,image,log,*,resume=False):
    from .experiment import register
    c = manifest["config"]; options = optimizer_config(c)
    identity = manifest["fit_fingerprint"]
    if "pair_selection" not in manifest["files"]:
        selection = select_pairs(image,source.ids,target.ids,c["fit_pair_top_k"])
        selection["router_sha256"] = manifest["files"]["image"]["sha256"]
        path = directory/"pair_selection.json"
        write_json(path,selection)
        register(directory,manifest,"pair_selection",_file_entry(directory,path))
    selection = json.loads(checked_file(directory,manifest["files"]["pair_selection"]).read_text(encoding="utf-8"))
    if selection["router_sha256"] != manifest["files"]["image"]["sha256"]:
        raise ValueError("Saved selection belongs to a different image router.")
    selected = selection_edges(selection,source.ids,target.ids,c["fit_pair_top_k"])

    def validate_kernel(s):
        for name,images in (("source",x),("target",y)):
            pair_kernels.validate_collection(s[name],images.shape,c["kernel_rank"])

    if "kernels" in manifest["files"]:
        kernel = torch.load(checked_file(directory,manifest["files"]["kernels"]),map_location="cpu",weights_only=True)
        validate_kernel(kernel)
        fx,fy = pair_kernels.restore_collection(x,kernel["source"]),pair_kernels.restore_collection(y,kernel["target"])
    else:
        fx,sx = pair_kernels.fit_collection(x,c["kernel_rank"],c["kernel"],c["projection"]["confidence"],c["kernel"]["seed"],log,"source")
        fy,sy = pair_kernels.fit_collection(y,c["kernel_rank"],c["kernel"],c["projection"]["confidence"],c["kernel"]["seed"]+1,log,"target")
        kernel = dict(source=sx,target=sy)
        entry,kernel = _verified_save(directory,directory/"kernels.pt",kernel,validate_kernel)
        register(directory,manifest,"kernels",entry)
    for name in ("source","target"):
        if kernel[name]["approximation"]["relative_rmse_max"] > c["kernel"]["error_warn_relative_rmse"]:
            warnings.warn(f"{name} per-image kernel approximation exceeds its error warning threshold; inspect kernel_checks.jsonl.")
    p,t = x.shape[1],y.shape[1]
    ec = c["estimator"]
    if "samples" not in manifest["files"]:
        samples = dict(training=objective.sample_pairs(p,t,seed=ec["seed"],per_row=ec["samples_per_row"],exact=ec["exact"],device=x.device),
                       audit=objective.sample_pairs(p,t,seed=ec["seed"]+1,count=ec["audit_samples"],device=x.device))
        entry,_ = _verified_save(directory,directory/"samples.pt",move(samples,"cpu"),
            lambda s: validate_indices(s,p,t,c))
        register(directory,manifest,"samples",entry)
    samples = torch.load(checked_file(directory,manifest["files"]["samples"]),map_location="cpu",weights_only=True)
    validate_indices(samples,p,t,c)
    samples = move(samples,x.device)
    path = directory/"pairs.jsonl"
    inventory = _read_pair_inventory(path,resume=resume,log=log)
    done = {(e["source_id"],e["target_id"]):e for e in inventory}
    if len(done) != len(inventory) or set(done)-selected:
        raise ValueError("Invalid selected-pair inventory.")
    summary = dict(pair_count=0,full_pair_count=len(x)*len(y),keep_mass=c["partial"]["keep_mass"],
                   max_mass_error=0.,max_capacity_error=0.,max_float32_objective_change=0.)
    for i,sid in enumerate(source.ids):
        for j,tid in enumerate(target.ids):
            if (sid,tid) not in selected:
                continue
            key = f"{i:06d}_{j:06d}_{digest([sid,tid])[:12]}"
            # Same persisted support across pairs (common random numbers). Cost
            # values/scales are train-derived and recomputed from immutable banks.
            pairs = objective.with_cost(samples["training"],x[i],y[j])
            audit = objective.with_cost(samples["audit"],x[i],y[j],scale=pairs["cost_scale"])
            entry,state = fit_unit(directory,directory/"plans/pairs"/f"{key}.pt",options,(p,t,c["transport_rank"]),
                identity,fx[i],fy[j],pairs,audit,log,resume=resume,registered=done.get((sid,tid)),compact_history=True,
                metadata=dict(source_id=sid,target_id=tid,source_index=i,target_index=j,cost_scale=pairs["cost_scale"],
                              objective_version="transported_kde_mi_mass_weighted_v1"))
            if (sid,tid) not in done:
                entry.update(source_id=sid,target_id=tid,source_index=i,target_index=j)
                with path.open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(entry)+"\n"); handle.flush(); os.fsync(handle.fileno())
                _sync_directory(directory)
                done[sid,tid] = entry
            _cleanup_latest(directory,entry,log)
            stored_audit = objective.evaluate(*(state[k].to(device=x.device,dtype=torch.float64) for k in ("q","r","g")),
                kernel["source"]["factors"][i].to(fx),kernel["target"]["factors"][j].to(fy),audit,
                lam=options["lam"],reg=options["reg"],partial=True,chunk_size=options["chunk_size"],log_floor=options["log_floor"])
            last = state["solver_report"]["history"][-1]
            delta = {k:stored_audit[k]-last["audit"][k] for k in ("objective","cost","mi","entropy","exact_mass")}
            constraints = state["storage_constraints"]
            summary["pair_count"] += 1
            summary["max_mass_error"] = max(summary["max_mass_error"],constraints["mass_error"])
            summary["max_capacity_error"] = max(summary["max_capacity_error"],constraints["plan_row_relative_max"],constraints["plan_column_relative_max"])
            summary["max_float32_objective_change"] = max(summary["max_float32_objective_change"],abs(delta["objective"]))
            log.append_jsonl("pair_diagnostics.jsonl",dict(source_id=sid,target_id=tid,last_iteration=last,
                storage_constraints=constraints,quantization=state["quantization"],float32_audit=stored_audit,float32_minus_float64_audit=delta))
            print(f"Low-rank partial pair {summary['pair_count']}/{len(selected)}: {sid} -> {tid}",flush=True)
    register(directory,manifest,"pairs",_file_entry(directory,path))
    manifest["pair_count"] = len(done)
    retained = [row["retained_probability"] for row in selection["rows"]]
    summary["retained_routing_probability"] = dict(min=min(retained),mean=sum(retained)/len(retained),max=max(retained))
    return dict(partial=summary,kernel_approximation={k:kernel[k]["approximation"] for k in ("source","target")},device=str(x.device),
        resources=manifest["resources"],
        limitation="Selected per-image-pair rank-constrained partial InfoOT, sampled objective and positive-kernel approximation. Numerical agreement is not evidence of image-quality improvement.")


def validate_indices(samples,n,m,c):
    # Reuse shared index/seed validation without persisting redundant pair costs.
    with_dummy_costs = {key:dict(pairs,cost=torch.ones(pairs["count"],dtype=torch.float64),cost_scale=1.)
                       for key,pairs in samples.items()}
    objective.validate_samples(with_dummy_costs,n,m,c)
