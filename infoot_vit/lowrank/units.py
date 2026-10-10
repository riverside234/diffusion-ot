"""One resumable constrained factor fit, shared by global and pair experiments."""
from pathlib import Path
import torch

from ..infoot_helper.fit_mapping import _verified_save
from ..infoot_helper.feature_bank import checked_file
from .storage import pack_factors,validate_factors,checkpoint_digest
from . import solver


def fit_unit(directory, path, config, shape, identity, fx, fy, pairs, audit, log, *,
             resume=False, registered=None, metadata=None, compact_history=False,verified_save=None):
    directory,path = Path(directory),Path(path)
    latest = path.with_suffix("")/"latest.pt"
    metadata = metadata or {}
    verified_save = verified_save or _verified_save

    def validate(state, checkpoint=False):
        report = validate_factors(state,config,shape)
        if state["fit_fingerprint"] != identity or any(state.get(k) != v for k,v in metadata.items()):
            raise ValueError("Factor fit identity mismatch.")
        if checkpoint:
            if state.get("checkpoint_sha256") != checkpoint_digest(state):
                raise ValueError("Checkpoint factor/metadata checksum mismatch.")
        elif state["solver_report"]["status"] != "converged_sampled_objective":
            raise ValueError("Nonconverged factor artifact.")
        return report

    if registered is not None:
        state = torch.load(checked_file(directory,registered),map_location="cpu",weights_only=True)
        validate(state)
        return registered,state

    def checkpoint(factors, step, record=None):
        state = pack_factors(factors,config,step=step,record=record,fit_fingerprint=identity,**metadata)
        state["checkpoint_sha256"] = checkpoint_digest(state)
        verified_save(directory,latest,state,lambda s:validate(s,True))

    start,last_record = 0,None
    if resume and latest.exists():
        state = torch.load(latest,map_location="cpu",weights_only=True)
        constraints = validate(state,True)
        loaded = tuple(state[k].to(device=fx.device,dtype=torch.float64) for k in ("q","r","g"))
        # An explicit optimization restart repair; stored/inference factors are
        # never silently renormalized. Report its magnitude and original errors.
        # Float32 may erase tiny entries: revive only at the double subnormal
        # floor for this logged solver restart, before constrained projection.
        positive = tuple(t.clamp_min(torch.finfo(t.dtype).tiny) for t in loaded)
        factors,_ = solver.project(*positive,config)
        start,last_record = state["step"],state.get("record")
        log.event("resume_float32_projection",unit=str(path.relative_to(directory)),step=start,
                  stored_constraints=constraints,factor_delta=solver.relative_change(loaded,factors))
    else:
        factors = solver.initialize(*shape,config,fx.device)
        checkpoint(factors,0)
    recovering = last_record and last_record["status"] == "converged_sampled_objective"
    if start >= config["max_steps"] and not recovering:
        raise ValueError("Checkpoint reached max_steps. Increase only optimizer.max_steps to continue this fit.")
    options = dict(config,max_steps=max(config["max_steps"],start+1) if recovering else config["max_steps"])
    unit = str(path.relative_to(directory))

    def on_step(factors,record):
        final = record["status"] != "max_steps" or record["step"] == options["max_steps"]
        if final or record["step"] % config.get("log_every",1) == 0:
            log.append_jsonl("iterations.jsonl",dict(unit=unit,**record))
        if final or record["step"] % config["checkpoint_every"] == 0:
            checkpoint(factors,record["step"],record)

    factors,report = solver.solve(factors,fx,fy,pairs,audit,options,start_step=start,on_step=on_step)
    if report["status"] != "converged_sampled_objective":
        raise RuntimeError(f"Low-rank InfoOT {report['status']}; diagnostics/checkpoint retained for {unit}.")
    if compact_history:
        report["history"] = report["history"][-1:]
    state = pack_factors(factors,config,solver_report=report,fit_fingerprint=identity,**metadata)
    entry,state = verified_save(directory,path,state,validate)
    return entry,state
