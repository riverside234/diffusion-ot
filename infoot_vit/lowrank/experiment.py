"""Immutable low-rank fit artifacts, resumable float32 checkpoints and logging."""
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import uuid
import warnings

import ot
import torch

from ..infoot_helper.feature_bank import FeatureBank, compatible_banks, digest, file_hash, checked_file, write_json, save_tensor
from ..infoot_helper.sampling import sample_ids, sampling_record
from ..infoot_helper.conditional import BalancedModel
from ..infoot_helper.storage import store_plan_state, quantize
from ..infoot_helper.fit_mapping import _verified_save, _cleanup_latest, _sync_directory
from ..infoot_helper.run_logging import RunLog
from .config import canonical, resources, MODE, SCHEMA,PARTIAL_MODE,PARTIAL_SCHEMA
from ..infoot_helper.device import resolve_device,move
from .units import fit_unit
from . import kernels, kernel_audit, objective, solver
from .storage import VERSION


def implementation():
    here = Path(__file__).parent
    shared = ("conditional", "partial", "infoot", "feature_bank", "sampling", "storage", "fit_mapping", "mapping", "device", "pair_selection")
    return {str(p.relative_to(here.parent)):file_hash(p) for p in sorted(here.glob("*.py"))} | {
        "POT_Dykstra":digest(inspect.getsource(solver._LR_Dysktra)),
        **{f"infoot_helper/{name}.py":file_hash(here.parent/f"infoot_helper/{name}.py") for name in shared}}


def inspect_fit(raw,root):
    c = canonical(raw); root = Path(root)
    banks, selections = [], []
    for key in ("source_bank","target_bank"):
        path = root/c[key]/"manifest.json"
        bank = json.loads(path.read_text(encoding="utf-8"))
        if digest({k:v for k,v in bank.items() if k != "artifact_id"}) != bank.get("artifact_id") or bank["split"] != "train":
            raise ValueError("Invalid training bank fingerprint/split.")
        ids = sample_ids(bank["ids"],**dict(count=c["sampling"]["images_per_domain"],seed=c["sampling"]["seed"]))
        banks.append(bank); selections.append(sampling_record(bank["ids"],ids,c["sampling"]["seed"]))
    if (banks[0]["representation"] != banks[1]["representation"] or banks[0]["domain"] == banks[1]["domain"]
            or set(banks[0]["ids"]) & set(banks[1]["ids"])):
        raise ValueError("Fit banks need identical representations and distinct domains/IDs.")
    rep = banks[0]["representation"]; p = rep["grid"][0]*rep["grid"][1]
    estimate = resources(len(selections[0]["ordered_ids"]),len(selections[1]["ordered_ids"]),p,rep["dim"],c)
    return dict(config=c,resources=estimate,sampling=dict(source=selections[0],target=selections[1]),
                bank_ids=[b["artifact_id"] for b in banks],representation=rep,
                runtime=dict(torch=str(torch.__version__),pot=ot.__version__),implementation=implementation())


def compact_inspection(report):
    return dict(report,sampling={key:dict(count=v["selected_count"],seed=v["seed"],ordered_ids_sha256=digest(v["ordered_ids"]))
                                for key,v in report["sampling"].items()})


def fingerprint(report):
    # Iteration budgets can change on explicit resume; objectives cannot.
    c = dict(report["config"],optimizer={k:v for k,v in report["config"]["optimizer"].items() if k != "max_steps"},
             image_solver={k:v for k,v in report["config"]["image_solver"].items() if k != "max_outer_steps"})
    return digest(dict(report,config=c,resources=None))


def register(directory,manifest,key,entry):
    manifest["files"][key] = entry
    write_json(directory/"manifest.json",manifest)
    with (directory/"manifest.json").open("r+b") as handle: os.fsync(handle.fileno())
    _sync_directory(directory)


def validate_kernels(state,shape,dimension=None,kernel_config=None,projection_multiplier=1.):
    if state.get("storage_version") != VERSION:
        raise ValueError("Kernel storage version mismatch.")
    for key,n in (("fx",shape[0]),("fy",shape[1])):
        f = state[key]
        method = state["source" if key == "fx" else "target"]["method"]
        if (f.shape != (n,shape[2]) or f.dtype != torch.float32 or not torch.isfinite(f).all()
                or (f < 0).any() or (f.double().sum(1) <= 0).any()
                or (method == kernels.LEGACY and (f.double().square().sum(1)-1).abs().max() > 2e-6)):
            raise ValueError("Invalid float32 kernel factors.")
    for key in ("source", "target"):
        s = state[key]
        dim = len(s["mean"]) if dimension is None else dimension
        if (s["method"] not in kernels.METHODS or s["rank"] != shape[2]
                or s["mean"].shape != (dim,) or s["omega"].shape != (dim, shape[2])
                or any(t.dtype != torch.float64 or not torch.isfinite(t).all() for t in (s["mean"], s["omega"]))
                or not all(isinstance(s[k], (int, float)) and 0 < s[k] < float("inf") for k in ("scale", "h", "sigma"))
                or s["sigma"] != s["h"]*s["scale"]):
            raise ValueError("Invalid saved kernel parameters/bandwidth.")
        if s["method"] != kernels.LEGACY:
            expected_a = kernels.oprf_coefficient(dim,s["h"]) if s["method"] == kernels.OPRF else 0.
            if (type(s.get("orthogonal")) is not bool or not isinstance(s.get("a"),(int,float))
                    or not math.isclose(s["a"],expected_a,rel_tol=1e-12,abs_tol=1e-14)):
                raise ValueError("Invalid saved positive-kernel scale compensation.")
        if kernel_config is not None and (s["method"] != kernel_config.get("method",kernels.LEGACY)
                or s.get("orthogonal",False) != kernel_config.get("orthogonal",False)):
            raise ValueError("Saved kernel method differs from the fitted configuration.")
    if kernel_config is not None and kernel_config.get("error_policy") == "error":
        checks = state.get("approximation",{})
        if set(checks) != {"source","target"} or not all("float32_roundtrip" in v for v in checks.values()):
            raise ValueError("Strict kernel artifact lacks its raw/storage acceptance audits.")
        if not kernel_audit.acceptance(checks,kernel_config)["accepted"]:
            raise ValueError("Saved kernel artifact failed the configured accuracy limits.")
    if projection_multiplier != 1:
        projection = state.get("projection",{})
        checks = projection.get("approximation",{})
        if (projection.get("bandwidth_multiplier") != projection_multiplier
                or kernel_config is None or set(checks) != {"source","target"}
                or not all("float32_roundtrip" in v for v in checks.values())
                or any(projection.get(f"{key}_h") != state[key]["h"]*projection_multiplier
                       for key in ("source","target"))
                or not kernel_audit.acceptance(checks,kernel_config)["accepted"]):
            raise ValueError("Missing, mismatched or rejected projection-bandwidth kernel audit.")


def _balanced_kernels(directory,manifest,x,y,log):
    c = manifest["config"]; kc = c["kernel"]; chunk = c["optimizer"]["chunk_size"]
    multiplier = c["projection"]["bandwidth_multiplier"]
    flatx,flaty = x.flatten(0,1),y.flatten(0,1)
    shape = (len(flatx),len(flaty),c["kernel_rank"])
    print(f"Kernel audit: method={kc['method']}, rank={c['kernel_rank']}, policy={kc['error_policy']}",flush=True)
    if "kernels" in manifest["files"]:
        state = torch.load(checked_file(directory,manifest["files"]["kernels"]),weights_only=True)
        validate_kernels(state,shape,flatx.shape[1],kc,multiplier)
        fx = kernels.features(flatx,state["source"],chunk)
        fy = kernels.features(flaty,state["target"],chunk)
        if not torch.equal(fx.float(),state["fx"].to(x.device)) or not torch.equal(fy.float(),state["fy"].to(y.device)):
            raise ValueError("Recomputed float64 kernel factors differ from the saved snapshot.")
        checks,reference = state["approximation"],state.get("reference")
    else:
        fx,sx = kernels.fit_features(flatx,c["kernel_rank"],kc["h"],kc["seed"],chunk,
                                    method=kc["method"],orthogonal=kc["orthogonal"])
        fy,sy = kernels.fit_features(flaty,c["kernel_rank"],kc["h"],kc["seed"]+1,chunk,
                                    method=kc["method"],orthogonal=kc["orthogonal"])
        checks = {}
        for i,(name,z,f,s) in enumerate((("source",flatx,fx,sx),("target",flaty,fy,sy))):
            args = dict(seed=kc["seed"]+10+i,count=kc["check_pairs"],density_queries=kc["density_queries"],chunk_size=chunk)
            checks[name] = kernels.error_report(z,f,s,**args)
            checks[name]["float32_roundtrip"] = kernels.error_report(z,f,s,storage_roundtrip=True,**args)
            print(f"{name}: kernel relative RMSE={checks[name]['relative_rmse']:.4g}, "
                  f"mean density relative error={checks[name]['density_relative_error_mean']:.4g}",flush=True)
        reference = None
    quality = kernel_audit.acceptance(checks,kc)
    # Persist acceptance before the optional reference, including if that audit fails.
    for filename,report in (("kernel_approximation.json",checks),("kernel_quality.json",quality)):
        write_json(directory/filename,report); log.write_json(filename,report)
    if "kernels" not in manifest["files"] and kc["reference_images"]:
        reference = kernel_audit.reference_report(x,y,fx,fy,sx,sy,kc,c["optimizer"])
    if reference is not None:
        write_json(directory/"kernel_reference.json",reference); log.write_json("kernel_reference.json",reference)
    log.event("kernel_accuracy_checked",accepted=quality["accepted"],policy=quality["policy"],failures=quality["failures"])
    if not quality["accepted"] and kc["error_policy"] == "error":
        raise RuntimeError("Kernel accuracy acceptance failed; transport fitting has not started. "
                           "See kernel_quality.json, kernel_approximation.json and kernel_reference.json. "
                           "Compare methods/ranks in a fresh --kernel-check-only run.")
    for name,check in checks.items():
        if check["relative_rmse"] > kc["error_warn_relative_rmse"]:
            warnings.warn(f"{name} positive-kernel relative RMSE={check['relative_rmse']:.3g}; inspect approximation before claiming quality.")
    projection = None
    if multiplier != 1:
        if "kernels" in manifest["files"]:
            projection = state["projection"]
        else:
            projection_checks = {}
            for i,(name,z,s) in enumerate((("source",flatx,sx),("target",flaty,sy))):
                ps = kernels.projection_state(s,multiplier)
                pf = kernels.features(z,ps,chunk)
                args = dict(seed=kc["seed"]+20+i,count=kc["check_pairs"],density_queries=kc["density_queries"],chunk_size=chunk)
                projection_checks[name] = kernels.error_report(z,pf,ps,**args)
                projection_checks[name]["float32_roundtrip"] = kernels.error_report(z,pf,ps,storage_roundtrip=True,**args)
                del pf
            projection = dict(bandwidth_multiplier=multiplier,source_h=sx["h"]*multiplier,
                target_h=sy["h"]*multiplier,approximation=projection_checks,
                quality=kernel_audit.acceptance(projection_checks,dict(kc,error_policy="error")))
        write_json(directory/"projection_kernel_quality.json",projection)
        log.write_json("projection_kernel_quality.json",projection)
        log.event("projection_kernel_accuracy_checked",**projection["quality"])
        if not projection["quality"]["accepted"]:
            raise RuntimeError("Projection kernel accuracy acceptance failed; transport fitting has not started. "
                               "See projection_kernel_quality.json. Narrow projection kernels need independent validation.")
    if "kernels" not in manifest["files"]:
        qfx,ex = quantize(fx); qfy,ey = quantize(fy)
        state = dict(fx=qfx,fy=qfy,source=move(sx,"cpu"),target=move(sy,"cpu"),storage_version=VERSION,
                     quantization=dict(fx=ex,fy=ey),approximation=checks,quality=quality,reference=reference)
        if projection is not None:
            state["projection"] = projection
        entry,_ = _verified_save(directory,directory/"kernels.pt",state,
                                  lambda s:validate_kernels(s,shape,flatx.shape[1],kc,multiplier))
        register(directory,manifest,"kernels",entry)
    return fx,fy,checks,reference,quality


def fit(raw,*,root,output_root=None,resume=None,kernel_check_only=False):
    root = Path(root).resolve()
    mode = canonical(raw)["mode"]
    if kernel_check_only and mode != MODE:
        raise ValueError("--kernel-check-only currently supports grouped_patch_lowrank only.")
    schema = PARTIAL_SCHEMA if mode == PARTIAL_MODE else SCHEMA
    directory = Path(resume).resolve() if resume else (Path(output_root) if output_root else root/"outputs/infoot_vit")/f"{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True,exist_ok=bool(resume))
    manifest = None
    with RunLog(directory,"lowrank_grouped_fit",dict(requested=raw,resume=str(resume))) as log:
        try:
            report = inspect_fit(raw,root); c = report["config"]; identity = fingerprint(report)
            log.write_json("inspection.json",report); print(json.dumps(compact_inspection(report),indent=2),flush=True)
            for estimate,limit in ((report["resources"]["estimated_final_artifact_bytes"],c["resources"]["max_artifact_bytes"]),
                                   (report["resources"]["estimated_working_gib"],c["resources"]["max_working_gib"])):
                if estimate >= limit: raise MemoryError(f"Resource estimate {estimate} exceeds budget {limit}.")
            device = resolve_device(c["device"])
            torch.set_num_threads(c["threads"])
            if resume:
                # Preflight and completed-fit verification are read-only. Only
                # adopt an unfinished manifest after accepting this resume, so
                # a rejected request cannot mark an existing artifact failed.
                saved_manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
                if saved_manifest["schema"] != schema or saved_manifest["fit_fingerprint"] != identity:
                    raise ValueError("Low-rank resume fingerprint changed; use a new fit directory.")
                if saved_manifest["status"] == "complete":
                    from .mapping import LowRankMapper
                    completed = LowRankMapper.load(directory)
                    if mode == PARTIAL_MODE:
                        for entry in completed.pairs.values():
                            _cleanup_latest(directory,entry,log)
                    else:
                        _cleanup_latest(directory,saved_manifest["files"]["factors"],log)
                    _cleanup_latest(directory,saved_manifest["files"]["image"],log)
                    return directory
                manifest = saved_manifest
                manifest.update(config=c,status="fitting"); manifest.pop("error",None)
            source,target = [FeatureBank.load(root/c[key]) for key in ("source_bank","target_bank")]
            compatible_banks(source,target)
            source = source.subset(report["sampling"]["source"]["ordered_ids"])
            target = target.subset(report["sampling"]["target"]["ordered_ids"])
            if manifest is None:
                manifest = dict(schema=schema,status="fitting",fit_fingerprint=identity,config=c,files={},
                    source_bank=dict(path=os.path.relpath(source.path,directory),artifact_id=source.artifact_id),
                    target_bank=dict(path=os.path.relpath(target.path,directory),artifact_id=target.artifact_id),
                    source_ids=source.ids,target_ids=target.ids,source_domain=source.manifest["domain"],
                    target_domain=target.manifest["domain"],representation=source.representation,
                    sampling=report["sampling"],resources=report["resources"],runtime=report["runtime"],
                    implementation=report["implementation"],precision=dict(fit="float64",factors="float32",output="query dtype/device"))
            write_json(directory/"manifest.json",manifest)
            x,y = source.features.to(device=device,dtype=torch.float64),target.features.to(device=device,dtype=torch.float64)
            n,m = len(x)*x.shape[1],len(y)*y.shape[1]
            flatx,flaty = x.reshape(n,-1),y.reshape(m,-1)
            shape = (n,m,c["transport_rank"])
            kernel_result = None
            if mode == MODE and (kernel_check_only or c["kernel"]["error_policy"] == "error" or c["kernel"]["reference_images"]
                                 or c["projection"]["bandwidth_multiplier"] != 1):
                kernel_result = _balanced_kernels(directory,manifest,x,y,log)
                if kernel_check_only:
                    manifest["status"] = "kernel_checked"
                    write_json(directory/"manifest.json",manifest)
                    print("Kernel-only diagnostics saved; resume this directory with identical settings for the full fit.",flush=True)
                    return directory
            # Existing balanced image router: unchanged whole-image geometry/MI.
            if "image" not in manifest["files"]:
                def image_step(plan,record):
                    save_tensor(directory/"plans/image/latest.pt",store_plan_state(dict(plan=plan.cpu(),report=record)))
                    log.append_jsonl("image_iterations.jsonl",record)
                    if record["iteration"] == 1 or record["iteration"] % 25 == 0 or record["status"] != "running":
                        print(f"Image router {record['iteration']}/{c['image_solver']['max_outer_steps']}: "
                              f"objective={record['objective']:.8g}, cost={record['cost']:.8g}, "
                              f"mi_term={record['mi_term']:.8g}, entropy_term={record['entropy_term']:.8g}, "
                              f"lam={c['image_solver']['lam']:.6g}, reg={c['image_solver']['reg']:.6g}, "
                              f"delta={record['plan_delta_l1']:.3g}, "
                              f"effective_targets={record['mean_row_effective_targets']:.1f}, status={record['status']}",flush=True)
                image = BalancedModel.fit(x.reshape(len(x),-1),y.reshape(len(y),-1),c["image_solver"],on_step=image_step)
                image_report = {key:value for key,value in image.state.items() if key != "plan"}
                write_json(directory/"image_report.json",image_report)
                log.write_json("image_report.json",image_report)
                if image.state["status"] != "converged":
                    last = image.state["history"][-1]
                    raise RuntimeError(f"Image router {image.state['status']} after {last['iteration']} outer iterations: "
                        f"plan_delta_l1={last['plan_delta_l1']:.3g}, tolerance={c['image_solver']['outer_tolerance']:.3g}, "
                        f"inner_error={last['inner']['error']:.3g}. Patch fitting has not started. "
                        "See image_report.json and logs/*/image_iterations.jsonl. For an outer-budget failure, "
                        "increase --image-max-steps (the unregistered router restarts); changing h/lam/reg needs a fresh fit.")
                entry,_ = _verified_save(directory,directory/"plans/image.pt",store_plan_state(image.state),
                    lambda s:BalancedModel(x.reshape(len(x),-1),y.reshape(len(y),-1),s))
                register(directory,manifest,"image",entry); _cleanup_latest(directory,entry,log)
            else:
                saved_image = torch.load(checked_file(directory,manifest["files"]["image"]),weights_only=True)
                if saved_image["status"] != "converged": raise ValueError("Invalid registered image router.")
                image = BalancedModel(x.reshape(len(x),-1),y.reshape(len(y),-1),saved_image)
                _cleanup_latest(directory,manifest["files"]["image"],log)
            # Pair selection must use the SAME quantized router on fresh/resumed runs.
            image = BalancedModel(x.reshape(len(x),-1),y.reshape(len(y),-1),
                torch.load(checked_file(directory,manifest["files"]["image"]),map_location="cpu",weights_only=True))
            if mode == PARTIAL_MODE:
                from .partial_experiment import fit_pairs
                fit_report = fit_pairs(directory,manifest,source,target,x,y,image,log,resume=bool(resume))
                write_json(directory/"fit_report.json",fit_report); log.write_json("fit_report.json",fit_report)
                total_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                if total_bytes+1024*1024 >= c["resources"]["max_artifact_bytes"]:
                    raise MemoryError("Final partial fit artifacts/logs exceed the configured budget.")
                manifest.update(status="complete",artifact_bytes_before_report=total_bytes)
                manifest["artifact_id"] = digest({k:v for k,v in manifest.items() if k != "artifact_id"})
                write_json(directory/"manifest.json",manifest)
                return directory
            fx,fy,checks,reference,quality = kernel_result or _balanced_kernels(directory,manifest,x,y,log)
            if "samples" not in manifest["files"]:
                ec = c["estimator"]
                pairs = objective.with_cost(objective.sample_pairs(n,m,seed=ec["seed"],per_row=ec["samples_per_row"],exact=ec["exact"],device=device),flatx,flaty)
                audit = objective.with_cost(objective.sample_pairs(n,m,seed=ec["seed"]+1,count=ec["audit_samples"],device=device),flatx,flaty,scale=pairs["cost_scale"])
                entry,_ = _verified_save(directory,directory/"samples.pt",move(dict(training=pairs,audit=audit),"cpu"),
                    lambda s:objective.validate_samples(s,n,m,c))
                register(directory,manifest,"samples",entry)
            else:
                samples = torch.load(checked_file(directory,manifest["files"]["samples"]),weights_only=True)
                objective.validate_samples(samples,n,m,c)
                pairs,audit = move(samples["training"],device),move(samples["audit"],device)
            entry,state = fit_unit(directory,directory/"factors.pt",c["optimizer"],shape,identity,
                fx,fy,pairs,audit,log,resume=bool(resume),registered=manifest["files"].get("factors"),
                verified_save=_verified_save)
            if "factors" not in manifest["files"]:
                register(directory,manifest,"factors",entry)
            final_report = state["solver_report"]
            _cleanup_latest(directory,manifest["files"]["factors"],log)
            # Same independent audit pairs before/after float32 storage; the
            # raw audit was saved at the converged float64 iteration.
            disk_kernel = torch.load(checked_file(directory,manifest["files"]["kernels"]),weights_only=True)
            stored_audit = objective.evaluate(
                *(state[key].to(device=device,dtype=torch.float64) for key in ("q","r","g")),
                disk_kernel["fx"].to(device=device,dtype=torch.float64),disk_kernel["fy"].to(device=device,dtype=torch.float64),audit,
                lam=c["optimizer"]["lam"],reg=c["optimizer"]["reg"],log_floor=c["optimizer"]["log_floor"],
                chunk_size=c["optimizer"]["chunk_size"])
            raw_audit = final_report["history"][-1]["audit"]
            quantized_delta = {key:stored_audit[key]-raw_audit[key] for key in ("objective","cost","mi","entropy","exact_mass")}
            image_state = torch.load(checked_file(directory,manifest["files"]["image"]),weights_only=True)
            rows = image_state["plan"].to(device=device,dtype=torch.float64)
            rows = rows/rows.sum(1,keepdim=True)  # Diagnostic conditional probabilities only.
            image_entropy = -(rows*rows.clamp_min(1e-300).log()).sum(1)
            image_diagnostics = dict(mean_row_entropy=float(image_entropy.mean()),
                mean_effective_targets=float(image_entropy.exp().mean()),mean_max_weight=float(rows.max(1).values.mean()),
                row_residual=image_state["row_residual"],column_residual=image_state["column_residual"])
            total_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            if total_bytes >= c["resources"]["max_artifact_bytes"]:
                raise MemoryError(f"Saved fit directory is {total_bytes} bytes, exceeding the configured artifact budget.")
            manifest.update(status="complete",artifact_bytes_before_report=total_bytes)
            manifest["artifact_id"] = digest({k:v for k,v in manifest.items() if k != "artifact_id"})
            fit_report = dict(resources=report["resources"],kernel_approximation=checks,solver=final_report,device=str(device),
                kernel_quality=quality,kernel_reference=reference,
                image_router=image_diagnostics,
                factor_quantization=state["quantization"],storage_constraints=state["storage_constraints"],
                float32_audit=stored_audit,float32_minus_float64_audit=quantized_delta,
                cost_scale=pairs["cost_scale"],estimator={k:v for k,v in pairs.items() if k not in {"i","j","cost"}},
                limitation="Convergence of a fixed sampled, positive-kernel, rank-constrained objective; not dense InfoOT or demonstrated image quality.")
            write_json(directory/"fit_report.json",fit_report)
            log.write_json("fit_report.json",fit_report)
            total_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            if total_bytes+1024*1024 >= c["resources"]["max_artifact_bytes"]:
                raise MemoryError("Final report/diagnostics exceed the artifact budget (including 1 MiB completion reserve).")
            write_json(directory/"manifest.json",manifest)
        except BaseException as exc:
            if manifest is not None:
                manifest.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
                manifest.pop("artifact_id",None); write_json(directory/"manifest.json",manifest)
            raise
    return directory
