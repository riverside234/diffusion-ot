"""Factor-space partial InfoOT references; no image-quality claims."""
from copy import deepcopy
import json

import pytest
import torch

from test_infoot_vit_mapping import banks,cpu_threads,mapped,config as dense_config
from test_infoot_vit_lowrank import tiny_config,problem
from infoot_vit.lowrank import objective,solver,kernels
from infoot_vit.lowrank.config import canonical,optimizer_config,resources,PARTIAL_MODE
from infoot_vit.lowrank.experiment import fit
from infoot_vit.lowrank.storage import pack_factors,validate_factors
from infoot_vit.lowrank.projection import partial_project
from infoot_vit.lowrank.pair_kernels import support_threshold
from infoot_vit.infoot_helper.partial import information
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.conditional import normalize_rows,BalancedModel
from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.fit_mapping import fit_mapping


def partial_config(mass=.7,k=1):
    c = tiny_config()
    c.update(mode=PARTIAL_MODE,partial=dict(keep_mass=mass),fit_pair_top_k=k,
             projection=dict(confidence=dict(support_calibration="disabled"),patch_chunk_size=2))
    return c


@pytest.mark.parametrize("mass",[.2,.8,1.])
def test_partial_factor_capacities_mass_and_kl_projection(mass):
    c = optimizer_config(canonical(partial_config(mass)))
    factors = solver.initialize(6,7,3,c,"cpu")
    q,r,g = factors
    plan = (q/g)@r.T
    assert plan.sum() == pytest.approx(mass,abs=1e-10)
    assert (plan.sum(1) <= 1/6+1e-10).all()
    assert (plan.sum(0) <= 1/7+1e-10).all()
    torch.testing.assert_close(q.sum(0),g,atol=1e-11,rtol=1e-11)
    again,_ = solver.project(*factors,c)
    for a,b in zip(again,factors): torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    state = pack_factors(factors,c)
    validate_factors(state,c,(6,7,3))
    assert state["storage_constraints"]["mass_error"] < 1e-7
    assert state["q"].dtype == torch.float32
    bad = deepcopy(state); bad["g"] *= 1.1
    with pytest.raises(ValueError,match="marginals|constraints"):
        validate_factors(bad,c)


def test_partial_factor_projection_matches_independent_convex_kl_reference():
    import numpy as np
    from scipy.optimize import minimize,LinearConstraint,Bounds
    c = optimizer_config(canonical(partial_config(.55)))
    c["min_g"] = .05
    rng = torch.Generator().manual_seed(31)
    q,r,g = [torch.rand(*shape,generator=rng,dtype=torch.float64)+.2 for shape in ((3,2),(4,2),(2,))]
    result,_ = solver.project(q,r,g,c)
    original = torch.cat([x.flatten() for x in (q,r,g)]).numpy()
    got = torch.cat([x.flatten() for x in result]).numpy()
    # Independent linear constraints on the concatenated factors, not Gamma.
    eq = np.zeros((5,16)); eq[-1,-2:] = 1
    cap = np.zeros((7,16))
    for k in range(2):
        eq[k,k:6:2] = 1; eq[k,14+k] = -1
        eq[2+k,6+k:14:2] = 1; eq[2+k,14+k] = -1
    for i in range(3): cap[i,2*i:2*i+2] = 1
    for i in range(4): cap[3+i,6+2*i:8+2*i] = 1
    limits = np.r_[np.full(3,1/3),np.full(4,1/4)]
    target = np.r_[np.zeros(4),.55]
    solved = minimize(lambda z: np.sum(z*np.log(z/original)-z+original),got,
        jac=lambda z: np.log(z/original),method="SLSQP",bounds=Bounds(np.r_[np.full(14,1e-15),.05,.05],np.inf),
        constraints=[LinearConstraint(eq,target,target),LinearConstraint(cap,-np.inf,limits)],
        options=dict(ftol=1e-12,maxiter=1000))
    assert solved.success,solved.message
    np.testing.assert_allclose(got,solved.x,rtol=3e-6,atol=3e-8)


@pytest.mark.parametrize("mass",[.3,1.])
@pytest.mark.parametrize("chunk",[1,13,100])
def test_partial_mi_cost_entropy_and_gradients_match_dense(mass,chunk):
    x,y,fx,fy,_,_,_,_,pairs = problem()
    c = optimizer_config(canonical(partial_config(mass)))
    factors = solver.initialize(len(x),len(y),2,c,"cpu")
    got,gradients = objective.evaluate(*factors,fx,fy,pairs,lam=.3,reg=.05,partial=True,gradient=True,chunk_size=chunk)
    q,r,g = [t.clone().requires_grad_() for t in factors]
    plan = (q/g)@r.T
    cost = (plan*torch.cdist(x,y)).sum()
    mi = information(plan,fx@fx.T,fy@fy.T)
    entropy = (plan*(plan.log()-1)).sum()
    loss = cost-.3*mi+.05*entropy
    for key,value in (("cost",cost),("mi",mi),("entropy",entropy),("objective",loss)):
        assert got[key] == pytest.approx(float(value.detach()),abs=2e-12)
    for a,b in zip(gradients,torch.autograd.grad(loss,(q,r,g))):
        torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)


def test_partial_projection_agrees_with_dense_approximate_kernel_and_rejection():
    x,y,fx,fy,sx,_,_,_,_ = problem()
    c = optimizer_config(canonical(partial_config(.7)))
    q,r,g = solver.initialize(len(x),len(y),2,c,"cpu")
    fq = kernels.features(x*.95+.01,sx)
    support = dict(threshold=None)
    mapped,gamma,diag = partial_project(fq,fx,fy,y,(q,r,g),support,2)
    plan = (q/g)@r.T; kquery = fq@fx.T; ky = fy@fy.T
    scores = (kquery@plan@ky.T)/ky.mean(0)
    weights = normalize_rows(scores)
    torch.testing.assert_close(mapped,weights@y,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(gamma,(kquery@plan.sum(1))/kquery.mean(1),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(diag["patch_entropy"],-(weights*weights.log()).sum(1),atol=1e-12,rtol=1e-12)
    changed,_,_ = partial_project(fq,fx,fy,y,(q,r,g),support,1)
    torch.testing.assert_close(changed,mapped,atol=1e-12,rtol=1e-12)


def test_partial_fit_storage_sparse_mapping_resume_and_corrupt_pair(tmp_path,banks,monkeypatch):
    directory = fit(partial_config(),root=tmp_path)
    mapper = FeatureMapper.load(directory,device="cpu")
    assert mapper.mode == PARTIAL_MODE and len(mapper.pairs) == 2
    assert not list(directory.rglob("latest.pt"))
    assert mapper.fx.dtype == torch.float64 and mapper.fx.device.type == "cpu"
    monkeypatch.setattr(solver,"solve",lambda *a,**k:pytest.fail("Inference/resume refit"))
    monkeypatch.setattr(kernels,"fit_features",lambda *a,**k:pytest.fail("Inference kernel fit"))
    a = mapped(mapper,banks[2],return_metadata=True,chunk_size=1)
    b = mapped(FeatureMapper.load(directory),banks[2],return_metadata=True,chunk_size=2)
    torch.testing.assert_close(a.mapped_features,b.mapped_features,atol=1e-12,rtol=0)
    torch.testing.assert_close(a.match_confidence,b.match_confidence,atol=1e-12,rtol=0)
    assert a.valid_mask.all() and (a.match_confidence < 1).all()
    for record in a.diagnostics["queries"]:
        assert record["fit_pair_discarded_routing_mass"] > 0
        assert record["matched_rejected_accounting_error"] < 1e-10
    assert fit(partial_config(),root=tmp_path,resume=directory) == directory
    entry = next(iter(mapper.pairs.values()))
    (directory/entry["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="hash|SHA|fingerprint|checksum|changed artifact"):
        FeatureMapper.load(directory)


def test_partial_full_pair_equivalence_when_k_covers_all(tmp_path,banks):
    a = FeatureMapper.load(fit(partial_config(k=None),root=tmp_path))
    b = FeatureMapper.load(fit(partial_config(k=200),root=tmp_path))
    ra,rb = [mapped(m,banks[2],return_metadata=True) for m in (a,b)]
    assert len(a.pairs) == len(b.pairs) == 4
    torch.testing.assert_close(ra.mapped_features,rb.mapped_features,atol=1e-12,rtol=0)
    assert ra.diagnostics["queries"][0]["fit_pair_discarded_routing_mass"] < 1e-10


def test_partial_no_full_patch_matrix_in_objective_or_projection():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    x,y,fx,fy,sx,_,_,_,pairs = problem()
    c = optimizer_config(canonical(partial_config()))
    q,r,g = solver.initialize(6,7,2,c,"cpu")
    fq = kernels.features(x,sx)
    class NoMatrices(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out = func(*args,**(kwargs or {}))
            for v in tree_flatten(out)[0]:
                if isinstance(v,torch.Tensor):
                    assert v.shape not in {(6,6),(7,7),(6,7),(7,6)},(func,v.shape)
            return out
    with NoMatrices():
        objective.evaluate(q,r,g,fx,fy,pairs,lam=.3,reg=.05,partial=True,gradient=True,chunk_size=4)
        partial_project(fq,fx,fy,y,(q,r,g),dict(threshold=None),2)


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="Lab GPU required"))])
@pytest.mark.parametrize("mode",["grouped_patch_lowrank",PARTIAL_MODE])
def test_numerical_device_propagates_to_router_kernels_solver_and_mapping(tmp_path,banks,device,mode,monkeypatch):
    c = partial_config() if mode == PARTIAL_MODE else tiny_config()
    c["device"] = device
    evaluate = objective.evaluate
    observed_gradients = []
    def spy(*args,**kwargs):
        assert all(t.device.type == device for t in args[:5])
        assert all(args[5][k].device.type == device for k in ("i","j","cost"))
        result = evaluate(*args,**kwargs)
        if kwargs.get("gradient"):
            assert all(t.device.type == device and torch.isfinite(t).all() for t in result[1])
            observed_gradients.append(True)
        return result
    monkeypatch.setattr(objective,"evaluate",spy)
    monkeypatch.setattr(solver,"evaluate",spy)
    directory = fit(c,root=tmp_path)
    mapper = FeatureMapper.load(directory,device=device)
    assert observed_gradients
    assert all(t.device.type == device for t in (mapper.x,mapper.y,mapper.image.plan,mapper.image.ky))
    result = mapped(mapper,banks[2],return_metadata=True)
    assert result.mapped_features.device == banks[2].features.device  # Public output contract.
    if mode == PARTIAL_MODE:
        assert mapper.fx.device.type == mapper.fy.device.type == device
        for factors,_ in mapper._factor_cache.values():
            assert all(t.device.type == device for t in factors)
    else:
        assert mapper.cross.device.type == mapper.target_kernel_features.device.type == device
    cfg = dense_config("grouped_partial"); cfg["device"] = device
    dense = FeatureMapper.load(fit_mapping(cfg,root=tmp_path))
    assert dense.image.plan.device.type == dense.shared["kx"].device.type == device
    assert torch.isfinite(mapped(dense,banks[2],return_metadata=True).mapped_features).all()


def test_cuda_default_clear_failure_and_partial_resource_budget(monkeypatch):
    assert canonical({})["device"] == canonical(dict(mode=PARTIAL_MODE))["device"] == "cuda"
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA requested but unavailable"):
        resolve_device("cuda")
    assert resolve_device("cpu").type == "cpu"
    c = canonical(dict(mode=PARTIAL_MODE))
    assert resources(2000,2000,196,768,c)["estimated_final_artifact_bytes"] < 4_000_000_000


def test_loo_kernel_factor_support_matches_dense():
    _,_,fx,_,_,_,_,_,_ = problem()
    confidence = canonical(partial_config())["projection"]["confidence"]
    confidence.update(support_calibration="fit_leave_one_out_log_density",support_quantile=.25)
    support = support_threshold(fx,confidence)
    dense = fx@fx.T; dense.fill_diagonal_(0)
    assert support["threshold"] == pytest.approx(float(torch.quantile((dense.sum(1)/(len(fx)-1)).log(),.25)),abs=1e-12)


def test_partial_descent_checks_caps_and_decreases_same_sampled_objective():
    x,y,fx,fy,_,_,_,_,pairs = problem()
    c = optimizer_config(canonical(partial_config()))
    c.update(max_steps=10,stationarity_tolerance=1e-12)
    factors = solver.initialize(len(x),len(y),2,c,"cpu")
    final,report = solver.solve(factors,fx,fy,pairs,pairs,c)
    history = report["history"]
    assert len(history) > 1 and history[-1]["objective"] < history[0]["objective"]
    assert all(b["objective"] <= a["objective"]+1e-12 for a,b in zip(history,history[1:]))
    solver.check(*final,c["constraint_tolerance"],c["min_g"],**solver.constraint_args(c))


@pytest.mark.parametrize("where",["verify","registered"])
def test_partial_interrupted_final_save_resume_cleanup(tmp_path,banks,monkeypatch,where):
    from infoot_vit.lowrank import units,partial_experiment
    c = partial_config(k=None); c["optimizer"]["max_steps"] = 1
    verify,cleanup = units._verified_save,partial_experiment._cleanup_latest
    def fail_verify(directory,path,state,validator):
        if path.parent.name == "pairs":
            raise OSError("simulated pair verification failure")
        return verify(directory,path,state,validator)
    def fail_cleanup(directory,entry,log):
        if entry["file"].startswith("plans/pairs/") or entry["file"].startswith("plans\\pairs\\"):
            raise KeyboardInterrupt("after pair registration")
        return cleanup(directory,entry,log)
    monkeypatch.setattr(units,"_verified_save",fail_verify if where == "verify" else verify)
    monkeypatch.setattr(partial_experiment,"_cleanup_latest",fail_cleanup if where == "registered" else cleanup)
    with pytest.raises(OSError if where == "verify" else KeyboardInterrupt):
        fit(c,root=tmp_path)
    directory = next((tmp_path/"outputs/infoot_vit").iterdir())
    assert list(directory.rglob("latest.pt"))
    monkeypatch.setattr(units,"_verified_save",verify)
    monkeypatch.setattr(partial_experiment,"_cleanup_latest",cleanup)
    monkeypatch.setattr(BalancedModel,"fit",lambda *a,**k:pytest.fail("Refitted completed router"))
    monkeypatch.setattr(kernels,"fit_features",lambda *a,**k:pytest.fail("Refitted saved kernel"))
    fit(c,root=tmp_path,resume=directory)
    assert not list(directory.rglob("latest.pt"))
    mapper = FeatureMapper.load(directory)
    assert len(mapper.pairs) == 4
    assert torch.isfinite(mapped(mapper,banks[2],return_metadata=True).mapped_features).all()


def test_failed_partial_fit_resume_preserves_checkpoint_and_rejects_tampering(tmp_path,banks):
    c = partial_config(); c["optimizer"].update(max_steps=1,stationarity_tolerance=1e-15)
    with pytest.raises(RuntimeError,match="max_steps"):
        fit(c,root=tmp_path)
    directory = next((tmp_path/"outputs/infoot_vit").iterdir())
    checkpoint = next(directory.rglob("latest.pt"))
    assert torch.load(checkpoint,weights_only=True)["step"] == 1
    c["optimizer"]["max_steps"] = 2
    with pytest.raises(RuntimeError,match="max_steps"):
        fit(c,root=tmp_path,resume=directory)
    state = torch.load(checkpoint,weights_only=True)
    assert state["step"] == 2
    state["step"] = 0; torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="checksum"):
        fit(c,root=tmp_path,resume=directory)


def test_all_offline_experiment_yamls_select_cuda():
    from pathlib import Path
    import yaml
    from infoot_vit.infoot_helper.fit_mapping import validate_config
    for path in (Path(__file__).resolve().parents[1]/"infoot_vit/configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["device"] == "cuda",path
        (canonical if config["mode"].endswith("_lowrank") else validate_config)(config)
