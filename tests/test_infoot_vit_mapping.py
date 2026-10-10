from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import ot
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infoot_vit.infoot_helper import infoot as legacy
from infoot_vit.infoot_helper.conditional import BalancedModel, calibrate_support, partial_projection
from infoot_vit.infoot_helper.feature_bank import BankWriter, FeatureBank, compatible_banks, digest, file_hash, write_json
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, inspect_fit, CONFIDENCE, validate_config
from infoot_vit.infoot_helper.mapping import FeatureMapper, MappingResult, load_mapped, select_images
from infoot_vit.infoot_helper.partial import (
    distance, kernel_state, information, information_gradient, entropy_subproblem,
    solve_partial, solver_config, feasibility, objective,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def representation(dim=3):
    return dict(grid=[2, 2], dim=dim, patch_order="row_major_no_special_tokens", encoder={"revision": "synthetic"},
                layer="synthetic_patch_layer", preprocessing={"rgb": "synthetic"}, normalization="none")


def bank(root, name, x, domain, split="train", ids=None):
    ids = ids or [f"{domain}_{split}_{i}" for i in range(len(x))]
    w = BankWriter(root / name, domain=domain, split=split, representation=representation(x.shape[-1]), provenance={})
    for block in range(0, len(x), 2):
        w.append(x[block:block+2], [dict(sample_id=i, domain=domain) for i in ids[block:block+2]])
    w.finish()
    return FeatureBank.load(root / name)


@pytest.fixture
def banks(tmp_path):
    g = torch.Generator().manual_seed(503)
    x = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
    y = torch.randn(3, 4, 3, generator=g, dtype=torch.float64) + .3
    source = bank(tmp_path, "source", x, "cat")
    target = bank(tmp_path, "target", y, "dog")
    query = bank(tmp_path, "query", x * .9 + .02, "cat", "val")
    return source, target, query


def config(mode, mass=.8):
    c = dict(mode=mode, source_bank="source", target_bank="target", solver=dict(lam=0., reg=.4, h=.7),
             projection=dict(confidence=dict(support_calibration="disabled")))
    if mode == "grouped_partial":
        c["partial"] = dict(keep_mass=mass, solver=dict(lam=0., reg=.4, h=.7))
    return c


def mapped(mapper, bank, **kwargs):
    return mapper.map_features(bank.features, bank.ids, valid_mask=torch.ones(bank.features.shape[:2], dtype=torch.bool), **kwargs)


def test_lossless_noncontiguous_maps():
    x = torch.arange(24).reshape(2, 3, 4).transpose(1, 2)
    assert not x.is_contiguous()
    assert torch.equal(x, x.reshape(2, -1).reshape_as(x))


@pytest.mark.parametrize("mode", ["whole_map", "patch_global", "grouped_patch", "grouped_partial"])
def test_fit_save_reload_unseen_batch_independence_no_refit(tmp_path, banks, monkeypatch, mode):
    source, target, query = banks
    observed = []
    original = BalancedModel.fit.__func__
    def spy(cls, x, y, *args, **kwargs):
        observed.append((len(x), len(y)))
        return original(cls, x, y, *args, **kwargs)
    monkeypatch.setattr(BalancedModel, "fit", classmethod(spy))
    directory = fit_mapping(config(mode), root=tmp_path)
    assert directory.parent == tmp_path / "outputs/infoot_vit"
    assert list((directory / "plans").rglob("latest.pt"))
    assert observed == ([(8, 12)] if mode == "patch_global" else [(2, 3), (8, 12)] if mode == "grouped_patch" else [(2, 3)])
    monkeypatch.setattr(BalancedModel, "fit", lambda *a, **k: pytest.fail("inference refit"))
    monkeypatch.setattr(legacy.FusedInfoOT, "solve", lambda *a, **k: pytest.fail("legacy inference refit"))
    mapper = FeatureMapper.load(directory)
    a = mapped(mapper, query, return_metadata=True, chunk_size=2)
    b = mapped(FeatureMapper.load(directory), query, return_metadata=True, chunk_size=1)
    torch.testing.assert_close(a.mapped_features, b.mapped_features, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.match_confidence, b.match_confidence, atol=1e-12, rtol=1e-12)
    assert torch.equal(a.valid_mask, b.valid_mask)
    perm = torch.tensor([1, 0])
    c = mapper.map_features(query.features[perm], [query.ids[i] for i in perm],
        valid_mask=torch.ones(2, 4, dtype=torch.bool), return_metadata=True)
    torch.testing.assert_close(a.mapped_features, c.mapped_features[perm], atol=1e-12, rtol=1e-12)
    assert a.mapped_features.shape == (2, 4, 3)
    assert not a.mapped_features.requires_grad
    if mode == "whole_map":
        alpha = mapper.image.conditional_weights(query.features.reshape(2, -1))
        torch.testing.assert_close(a.mapped_features, (alpha @ target.features.reshape(3, -1)).reshape(2, 4, 3))
    if mode == "grouped_partial":
        with pytest.raises(ValueError, match="metadata"):
            mapped(mapper, query)
        for row in a.diagnostics["queries"]:
            assert row["matched_rejected_accounting_error"] < 1e-14
            error = (torch.tensor(row["ot_rejected_mass"]) + torch.tensor(row["support_invalid_retained_mass"])
                     + a.match_confidence[query.ids.index(row["query_id"])].float() - 1).abs().max()
            assert error < 1e-6
    result, manifest = mapper.project_bank(query, tmp_path / "projected")
    restored, state, _ = load_mapped(tmp_path / "projected", mapper_id=mapper.manifest["artifact_id"])
    torch.testing.assert_close(result.mapped_features, restored.mapped_features)
    assert state["ids"] == query.ids
    with pytest.raises(ValueError, match="identity"):
        load_mapped(tmp_path / "projected", mapper_id="wrong")


def test_balanced_conditional_reference_target_density_and_local_repair():
    x = torch.tensor([[0.], [1.], [3.]], dtype=torch.float64)
    y = torch.tensor([[0.], [.1], [.2], [4.]], dtype=torch.float64)
    m = BalancedModel.fit(x, y, dict(lam=0., h=1., reg=.5))
    query = torch.tensor([[.2], [2.], [100.]], dtype=torch.float64)
    kq = m.query_kernel(query).exp()
    direct = kq @ m.plan @ m.ky.T / (kq.mean(1)[:, None] * m.fy[None])
    expected = direct / direct.sum(1, keepdim=True)
    torch.testing.assert_close(m.conditional_weights(query[:2]), expected[:2], atol=1e-12, rtol=1e-12)
    wrong = kq[:2] @ m.plan @ m.ky.T
    assert not torch.allclose(expected[:2], wrong / wrong.sum(1, keepdim=True))
    assert torch.isfinite(m.project(query)).all()  # Very far queries remain numerical, not claimed in-support.
    torch.testing.assert_close(m.project(query), torch.cat([m.project(row[None]) for row in query]))
    theta = m.pair_weights(query)
    torch.testing.assert_close(theta.sum(1), m.conditional_weights(query))
    old = legacy.FusedInfoOT(x, y, h=1.); old.P = m.plan
    torch.testing.assert_close(m.conditional_weights(query[:2]),
        old.conditional_score(query[:2]) / old.conditional_score(query[:2]).sum(1, keepdim=True))
    changed = BalancedModel(x, y, m.state, bandwidth_multiplier=.5)
    assert not torch.allclose(changed.project(query[:2]), m.project(query[:2]))


def test_grouped_dense_and_explicit_target_id_join(tmp_path, banks):
    _, _, query = banks
    directory = fit_mapping(config("grouped_patch"), root=tmp_path)
    mapper = FeatureMapper.load(directory)
    result = mapped(mapper, query)
    for bi, patches in enumerate(query.features):
        alpha = mapper.image.conditional_weights(patches.reshape(1, -1))[0]
        score = mapper.patch.scores(patches).reshape(4, 3, 4)
        beta = score / score.sum(2, keepdim=True)
        expected = torch.einsum("j,pjq,jqd->pd", alpha, beta, mapper.y)
        torch.testing.assert_close(result[bi], expected)
        global_weights = mapper.patch.conditional_weights(patches).reshape(4, 3, 4)
        torch.testing.assert_close(beta, global_weights / global_weights.sum(2, keepdim=True))
        torch.testing.assert_close((alpha[None, :, None] * beta).sum(2), alpha.expand(4, -1))
    # Permute an ALREADY fitted target support, plan columns and associated IDs.
    state = deepcopy(mapper.patch.state)
    permutation = torch.tensor([2, 0, 1])
    columns = (permutation[:, None] * 4 + torch.arange(4)).flatten()
    state["plan"] = state["plan"][:, columns]
    state["target_ids"] = [mapper.target.ids[i] for i in permutation]
    path = directory / mapper.manifest["models"]["patch"]["file"]
    torch.save(state, path)
    manifest = mapper.manifest
    manifest["models"]["patch"]["sha256"] = file_hash(path)
    manifest["artifact_id"] = digest({k:v for k,v in manifest.items() if k != "artifact_id"})
    write_json(directory / "manifest.json", manifest)
    torch.testing.assert_close(result, mapped(FeatureMapper.load(directory), query))


@pytest.mark.parametrize("vary_mass",[False,True])
def test_information_gradient_variable_marginals_and_balanced_direction(vary_mass):
    x = torch.tensor([[0.], [.4], [2.]], dtype=torch.float64)
    kx, _ = kernel_state(x, .8); ky, _ = kernel_state(x * 2 + .1, 1.)
    plan = torch.tensor([[.1, .03, .02], [.01, .13, .04], [.03, .06, .18]], dtype=torch.float64)
    grad = information_gradient(plan, kx, ky)
    direction = torch.tensor([[.3, -.1, 0], [0, .2, -.1], [-.2, .1, -.2]], dtype=torch.float64)
    if vary_mass:
        direction[0,0]+=.1
    eps = 1e-6
    finite = (information(plan + eps*direction, kx, ky) - information(plan - eps*direction, kx, ky)) / (2*eps)
    torch.testing.assert_close((grad*direction).sum(), finite, atol=1e-9, rtol=1e-8)
    wrong = -legacy.migrad(plan, kx, ky)
    assert abs(float(((grad-wrong)*direction).sum())) > 1e-3
    balanced = torch.eye(3, dtype=torch.float64) * .2 + torch.ones(3, 3, dtype=torch.float64) * (2/45)
    feasible_direction = torch.tensor([[1.,-1.,0],[-1.,1.,0],[0,0,0]], dtype=torch.float64)
    torch.testing.assert_close(information(balanced, kx, ky), (balanced * legacy.ratio(balanced, kx, ky).log()).sum())
    torch.testing.assert_close((information_gradient(balanced,kx,ky)*feasible_direction).sum(),
                               (-legacy.migrad(balanced,kx,ky)*feasible_direction).sum(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mass", [.7, .8, .9, 1.])
def test_partial_no_mi_limit_feasibility_entropy_shift_and_rejection(mass):
    cost = torch.tensor([[0., .1, 10.], [.1, 0., 10.], [10., 10., 20.]], dtype=torch.float64)
    k = torch.eye(3, dtype=torch.float64)
    cfg = solver_config(dict(lam=0., reg=.2, cost_scale=1., max_inner_steps=50000))
    a = b = torch.full((3,), 1/3, dtype=torch.float64)
    plan, report = solve_partial(cost, k, k, keep_mass=mass, config=cfg)
    feasibility(plan, a, b, mass, cfg)
    direct, _ = entropy_subproblem(a,b,cost,mass,cfg)
    shifted, _ = entropy_subproblem(a,b,cost-500,mass,cfg)
    torch.testing.assert_close(plan, direct, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(plan, shifted, atol=1e-9, rtol=1e-8)
    omega = (torch.special.xlogy(plan,plan)-plan).sum()
    torch.testing.assert_close(torch.special.xlogy(plan,plan).sum()-omega, plan.sum())
    if mass < 1:
        assert plan.sum(1)[2] < a[2] - .05
        assert (plan > 0).all()  # Fixed mass is not a required count of zero rows.
    else:
        torch.testing.assert_close(plan.sum(1), a, atol=1e-8, rtol=1e-8)
    with pytest.raises(ValueError, match="Infeasible"):
        feasibility(plan * 2, a, b, mass, cfg)


def test_partial_information_full_objective_is_nonincreasing_and_stalls_explicit():
    x = torch.tensor([[0.],[1.],[2.]], dtype=torch.float64)
    k,_ = kernel_state(x,.7)
    plan, report = solve_partial(distance(x,x),k,k,config=dict(lam=.1,reg=.2))
    values = [r["objective"] for r in report["history"]]
    assert report["status"] == "converged"
    assert all(b <= a+1e-12 for a,b in zip(values,values[1:]))
    _, short = solve_partial(distance(x,x),k,k,config=dict(lam=.1,reg=.2,max_outer_steps=1))
    assert short["status"] == "max_outer_steps"


def test_confidence_original_proposal_nonuniform_masses_and_rescaling():
    x = torch.tensor([[0.],[1.],[3.]],dtype=torch.float64)
    y = torch.tensor([[0.],[.1],[2.]],dtype=torch.float64)
    _,sx = kernel_state(x,.8); ky,sy=kernel_state(y,.8)
    a = torch.tensor([.2,.3,.5],dtype=torch.float64)
    b = torch.tensor([.4,.3,.3],dtype=torch.float64)
    plan = torch.tensor([[.08,.01,0],[.03,.08,.01],[.08,.03,.06]],dtype=torch.float64)
    q = x + .02
    args = (q,x,y,plan,a,b,sx,sy,.8,dict(threshold=None))
    candidate,g,diag = partial_projection(*args)
    kq = torch.exp(-.5*(distance(q,x)/(sx*.8)).square())
    score = b[None] * (kq @ plan @ ky.T) / (ky @ b)[None]
    w = score/score.sum(1,keepdim=True)
    torch.testing.assert_close(candidate,w@y)
    torch.testing.assert_close(g,(kq@plan.sum(1))/(kq@a))
    wrong = b[None]*(kq@plan@ky.T)/(ky@(plan.sum(0)/plan.sum()))[None]
    assert not torch.allclose(w,wrong/wrong.sum(1,keepdim=True))
    scaled = partial_projection(q,x,y,plan*.2,a,b,sx,sy,.8,dict(threshold=None))
    torch.testing.assert_close(candidate,scaled[0]); torch.testing.assert_close(g*.2,scaled[1])
    assert (g>=0).all() and (g<=1).all()
    assert not torch.allclose(g,plan.sum(1)/a)
    far = partial_projection(q+1e5,x,y,plan,a,b,sx,sy,.8,dict(threshold=-100.))
    assert not far[2]["support_valid"].any() and not far[1].any()
    assert torch.isfinite(far[0]).all()


def test_delta_kernel_zero_retention_and_fit_only_calibration():
    x = torch.tensor([[0.],[10.],[20.]],dtype=torch.float64)
    _,scale=kernel_state(x,.01)
    a = b = torch.full((3,),1/3,dtype=torch.float64)
    plan = torch.diag(torch.tensor([.25,.25,0.],dtype=torch.float64))
    candidate,g,d = partial_projection(x,x,x,plan,a,b,scale,scale,.01,dict(threshold=None),
                                       target_kernel=torch.eye(3,dtype=torch.float64),
                                       query_logs=torch.eye(3,dtype=torch.float64).log())
    torch.testing.assert_close(g,plan.sum(1)/a)
    assert not candidate[2].any() and not d["weights"][2].any()
    support = calibrate_support(x,scale,.7,CONFIDENCE)
    assert support["policy"] == "fit_leave_one_out_log_density" and support["heuristic"]


def test_cache_mismatch_rejected_and_resume_reuses_complete_pairs(tmp_path,banks,monkeypatch):
    c = config("grouped_partial")
    directory = fit_mapping(c,root=tmp_path)
    import infoot_vit.infoot_helper.fit_mapping as fit_module
    monkeypatch.setattr(fit_module,"solve_partial",lambda *a,**k:pytest.fail("pair refit"))
    assert fit_mapping(c,root=tmp_path,resume=directory) == directory
    changed=deepcopy(c); changed["partial"]["keep_mass"] = .7
    with pytest.raises(ValueError,match="fingerprint"):
        fit_mapping(changed,root=tmp_path,resume=directory)
    source,_,_=banks
    shard = source.path / source.manifest["shards"][0]["file"]
    with shard.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError,match="changed"):
        FeatureMapper.load(directory)


def test_invalid_banks_and_resource_dry_run(tmp_path,banks):
    source,target,query=banks
    with pytest.raises(ValueError,match="train split"):
        compatible_banks(query,target)
    changed=deepcopy(target); changed.manifest["representation"]["encoder"]={"revision":"other"}
    with pytest.raises(ValueError,match="mismatch"):
        compatible_banks(source,changed)
    for mass in (0,1.1):
        with pytest.raises(ValueError,match="keep_mass"):
            validate_config(config("grouped_partial",mass))
    with pytest.raises(ValueError,match="Degenerate"):
        kernel_state(torch.ones(2,3,dtype=torch.float64),.4)
    report=inspect_fit(config("grouped_partial"),tmp_path)
    assert report["resources"]["pairs"] == 6
    assert report["resources"]["plan_storage_bytes"] == 6*16*8


def test_selection_ties_stable_sampling_topk_and_averaging():
    alpha=torch.tensor([.4,.4,.2],dtype=torch.float64); ids=["b","a","c"]
    settings=dict(top_k_images=None,selection="argmax",seed=1)
    w,m=select_images(alpha,ids,"query",settings)
    assert w.tolist()==[0,1,0] and m==1
    top,_=select_images(alpha,ids,"query",dict(settings,selection="mean",top_k_images=1))
    torch.testing.assert_close(w,top)
    sample,_=select_images(alpha,ids,"query",dict(settings,selection="sample"))
    perm=torch.tensor([2,0,1]); rev=torch.argsort(perm)
    other,_=select_images(alpha[perm],[ids[i] for i in perm],"query",dict(settings,selection="sample"))
    torch.testing.assert_close(sample,other[rev])
    maps=torch.tensor([[-1.],[1.]],dtype=torch.float64)
    mean=torch.tensor([.5,.5],dtype=torch.float64)@maps
    assert not torch.any(mean==maps)  # Conditional averaging need not produce a target sample.


def test_partial_balanced_pair_limit(tmp_path,banks):
    directory=fit_mapping(config("grouped_partial",1.),root=tmp_path)
    mapper=FeatureMapper.load(directory)
    result=mapped(mapper,banks[2],return_metadata=True)
    torch.testing.assert_close(result.match_confidence,torch.ones_like(result.match_confidence),atol=1e-8,rtol=1e-8)
    pair=mapper._pair(mapper.source.ids[0],mapper.target.ids[0])
    local=BalancedModel.fit(mapper.x[0],mapper.y[0],mapper.config["partial"]["solver"])
    torch.testing.assert_close(pair["plan"],local.plan,atol=1e-9,rtol=1e-9)
    candidate,_,_=partial_projection(banks[2].features[0],mapper.x[0],mapper.y[0],pair["plan"],pair["a"],pair["b"],
        mapper.shared["sx"][0],mapper.shared["sy"][0],mapper.shared["h_projection"],dict(threshold=None))
    torch.testing.assert_close(candidate,local.project(banks[2].features[0]),atol=1e-10,rtol=1e-10)


def test_mask_all_invalid_policy():
    r=MappingResult(torch.randn(2,4,3),torch.zeros(2,4),torch.zeros(2,4,dtype=torch.bool),{})
    with pytest.raises(ValueError,match="cat_val"):
        r.conditioning(["cat_val_1","cat_val_2"])
    z,mask=r.conditioning(["cat_val_1","cat_val_2"],all_invalid_policy="bypass")
    assert mask.all() and torch.equal(z,r.mapped_features)


def test_partial_streamed_aggregation_equals_dense_accounting(tmp_path,banks):
    mapper=FeatureMapper.load(fit_mapping(config("grouped_partial"),root=tmp_path))
    query=banks[2]
    actual=mapped(mapper,query,return_metadata=True)
    for qi, patches in enumerate(query.features):
        theta=mapper.image.pair_weights(patches.reshape(1,-1))[0]
        numerator=torch.zeros_like(patches); denominator=torch.zeros(4,dtype=torch.float64)
        grouped_budget=torch.zeros(4,len(mapper.target.ids),dtype=torch.float64)
        for i,sid in enumerate(mapper.source.ids):
            for j,tid in enumerate(mapper.target.ids):
                pair=mapper._pair(sid,tid)
                candidate,g,d=partial_projection(patches,mapper.x[i],mapper.y[j],pair["plan"],pair["a"],pair["b"],
                    mapper.shared["sx"][i],mapper.shared["sy"][j],mapper.shared["h_projection"],mapper.shared["support"][i])
                matched=theta[i,j]*g[:,None]*d["weights"]
                rejected=theta[i,j]*(1-g)
                grouped_budget[:,j]+=matched.sum(1)+rejected
                numerator+=matched@mapper.y[j]
                denominator+=matched.sum(1)
        torch.testing.assert_close(actual.mapped_features[qi],numerator/denominator[:,None],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(actual.match_confidence[qi],denominator,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(grouped_budget,theta.sum(0).expand(4,-1),atol=1e-12,rtol=1e-12)


def test_interrupted_pair_bank_resumes_without_refitting_completed_pairs(tmp_path,banks,monkeypatch):
    import importlib
    module=importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    original=module.solve_partial
    count=0
    def interrupted(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            raise RuntimeError("simulated interruption")
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"solve_partial",interrupted)
    with pytest.raises(RuntimeError,match="simulated"):
        fit_mapping(config("grouped_partial"),root=tmp_path)
    directory=next((tmp_path/"outputs/infoot_vit").iterdir())
    entry=json.loads((directory/"pairs.jsonl").read_text().strip())
    old_hash=file_hash(directory/entry["file"])
    count=0
    def continuation(*args,**kwargs):
        nonlocal count
        count+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"solve_partial",continuation)
    fit_mapping(config("grouped_partial"),root=tmp_path,resume=directory)
    assert count==5 and file_hash(directory/entry["file"])==old_hash
    assert FeatureMapper.load(directory).manifest["pair_count"]==6


def test_failed_inner_solve_and_nonconverged_fit_are_not_accepted(tmp_path,banks):
    cost=torch.tensor([[0.,1.,5.],[1.,2.,.1]],dtype=torch.float64)
    a=torch.tensor([.4,.6],dtype=torch.float64); b=torch.tensor([.2,.3,.5],dtype=torch.float64)
    c=solver_config(dict(max_inner_steps=1,reg=.05))
    with pytest.raises((RuntimeError,ValueError)):
        entropy_subproblem(a,b,cost,.9,c)
    cfg=config("whole_map"); cfg["solver"].update(lam=.2,max_outer_steps=1)
    with pytest.raises(RuntimeError,match="max_outer_steps"):
        fit_mapping(cfg,root=tmp_path)
    directory=next((tmp_path/"outputs/infoot_vit").iterdir())
    assert (directory/"plans/image/latest.pt").exists()
    with pytest.raises(ValueError,match="incomplete"):
        FeatureMapper.load(directory)


def test_all_invalid_partial_projection_is_explicit_and_metadata_survives(tmp_path,banks):
    c=config("grouped_partial")
    c["projection"]["confidence"].update(support_calibration="fixed",support_log_threshold=1.,all_invalid_policy="bypass")
    mapper=FeatureMapper.load(fit_mapping(c,root=tmp_path))
    result=mapped(mapper,banks[2],return_metadata=True)
    assert not result.valid_mask.any() and not result.match_confidence.any() and not result.mapped_features.any()
    with pytest.raises(ValueError,match="All condition tokens rejected"):
        result.conditioning(banks[2].ids)
    mapper.project_bank(banks[2],tmp_path/"masked")
    recovered,_,_=load_mapped(tmp_path/"masked")
    assert not recovered.valid_mask.any()


def test_bank_invalid_ids_padding_grid_and_nonfinite(tmp_path):
    from infoot_vit.infoot_helper.feature_bank import validate_maps
    x=torch.randn(2,4,3); mask=torch.ones(2,4,dtype=torch.bool)
    for ids in (["a","a"],["a",None]):
        with pytest.raises(ValueError,match="stable image ID"):
            validate_maps(x,ids,representation(),mask)
    badmask=mask.clone(); badmask[0,0]=False
    with pytest.raises(ValueError,match="fully valid"):
        validate_maps(x,["a","b"],representation(),badmask)
    with pytest.raises(ValueError,match="fixed patch grid"):
        validate_maps(x,["a","b"],dict(representation(),grid=[3,2]),mask)
    x[0,0,0]=torch.nan
    with pytest.raises(ValueError,match="finite"):
        validate_maps(x,["a","b"],representation(),mask)


@pytest.mark.parametrize("solver",["euler","heun"])
def test_existing_sampler_mask_passthrough_and_architecture_unchanged(solver):
    from test_pdae_v2 import branch,activate_cross_attention
    from diffusion_ot.evaluation.stage1a_eval import integrate_pdae_flow
    torch.manual_seed(52)
    model=branch().eval(); activate_cross_attention(model)
    keys=list(model.state_dict())
    parameters=sum(p.numel() for p in model.parameters())
    z=torch.randn(2,4,768)
    valid=torch.tensor([[True,True,False,False],[False,False,False,False]])
    result=MappingResult(z,valid.float(),valid,{})
    tokens,padding=result.conditioning(["q1","q2"],all_invalid_policy="bypass")
    dirty=tokens.masked_fill(padding[...,None],1e6)
    noise=torch.randn(2,4,8,8)
    def sample(condition,chunk):
        return integrate_pdae_flow(model,model.semantic_transformer.base,noise,condition,num_steps=2,
            guidance_scale=1.5,solver=solver,condition_padding_mask=padding,batch_size=chunk)
    actual=sample(tokens,2)
    torch.testing.assert_close(actual,sample(dirty,1),atol=2e-6,rtol=2e-5)
    assert torch.isfinite(actual).all()
    assert list(model.state_dict())==keys and sum(p.numel() for p in model.parameters())==parameters


def test_explicit_reuse_image_router_without_second_solve(tmp_path,banks,monkeypatch):
    whole=fit_mapping(config("whole_map"),root=tmp_path)
    c=config("grouped_partial"); c["image_model"]=str(whole)
    monkeypatch.setattr(BalancedModel,"fit",lambda *a,**k:pytest.fail("unnecessary image refit"))
    directory=fit_mapping(c,root=tmp_path)
    assert FeatureMapper.load(directory).manifest["reused_models"]["image"]==FeatureMapper.load(whole).manifest["artifact_id"]


def test_rgb_siglip_bank_extraction_reuses_unpooled_encoder(tmp_path,monkeypatch):
    import yaml
    from infoot_vit.bank import build_bank
    train=dict(domain="cat",data_config="data.yaml",encoder=dict(kind="siglip2_vit_b16",frozen=True,local_dir="siglip"))
    (tmp_path/"train.yaml").write_text(yaml.safe_dump(train))
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(dict(manifest_dir="manifests",image_size=256,center_crop=True)))
    (tmp_path/"manifests").mkdir()
    records=[dict(sample_id=f"cat_{i}",domain="cat") for i in range(3)]
    (tmp_path/"manifests/cat_train.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    class Loader:
        def __init__(self,*args):pass
        def load(self,record):return torch.full((3,256,256),int(record["sample_id"][-1])*.2-.2)
    class Encoder:
        snapshot_identity={"model_id":"mock_SigLIP","revision":"test"}
        def to(self,*args):return self
        def eval(self):return self
        def __call__(self,rgb):
            assert rgb.shape[1:]==(3,256,256) and rgb.min()>=-1 and rgb.max()<=1
            return rgb[:,0,0,0,None,None].expand(-1,196,768)+torch.arange(196)[None,:,None]*.01
    monkeypatch.setattr(build_bank,"OriginalImageLoader",Loader)
    monkeypatch.setattr(build_bank.FrozenSiglipPatchEncoder,"from_local",lambda *a:Encoder())
    build_bank.main("cat",["--project-root",str(tmp_path),"--train-config","train.yaml","--batch-size","2","--device","cpu"])
    saved=FeatureBank.load(tmp_path/"data/infoot_vit/cat_train")
    assert saved.features.shape==(3,196,768)
    assert saved.manifest["representation"]["normalization"]=="none"
    assert saved.ids==[r["sample_id"] for r in records]


def test_fit_project_and_compare_cli(tmp_path,banks):
    import yaml
    from infoot_vit.infoot_fit import main as fit_main
    from infoot_vit.infoot_test import main as project_main
    from infoot_vit.infoot_compare import main as compare_main
    c=tmp_path/"config.yaml"; c.write_text(yaml.safe_dump(config("whole_map")))
    args=["--config",str(c),"--project-root",str(tmp_path),"--threads","1"]
    assert fit_main(args+["--dry-run"])==0
    assert not (tmp_path/"outputs").exists()
    assert fit_main(args)==0
    directory=next((tmp_path/"outputs/infoot_vit").iterdir())
    projection=["--mapping",str(directory),"--query-bank",str(banks[2].path),"--threads","1"]
    assert project_main(projection+["--dry-run"])==0
    assert project_main(projection+["--output-dir",str(tmp_path/"cli-projection")])==0
    assert compare_main(["--mappings",str(directory),"--query-bank",str(banks[2].path),
                         "--output-dir",str(tmp_path/"comparison"),"--threads","1"])==0
    comparison=json.loads((tmp_path/"comparison/comparison.json").read_text())
    assert comparison["query_ids"]==banks[2].ids


def test_fixed_checkpoint_generation_uses_cached_tokens_and_existing_mask(tmp_path,banks,monkeypatch):
    from types import SimpleNamespace
    import yaml
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    from infoot_vit.infoot_helper.evaluate_mapping import generate
    mapper=FeatureMapper.load(fit_mapping(config("grouped_partial"),root=tmp_path))
    result=mapped(mapper,banks[2],return_metadata=True)
    path=tmp_path/"checkpoint.pt"; path.write_bytes(b"fake-checkpoint")
    cfg=dict(domain="dog",data_config="data.yaml",class_conditioning=dict(null_label=1000))
    (tmp_path/"dog.yaml").write_text(yaml.safe_dump(cfg))
    evaluator=SimpleNamespace(branch=SimpleNamespace(encoder=SimpleNamespace(
        snapshot_identity=mapper.target.representation["encoder"],architecture_spec=dict(features=mapper.target.representation["layer"]))),
        transformer=None,vae=None,device="cpu",model_dtype=torch.float32,checkpoint_path=path)
    monkeypatch.setattr(stage1a_eval,"load_stage1a_evaluator",lambda *a,**k:evaluator)
    observed={}
    def sampler(branch,transformer,noise,tokens,**kwargs):
        observed.update(kwargs); observed["noise"]=noise.clone()
        torch.testing.assert_close(tokens,result.mapped_features.float())
        assert noise.shape==(2,4,32,32)
        return noise
    monkeypatch.setattr(stage1a_eval,"integrate_pdae_flow",sampler)
    monkeypatch.setattr(stage1a_eval,"decode_vae_latents",lambda vae,z,**k:torch.zeros(len(z),3,8,8))
    monkeypatch.setattr(ground_truth,"load_ground_truth_images",lambda cfg,records:torch.ones(len(records),3,8,8)*.5)
    output=tmp_path/"generation"; output.mkdir()
    report=generate(mapper,banks[2],result,output,root=tmp_path,train_config="dog.yaml",eval_config="unused.yaml",device="cpu")
    assert observed["condition_padding_mask"].dtype==torch.bool
    assert torch.equal(observed["condition_padding_mask"],~result.valid_mask)
    assert report["mapper_id"]==mapper.manifest["artifact_id"]
    assert report["checkpoint_sha256"]==file_hash(path)
    assert report["row_order"]==["original_source","top1_routed_target_reference_not_ground_truth","translated"]
    assert (output/"translation_grid.png").exists() and (output/"confidence_grid.png").exists()


def test_small_reg_log_solver_and_numeric_underflow_are_distinct_from_rejection():
    a=b=torch.tensor([.5,.5],dtype=torch.float64)
    cost=torch.tensor([[1.,1000.],[1000.,1.]],dtype=torch.float64)
    plan,report=entropy_subproblem(a,b,cost,.8,solver_config(dict(reg=1e-3)))
    assert torch.isfinite(plan).all() and report["mass_error"]<1e-9
    x=torch.tensor([[0.],[1.]],dtype=torch.float64); y=x+2
    _,scale=kernel_state(x,.01)
    rejected=torch.tensor([[.4,0.],[0.,0.]],dtype=torch.float64)
    args=(x[1:],x,y,rejected,a,b,scale,scale,.01)
    with pytest.raises(ValueError,match="underflow.*not genuine"):
        partial_projection(*args,dict(threshold=None))
    candidate,g,diag=partial_projection(*args,dict(threshold=1.))
    assert diag["confidence_underflow"].all() and diag["score_log_retry"].all()
    assert torch.isfinite(candidate).all() and not g.any()
