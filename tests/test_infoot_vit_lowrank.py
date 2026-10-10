"""Numerical references and artifact contracts, not image-quality evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads, mapped, bank, config as dense_config
from infoot_vit.lowrank import kernels, objective, solver
from infoot_vit.lowrank.config import canonical, resources
from infoot_vit.lowrank.experiment import fit, inspect_fit, validate_kernels
from infoot_vit.lowrank.mapping import LowRankMapper
from infoot_vit.lowrank.storage import pack_factors, validate_factors
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor, write_json, digest


def tiny_config():
    return dict(source_bank="source",target_bank="target",sampling=dict(images_per_domain=2,seed=42),
        transport_rank=2,kernel_rank=4,kernel=dict(h=.8,check_pairs=32,density_queries=2,error_warn_relative_rmse=10.),
        image_solver=dict(lam=0.,reg=.4,h=.7),threads=1,device="cpu",
        estimator=dict(exact=True,audit_samples=64),
        optimizer=dict(max_steps=3,stationarity_tolerance=10.,chunk_size=7,checkpoint_every=1,audit_every=1))


def problem(n=6,m=7):
    rng = torch.Generator().manual_seed(300)
    x,y = torch.randn(n,3,generator=rng,dtype=torch.float64),torch.randn(m,3,generator=rng,dtype=torch.float64)
    fx,sx = kernels.fit_features(x,5,.8,1,chunk_size=2)
    fy,sy = kernels.fit_features(y,5,.8,2,chunk_size=3)
    cfg = canonical(tiny_config())["optimizer"]
    factors = solver.initialize(n,m,2,cfg,"cpu")
    pairs = objective.with_cost(objective.sample_pairs(n,m,seed=3,exact=True),x,y,scale=1.,chunk_size=3)
    return x,y,fx,fy,sx,sy,cfg,factors,pairs


@pytest.mark.parametrize("lam,reg",[(0.,.05),(.3,.05),(.3,0.)])
def test_exact_sampled_objective_and_all_factor_gradients_match_dense(lam,reg):
    x,y,fx,fy,_,_,cfg,factors,pairs = problem()
    result,gradients = objective.evaluate(*factors,fx,fy,pairs,lam=lam,reg=reg,gradient=True,chunk_size=5)
    q,r,g = [v.clone().requires_grad_(True) for v in factors]
    plan = (q/g)@r.T
    kx,ky = fx@fx.T,fy@fy.T
    ratio = (kx@plan@ky.T)/(kx.mean(1)[:,None]*ky.mean(1)[None,:])
    cost = (plan*torch.cdist(x,y)).sum()
    mi = (plan*ratio.log()).sum()
    entropy = (plan*(plan.log()-1)).sum()
    loss = cost-lam*mi+reg*entropy
    for key,expected in (("cost",cost),("mi",mi),("entropy",entropy),("objective",loss)):
        assert result[key] == pytest.approx(float(expected.detach()),abs=2e-12)
    for got,expected in zip(gradients,torch.autograd.grad(loss,(q,r,g))):
        torch.testing.assert_close(got,expected,atol=2e-11,rtol=2e-11)


def test_kernel_bandwidth_positivity_error_probe_and_chunk_invariance():
    x,y,fx,fy,sx,sy,*_ = problem()
    assert sx["scale"] == pytest.approx(float((torch.cdist(x,x).square().mean()/2).sqrt()),abs=1e-12)
    assert (fx@fx.T >= 0).all()
    torch.testing.assert_close((fx@fx.T).diagonal(),torch.ones(len(x),dtype=torch.float64),atol=1e-14,rtol=0)
    torch.testing.assert_close(kernels.features(x,sx,1),fx,atol=1e-14,rtol=0)
    report = kernels.error_report(x,fx,sx,seed=50,count=30,density_queries=2,chunk_size=3)
    assert report["relative_rmse"] >= 0 and report["density_relative_error_max"] >= 0


def test_mirror_descent_keeps_constraints_and_reduces_infoot_objective():
    x,y,fx,fy,sx,sy,cfg,factors,pairs = problem()
    cfg.update(max_steps=12,stationarity_tolerance=1e-12,lam=.3,reg=.03,audit_every=2)
    audit = objective.with_cost(objective.sample_pairs(len(x),len(y),seed=60,count=256),x,y,scale=1.)
    final,report = solver.solve(factors,fx,fy,pairs,audit,cfg)
    assert len(report["history"]) > 1
    losses = [r["objective"] for r in report["history"]]
    assert all(b <= a+cfg["objective_tolerance"] for a,b in zip(losses,losses[1:]))
    assert losses[-1] < losses[0]
    check = solver.check(*final,cfg["constraint_tolerance"],cfg["min_g"])
    assert check["plan_row_relative_max"] < cfg["constraint_tolerance"]
    assert "standard_error" in report["history"][-1]["audit"]


def test_stratified_samples_cover_both_supports_are_fixed_and_unbiased():
    p = objective.sample_pairs(6,7,seed=42,per_row=2)
    again = objective.sample_pairs(6,7,seed=42,per_row=2)
    assert torch.equal(p["i"],again["i"]) and torch.equal(p["j"],again["j"])
    assert len(p["i"].unique()) == 6 and len(p["j"].unique()) == 7
    x,y,fx,fy,_,_,_,factors,exact = problem()
    baseline = objective.evaluate(*factors,fx,fy,exact,lam=.3,reg=.05)
    audit = objective.with_cost(objective.sample_pairs(6,7,seed=42,count=10000),x,y,scale=1.)
    approx = objective.evaluate(*factors,fx,fy,audit,lam=.3,reg=.05)
    assert abs(approx["objective"]-baseline["objective"]) < 4*approx["standard_error"]["objective"]


def test_float32_factor_roundtrip_marginals_and_corruption():
    *_,cfg,factors,pairs = problem()
    state = pack_factors(factors,cfg)
    assert all(state[k].dtype == torch.float32 for k in ("q","r","g"))
    assert torch.equal(state["q_rows"],state["q"].double().sum(1))
    assert state["quantization"]["q"]["max_abs_error"] > 0
    validate_factors(state,cfg,(6,7,2))
    bad = deepcopy(state); bad["q"] *= 1.01
    bad["q_rows"],bad["q_columns"] = bad["q"].double().sum(1),bad["q"].double().sum(0)
    with pytest.raises(ValueError,match="constraints"):
        validate_factors(bad,cfg)


def test_float32_tiny_entries_may_become_zero_without_rebalancing():
    cfg = canonical(tiny_config())["optimizer"]
    q = torch.tensor([[.5,1e-60],[1e-60,.5]],dtype=torch.float64)
    r,g = q.clone(),torch.tensor([.5,.5],dtype=torch.float64)
    state = pack_factors((q,r,g),cfg)
    assert state["quantization"]["q"]["underflow_entries"] == 2
    assert torch.equal(state["q"],torch.eye(2)*.5)
    validate_factors(state,cfg)
    f = torch.eye(2,dtype=torch.float64)
    pairs = objective.with_cost(objective.sample_pairs(2,2,seed=1,exact=True),f,f,scale=1.)
    values,grads = objective.evaluate(state["q"].double(),state["r"].double(),state["g"].double(),
                                     f,f,pairs,lam=.1,reg=.05,gradient=True)
    assert values["mi"] == pytest.approx(float(torch.tensor(2.).log()),abs=1e-7)
    assert all(torch.isfinite(v).all() for v in grads)
    projected,_ = solver.project(state["q"].double(),state["r"].double(),state["g"].double(),cfg)
    solver.check(*projected,cfg["constraint_tolerance"],cfg["min_g"])


def test_fit_grouped_projection_matches_dense_and_never_refits(tmp_path,banks,monkeypatch):
    directory = fit(tiny_config(),root=tmp_path)
    mapper = FeatureMapper.load(directory)
    assert isinstance(mapper,LowRankMapper)
    monkeypatch.setattr(solver,"solve",lambda *a,**k:pytest.fail("Inference fit"))
    monkeypatch.setattr(kernels,"fit_features",lambda *a,**k:pytest.fail("Inference kernel fit"))
    result = mapped(mapper,banks[2],return_metadata=True,chunk_size=1)
    roundtrip = mapped(LowRankMapper.load(directory),banks[2],return_metadata=True,chunk_size=2)
    torch.testing.assert_close(result.mapped_features,roundtrip.mapped_features,atol=1e-12,rtol=0)
    state = torch.load(directory/"factors.pt",weights_only=True)
    ks = torch.load(directory/"kernels.pt",weights_only=True)
    q,r,g = [state[k].double() for k in ("q","r","g")]
    fx,fy = ks["fx"].double(),ks["fy"].double()
    plan = (q/g)@r.T
    smoothing = plan@(fy@fy.T)/(fy@fy.T).mean(0)[None,:]
    for index,image in enumerate(banks[2].features):
        kq = kernels.features(image,ks["source"])@fx.T
        scores = kq@smoothing
        alpha = mapper.image.conditional_weights(image.reshape(1,-1))[0]
        expected = torch.zeros_like(image)
        for j in range(len(mapper.y)):
            block = scores[:,j*4:(j+1)*4]
            expected += alpha[j]*(block/block.sum(1,keepdim=True))@mapper.y[j]
        torch.testing.assert_close(result.mapped_features[index],expected,atol=2e-12,rtol=0)
    mapper.config["projection"]["target_chunk_size"] = 1
    torch.testing.assert_close(mapped(mapper,banks[2]),result.mapped_features,atol=2e-12,rtol=0)
    assert not list(directory.rglob("latest.pt"))
    mapper.project_bank(banks[2],tmp_path/"mapping_result")
    assert load_mapped(tmp_path/"mapping_result")[0].mapped_features.shape == banks[2].features.shape
    assert list((tmp_path/"mapping_result/logs").glob("*/mapping_report.json"))


def test_resource_budget_and_preserve_unchanged_raw_tokens(tmp_path,banks):
    c = canonical({})
    r = resources(2000,2000,196,768,c)
    assert r["saved_factor_bytes"] == 1605632000
    assert r["estimated_final_artifact_bytes"] < 4000000000
    assert r["objective_sample_count"] == 1568000
    before = banks[0].features.clone()
    report = inspect_fit(tiny_config(),tmp_path)
    assert report["sampling"]["source"]["seed"] == 42
    fit(tiny_config(),root=tmp_path)
    torch.testing.assert_close(banks[0].features,before,atol=0,rtol=0)


def test_config_sampling_and_cli_dry_run_are_deterministic(tmp_path,banks,capsys):
    import yaml
    from infoot_vit.infoot_fit_lowrank import main
    c = tiny_config()
    assert canonical(c) == canonical(canonical(c))
    first,second = inspect_fit(c,tmp_path),inspect_fit(c,tmp_path)
    assert first == second
    assert len(set(first["sampling"]["target"]["ordered_ids"])) == 2
    cfg = tmp_path/"lowrank.yaml"; cfg.write_text(yaml.safe_dump(c))
    assert main(["--config",str(cfg),"--project-root",str(tmp_path),"--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["transport_rank"] == 2
    assert not (tmp_path/"outputs").exists()
    bad = deepcopy(c); bad["sampling"]["images_per_domain"] = 4
    with pytest.raises(ValueError): inspect_fit(bad,tmp_path)
    bad = deepcopy(c); bad["source_bank"] = "query"
    with pytest.raises(ValueError,match="split"): inspect_fit(bad,tmp_path)


def test_estimator_and_kernel_metadata_validation():
    x,y,fx,fy,sx,sy,cfg,factors,pairs = problem()
    c = canonical(tiny_config()); c["estimator"].update(seed=3,audit_samples=64)
    audit = objective.with_cost(objective.sample_pairs(len(x),len(y),seed=4,count=64),x,y,scale=1.)
    samples = dict(training=pairs,audit=audit)
    objective.validate_samples(samples,len(x),len(y),c)
    bad = deepcopy(samples); bad["training"]["j"][0] = len(y)
    with pytest.raises(ValueError,match="index"): objective.validate_samples(bad,len(x),len(y),c)
    bad = deepcopy(samples); bad["audit"]["cost_scale"] = 2.
    with pytest.raises(ValueError,match="cost scale"): objective.validate_samples(bad,len(x),len(y),c)
    from infoot_vit.lowrank.storage import VERSION
    ks = dict(fx=fx.float(),fy=fy.float(),source=sx,target=sy,storage_version=VERSION)
    validate_kernels(ks,(len(x),len(y),5),3)
    ks["source"]["sigma"] *= 2
    with pytest.raises(ValueError,match="bandwidth"): validate_kernels(ks,(len(x),len(y),5),3)


def test_objective_and_gradient_chunk_invariance():
    *_,fx,fy,sx,sy,cfg,factors,pairs = problem()  # two leading raw feature arrays
    a,ga = objective.evaluate(*factors,fx,fy,pairs,lam=.3,reg=.05,gradient=True,chunk_size=1)
    b,gb = objective.evaluate(*factors,fx,fy,pairs,lam=.3,reg=.05,gradient=True,chunk_size=999)
    assert a == pytest.approx(b,abs=2e-12)
    for left,right in zip(ga,gb): torch.testing.assert_close(left,right,atol=2e-11,rtol=2e-11)


def test_failed_fit_resume_extends_budget_and_validates_checkpoint(tmp_path,banks,monkeypatch):
    from infoot_vit.lowrank import experiment
    c = tiny_config(); c["optimizer"].update(max_steps=1,stationarity_tolerance=1e-15)
    with pytest.raises(RuntimeError,match="max_steps"): fit(c,root=tmp_path)
    directory = next((tmp_path/"outputs/infoot_vit").iterdir())
    latest = directory/"factors/latest.pt"
    assert torch.load(latest,weights_only=True)["step"] == 1
    assert json.loads((directory/"manifest.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError,match="incomplete"): LowRankMapper.load(directory)
    monkeypatch.setattr(experiment.BalancedModel,"fit",lambda *a,**k:pytest.fail("Refitted registered image router"))
    monkeypatch.setattr(kernels,"fit_features",lambda *a,**k:pytest.fail("Refitted saved kernel"))
    c["optimizer"]["max_steps"] = 2
    with pytest.raises(RuntimeError,match="max_steps"): fit(c,root=tmp_path,resume=directory)
    assert torch.load(latest,weights_only=True)["step"] == 2
    runs = list((directory/"logs").iterdir())
    assert len(runs) == 2 and all((run/"run.json").exists() for run in runs)
    state = torch.load(latest,weights_only=True)
    state["step"] = 0  # Reject even a shape-/marginal-preserving corruption.
    save_tensor(latest,state)
    with pytest.raises(ValueError,match="checksum"): fit(c,root=tmp_path,resume=directory)


@pytest.mark.parametrize("where",["verify", "registered"])
def test_final_save_interrupt_cleanup_and_resume(tmp_path,banks,monkeypatch,where):
    from infoot_vit.lowrank import experiment
    c = tiny_config(); c["optimizer"]["max_steps"] = 1
    original_verify,original_cleanup = experiment._verified_save,experiment._cleanup_latest
    def fail_verify(directory,path,state,validator):
        if path.name == "factors.pt": raise OSError("simulated final verification failure")
        return original_verify(directory,path,state,validator)
    def fail_cleanup(directory,entry,log):
        if entry["file"] == "factors.pt": raise KeyboardInterrupt("after registration")
        return original_cleanup(directory,entry,log)
    monkeypatch.setattr(experiment,"_verified_save",fail_verify if where == "verify" else original_verify)
    monkeypatch.setattr(experiment,"_cleanup_latest",fail_cleanup if where == "registered" else original_cleanup)
    with pytest.raises(OSError if where == "verify" else KeyboardInterrupt): fit(c,root=tmp_path)
    directory = next((tmp_path/"outputs/infoot_vit").iterdir())
    manifest = json.loads((directory/"manifest.json").read_text())
    assert ("factors" in manifest["files"]) == (where == "registered")
    assert (directory/"factors/latest.pt").is_file()
    monkeypatch.setattr(experiment,"_verified_save",original_verify)
    monkeypatch.setattr(experiment,"_cleanup_latest",original_cleanup)
    if where == "registered":
        monkeypatch.setattr(solver,"solve",lambda *a,**k:pytest.fail("Refit verified final factors"))
    fit(c,root=tmp_path,resume=directory)
    mapper = LowRankMapper.load(directory)
    assert mapper.manifest["status"] == "complete" and not list(directory.rglob("latest.pt"))
    report = json.loads((directory/"fit_report.json").read_text())
    assert max(abs(v) for v in report["float32_minus_float64_audit"].values()) < 2e-6
    fit(c,root=tmp_path,resume=directory)  # Idempotent completed resume.


def test_saved_factor_corruption_is_not_a_missing_or_refittable_file(tmp_path,banks):
    directory = fit(tiny_config(),root=tmp_path)
    path = directory/"factors.pt"
    state = torch.load(path,weights_only=True); state["q"][0,0] *= 2
    save_tensor(path,state)
    with pytest.raises(ValueError,match="Missing/changed artifact"): LowRankMapper.load(directory)


def test_lowrank_and_dense_use_identical_frozen_pdae_protocol(tmp_path,banks,monkeypatch):
    from types import SimpleNamespace
    import yaml
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    from infoot_vit.infoot_helper.evaluate_mapping import generate
    from infoot_vit.infoot_helper.fit_mapping import fit_mapping
    from infoot_vit.infoot_compare import main as compare
    from infoot_vit.infoot_test import main as test_cli
    bank(tmp_path,"target_two",banks[1].features[:2],"dog",ids=banks[1].ids[:2])
    c = tiny_config(); c["target_bank"] = "target_two"
    dc = dense_config("grouped_patch"); dc["target_bank"] = "target_two"
    directories = [fit(c,root=tmp_path),fit_mapping(dc,root=tmp_path)]
    assert compare(["--mappings",*(str(d) for d in directories),"--query-bank",str(banks[2].path),
                    "--output-dir",str(tmp_path/"comparison"),"--count","2"]) == 0
    assert test_cli(["--mapping",str(directories[0]),"--query-bank",str(banks[2].path),"--dry-run"]) == 0
    mappers = [FeatureMapper.load(d) for d in directories]
    results = [mapped(m,banks[2],return_metadata=True) for m in mappers]
    checkpoint = tmp_path/"fixed.pt"; checkpoint.write_bytes(b"fixed-test-checkpoint")
    (tmp_path/"dog.yaml").write_text(yaml.safe_dump(dict(domain="dog",data_config="data.yaml",class_conditioning=dict(null_label=1000))))
    encoder = SimpleNamespace(snapshot_identity=banks[1].representation["encoder"],
                              architecture_spec=dict(features=banks[1].representation["layer"]))
    evaluator = SimpleNamespace(branch=SimpleNamespace(encoder=encoder),transformer=None,vae=None,
                               device="cpu",model_dtype=torch.float32,checkpoint_path=checkpoint)
    monkeypatch.setattr(stage1a_eval,"load_stage1a_evaluator",lambda *a,**k:evaluator)
    noises = []
    def sampler(branch,transformer,noise,tokens,**kwargs):
        assert not torch.is_grad_enabled()
        result = results[len(noises)]
        torch.testing.assert_close(tokens,result.mapped_features.float())
        assert torch.equal(kwargs["condition_padding_mask"],~result.valid_mask)
        noises.append(noise.clone())
        return noise
    monkeypatch.setattr(stage1a_eval,"integrate_pdae_flow",sampler)
    monkeypatch.setattr(stage1a_eval,"decode_vae_latents",lambda vae,z,**k:torch.zeros(len(z),3,8,8))
    monkeypatch.setattr(ground_truth,"load_ground_truth_images",lambda cfg,records:torch.full((len(records),3,8,8),.5))
    reports = []
    for i,(mapper,result) in enumerate(zip(mappers,results)):
        output = tmp_path/f"generation_{i}"; output.mkdir()
        reports.append(generate(mapper,banks[2],result,output,root=tmp_path,
            train_config="dog.yaml",eval_config="unused.yaml",checkpoint=checkpoint,device="cpu",seed=42,batch_size=i+1))
        assert (output/"translation_grid.png").exists()
    torch.testing.assert_close(noises[0],noises[1],atol=0,rtol=0)
    for key in ("query_ids","checkpoint_sha256","weights","per_image_noise_seeds","guidance_scale","num_steps","solver"):
        assert reports[0][key] == reports[1][key]


def test_projection_rejects_training_ids_omitted_from_fit(tmp_path,banks):
    directory = fit(tiny_config(),root=tmp_path)
    mapper = LowRankMapper.load(directory)
    omitted = next(sid for sid in banks[1].ids if sid not in mapper.target.ids)
    leaking = bank(tmp_path,"leaking",banks[2].features[:1],"cat","val",ids=[omitted])
    with pytest.raises(ValueError,match="disjoint|overlap|training"):
        mapper.project_bank(leaking,tmp_path/"leak_output")


def test_kernel_approximation_warning_is_explicit_and_persistent(tmp_path,banks):
    c = tiny_config(); c["kernel"]["error_warn_relative_rmse"] = .5
    with pytest.warns(UserWarning,match="positive-kernel relative RMSE"):
        directory = fit(c,root=tmp_path)
    checks = json.loads((directory/"fit_report.json").read_text())["kernel_approximation"]
    assert checks["source"]["relative_rmse"] > .5
    assert list(directory.glob("logs/*/kernel_approximation.json"))


def test_patch_fit_never_allocates_full_pairwise_tensors():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    n,m = 127,131
    class NoDensePatchMatrices(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            output = func(*args,**(kwargs or {}))
            for tensor in tree_leaves(output):
                if isinstance(tensor,torch.Tensor) and tensor.ndim >= 2:
                    assert tuple(tensor.shape[-2:]) not in {(n,n),(n,m),(m,n),(m,m)}, str(func)
            return output
    with NoDensePatchMatrices():
        rng = torch.Generator().manual_seed(44)
        x,y = (torch.randn(count,9,dtype=torch.float64,generator=rng) for count in (n,m))
        fx,sx = kernels.fit_features(x,8,.7,1,chunk_size=17)
        fy,_ = kernels.fit_features(y,8,.7,2,chunk_size=17)
        kernels.error_report(x,fx,sx,seed=3,count=33,density_queries=3,chunk_size=17)
        pairs = objective.with_cost(objective.sample_pairs(n,m,seed=4,per_row=2),x,y,chunk_size=17)
        audit = objective.with_cost(objective.sample_pairs(n,m,seed=5,count=71),x,y,scale=pairs["cost_scale"],chunk_size=17)
        cfg = canonical({})["optimizer"]; cfg.update(max_steps=2,chunk_size=17,audit_every=1)
        factors = solver.initialize(n,m,8,cfg,"cpu")
        final,report = solver.solve(factors,fx,fy,pairs,audit,cfg)
        assert report["history"] and final[0].shape == (n,8)
