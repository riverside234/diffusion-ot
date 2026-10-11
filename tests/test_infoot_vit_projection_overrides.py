"""Runtime bandwidth/top-k changes reuse immutable OT plans and training bases."""
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads, mapped, config
from test_infoot_vit_lowrank import tiny_config
from test_infoot_vit_lowrank_partial import partial_config
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional import BalancedModel, calibrate_support, normalize_rows
from infoot_vit.infoot_helper.feature_bank import file_hash
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped
from infoot_vit.infoot_helper.partial import distance
from infoot_vit.lowrank import kernels, solver, pair_kernels
from infoot_vit.lowrank.experiment import fit


def hashes(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file()}


def forbid_fitting(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Inference must not fit transport plans or kernel bases")
    monkeypatch.setattr(BalancedModel, "fit", unexpected)
    monkeypatch.setattr(solver, "solve", unexpected)
    monkeypatch.setattr(kernels, "fit_features", unexpected)
    monkeypatch.setattr("infoot_vit.infoot_helper.partial.solve_partial", unexpected)


def test_dense_partial_override_rebuilds_kernels_support_and_preserves_pairs(tmp_path, banks, monkeypatch):
    c = config("grouped_partial")
    c.update(fit_pair_top_k=1, device="cpu")
    c["partial"]["solver"]["h"] = .9
    c["projection"].update(bandwidth_multiplier=.5, confidence=dict(
        support_calibration="fit_leave_one_out_log_density", all_invalid_policy="bypass"))
    directory = fit_mapping(c, root=tmp_path)
    before = hashes(directory)
    base = FeatureMapper.load(directory)
    forbid_fitting(monkeypatch)
    settings = infoot_test.projection_settings(base.config, None, bandwidth=.14, top_k_images=1)
    mapper = FeatureMapper.load(directory, projection=settings)
    assert mapper.image.h == pytest.approx(.14)
    assert mapper.shared["h_projection"] == pytest.approx(.18)
    assert mapper.pairs == base.pairs and torch.equal(mapper.pair_mask, base.pair_mask)
    assert mapper.shared["support"] != base.shared["support"]
    for i, x in enumerate(mapper.x):
        assert mapper.shared["support"][i] == calibrate_support(x, mapper.shared["sx"][i], .9*(.14/.7), settings["confidence"])
    for j, y in enumerate(mapper.y):
        ky = (-.5*(distance(y,y)/(mapper.shared["sy"][j]*.18)).square()).exp()
        torch.testing.assert_close(mapper.projection_ky[j], ky)
    sid, tid = next(iter(mapper.pairs))
    i, j = mapper.source_index[sid], mapper.target_index[tid]
    query = banks[2].features[0]
    candidate, _, _, _ = mapper._partial_projection(query,i,j,sid,tid,mapper._partial_query(query,i))
    plan = mapper._pair(sid,tid)["plan"]
    kq = (-.5*(distance(query,mapper.x[i])/(mapper.shared["sx"][i]*.18)).square()).exp()
    ky = mapper.projection_ky[j]
    expected = normalize_rows((kq@plan@ky.T)/ky.mean(0))@mapper.y[j]
    torch.testing.assert_close(candidate,expected,atol=1e-11,rtol=1e-11)
    a = mapped(mapper,banks[2],return_metadata=True,chunk_size=1)
    b = mapped(mapper,banks[2],return_metadata=True,chunk_size=2)
    torch.testing.assert_close(a.mapped_features,b.mapped_features,rtol=0,atol=0)
    torch.testing.assert_close(a.match_confidence,b.match_confidence,rtol=0,atol=0)
    assert torch.equal(a.valid_mask,b.valid_mask)
    for row in a.diagnostics["queries"]:
        assert row["patch_projection_h"] == pytest.approx(.18)
        assert sum(w>0 for w in row["image_weights"]) == 1
        assert sum(row["image_weights"]) == pytest.approx(1.)
        assert row["matched_rejected_group_mass_residual"] < 1e-10
        assert row["fit_pair_discarded_routing_mass"] >= 0
    assert hashes(directory) == before


@pytest.mark.parametrize("partial", [False, True])
def test_lowrank_override_uses_saved_bases_and_matches_dense_algebra(tmp_path,banks,monkeypatch,partial):
    c = partial_config() if partial else tiny_config()
    c["kernel"].update(h=4.)  # Broad, accurate kernel for a deterministic small reference.
    c["kernel_rank"] = 64
    directory = fit(c,root=tmp_path)
    before = hashes(directory)
    baseline = FeatureMapper.load(directory)
    forbid_fitting(monkeypatch)
    settings = infoot_test.projection_settings(baseline.config,None,bandwidth=.35,top_k_images=1)
    settings["confidence"].update(support_calibration="fit_leave_one_out_log_density",all_invalid_policy="bypass")
    mapper = FeatureMapper.load(directory,projection=settings)
    assert mapper.image.h == pytest.approx(.35) and mapper.patch_projection_h == pytest.approx(2.)
    assert mapper.config["projection"] == settings
    assert mapper.manifest["config"]["projection"] == baseline.config["projection"]
    if partial:
        assert mapper.projection_kernel_report["accepted"]
    else:
        assert not hasattr(mapper, "projection_kernel_report")
    query = banks[2].features[0]
    if partial:
        assert mapper.pairs == baseline.pairs and torch.equal(mapper.pair_mask,baseline.pair_mask)
        for i, f in enumerate(mapper.fx):
            assert mapper.kernel["source"]["support"][i] == pair_kernels.support_threshold(f,settings["confidence"])
        torch.testing.assert_close(mapper.kernel["source"]["omega"],baseline.kernel["source"]["omega"],rtol=0,atol=0)
        sid,tid = next(iter(mapper.pairs))
        i,j = mapper.source_index[sid],mapper.target_index[tid]
        fq = mapper._partial_query(query,i)
        candidate,_,diag,_ = mapper._partial_projection(query,i,j,sid,tid,fq)
        (q,r,g),_ = mapper._factor_cache[sid,tid]
        plan = (q/g)@r.T
        kquery,ky = fq@mapper.fx[i].T,mapper.fy[j]@mapper.fy[j].T
        expected = normalize_rows((kquery@plan@ky.T)/ky.mean(0))@mapper.y[j]
        torch.testing.assert_close(candidate,expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(diag["raw_confidence"],(kquery@plan.sum(1))/kquery.mean(1),atol=1e-12,rtol=1e-12)
    else:
        state = torch.load(directory/"factors.pt",weights_only=True)
        q,r,g = (state[k].double() for k in ("q","r","g"))
        saved = torch.load(directory/"kernels.pt",weights_only=True)
        fx = kernels.features(mapper.x.flatten(0,1),kernels.projection_state(saved["source"],.5))
        fy = mapper.target_kernel_features.flatten(0,1)
        kquery = kernels.features(query,mapper.kernel_source)@fx.T
        ky,plan = fy@fy.T,(q/g)@r.T
        scores = (kquery@plan@ky.T)/ky.mean(0)
        j = int(mapper.image.conditional_weights(query.flatten()[None])[0].argmax())
        expected = normalize_rows(scores[:,j*4:(j+1)*4])@mapper.y[j]
        actual = mapped(mapper,banks[2],return_metadata=True).mapped_features[0]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    output = tmp_path/"mapped"
    result,manifest = mapper.project_bank(banks[2],output,chunk_size=1)
    again = mapped(mapper,banks[2],return_metadata=True,chunk_size=2)
    torch.testing.assert_close(result.mapped_features,again.mapped_features,atol=1e-12,rtol=0)
    assert manifest["projection"] == settings
    if partial:
        report = json.loads(next((output/"logs").glob("*/projection_kernel_quality.json")).read_text())
        assert report["accepted"] and report["patch_projection_h"] == 2.
    else:
        assert not list((output/"logs").glob("*/projection_kernel_quality.json"))
    assert hashes(directory) == before


@pytest.mark.parametrize("partial", [False, True])
def test_lowrank_topk_only_uses_original_bandwidth_without_audit(tmp_path,banks,monkeypatch,partial):
    directory = fit(partial_config(k=None) if partial else tiny_config(),root=tmp_path)
    base = FeatureMapper.load(directory)
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(kernels,"error_report",lambda *a,**k:pytest.fail("unnecessary bandwidth audit"))
    for k in (0,999):
        settings = infoot_test.projection_settings(base.config,None,top_k_images=k)
        result = mapped(FeatureMapper.load(directory,projection=settings),banks[2],return_metadata=True)
        torch.testing.assert_close(result.mapped_features,mapped(base,banks[2],return_metadata=True).mapped_features,rtol=0,atol=0)


def test_partial_failed_runtime_kernel_audit_still_warns_and_logs(tmp_path,banks,monkeypatch):
    c = partial_config()
    c["kernel"].update(h=4.)
    directory = fit(c,root=tmp_path)
    before = hashes(directory)
    forbid_fitting(monkeypatch)
    original = kernels.error_report
    def bad_audit(*args,**kwargs):
        return dict(original(*args,**kwargs),relative_rmse=99.)
    monkeypatch.setattr(kernels,"error_report",bad_audit)
    output = tmp_path/"warn-test"
    with pytest.warns(UserWarning,match="Projection kernel approximation"):
        assert infoot_test.main(["--mapping",str(directory),"--query-bank",str(banks[2].path),
            "--output-dir",str(output),"--projection-bandwidth",".35","--device","cpu"]) == 0
    report = json.loads(next((output/"logs").glob("*/projection_kernel_quality.json")).read_text())
    assert not report["accepted"] and report["approximation"]["source:0"]["relative_rmse"] == 99.
    assert (output/"mapped.pt").exists() and hashes(directory) == before


def test_balanced_cli_point_two_skips_projection_audit_without_changing_fit(tmp_path,banks,monkeypatch):
    c = tiny_config()
    c["kernel"]["h"] = c["image_solver"]["h"]
    directory = fit(c,root=tmp_path)
    before = hashes(directory)
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(kernels,"error_report",lambda *a,**k:pytest.fail("Projection accuracy audit"))
    output = tmp_path/"mapped-test"
    assert infoot_test.main(["--mapping",str(directory),"--query-bank",str(banks[2].path),
        "--output-dir",str(output),"--projection-bandwidth",".2","--device","cpu"]) == 0
    result,_,_ = load_mapped(output)
    assert torch.isfinite(result.mapped_features).all()
    for row in result.diagnostics["queries"]:
        assert row["image_projection_h"] == pytest.approx(.2)
        assert row["patch_projection_h"] == pytest.approx(.2)
    assert not list(output.rglob("projection_kernel_quality.json"))
    assert hashes(directory) == before


def test_partial_cli_records_effective_projection_and_topk(tmp_path,banks,capsys):
    directory = fit_mapping(config("grouped_partial"),root=tmp_path)
    output = tmp_path/"test"
    args = ["--mapping",str(directory),"--query-bank",str(banks[2].path),"--output-dir",str(output),
            "--projection-bandwidth",".2","--top-k-images","1","--device","cpu"]
    assert infoot_test.main(args) == 0
    result,_,manifest = load_mapped(output)
    assert manifest["projection"]["top_k_images"] == 1
    assert all(r["image_projection_h"] == pytest.approx(.2) for r in result.diagnostics["queries"])
    assert all(r["patch_projection_h"] == pytest.approx(.2) for r in result.diagnostics["queries"])
    capsys.readouterr()
    assert infoot_test.main(args+["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["projection"] == manifest["projection"]
