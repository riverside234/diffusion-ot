"""Explicit successful-pair inference; failed fitting artifacts remain immutable."""
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, cpu_threads, mapped
from test_infoot_vit_partial_batch import batched_config
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional import BalancedModel, normalize_rows
from infoot_vit.infoot_helper.feature_bank import file_hash, write_json, save_tensor
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped
from infoot_vit.infoot_helper.mapping_manifest import load_mapping_manifest
from infoot_vit.infoot_helper.partial import distance


@pytest.fixture
def failed_fit(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    original, calls = module.solve_partial_batch, 0
    def fail_one(cost, kx, ky, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            cost = cost.clone()
            cost[0,0,0] = float("nan")
        return original(cost,kx,ky,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module,"solve_partial_batch",fail_one)
        with pytest.raises(RuntimeError,match="1 partial pairs failed; 3 successful pairs saved"):
            fit_mapping(batched_config(),root=tmp_path)
    return next((tmp_path/"outputs/infoot_vit").iterdir())


def hashes(directory):
    return {str(p.relative_to(directory)):file_hash(p) for p in directory.rglob("*") if p.is_file()}


def test_success_subset_matches_direct_projection_and_separates_discarded_mass(failed_fit,banks,monkeypatch):
    before = hashes(failed_fit)
    with pytest.raises(ValueError,match="incomplete"):
        FeatureMapper.load(failed_fit)
    monkeypatch.setattr(BalancedModel,"fit",lambda *a,**k:pytest.fail("inference must not refit"))
    m = load_mapping_manifest(failed_fit,allow_failed_pairs=True)
    settings = infoot_test.projection_settings(m["config"],None,bandwidth=.35)
    mapper = FeatureMapper.load(failed_fit,allow_failed_pairs=True,projection=settings)
    assert mapper.manifest["status"] == "failed" and len(mapper.pairs) == 3
    assert mapper.failed_pair_mask.sum() == 1
    result = mapped(mapper,banks[2],return_metadata=True,chunk_size=1)
    again = mapped(mapper,banks[2],return_metadata=True,chunk_size=2)
    torch.testing.assert_close(result.mapped_features,again.mapped_features,rtol=0,atol=0)
    for query, actual, confidence, row in zip(banks[2].features,result.mapped_features,result.match_confidence,result.diagnostics["queries"]):
        raw = mapper.image.pair_weights(query.flatten()[None])[0]
        retained = raw[mapper.pair_mask].sum()
        theta = raw.masked_fill(~mapper.pair_mask,0)/retained
        numerator, expected_confidence = torch.zeros_like(query),torch.zeros_like(confidence)
        for sid,tid in mapper.pairs:
            i,j = mapper.source_index[sid],mapper.target_index[tid]
            plan = mapper._pair(sid,tid)["plan"]
            h = mapper.shared["h_projection"]
            kq = (-.5*(distance(query,mapper.x[i])/(h*mapper.shared["sx"][i])).square()).exp()
            ky = mapper.projection_ky[j]
            candidate = normalize_rows((kq@plan@ky.T)/ky.mean(0))@mapper.y[j]
            g = ((kq@plan.sum(1))/kq.mean(1)).clamp(0,1)
            numerator += theta[i,j]*g[:,None]*candidate
            expected_confidence += theta[i,j]*g
        torch.testing.assert_close(actual,numerator/expected_confidence[:,None],atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(confidence,expected_confidence,atol=1e-11,rtol=1e-11)
        assert row["failed_pair_discarded_routing_mass"] == pytest.approx(float(raw[mapper.failed_pair_mask].sum()))
        assert row["successful_pair_retained_routing_mass"] == pytest.approx(float(retained))
        assert sum(row[k] for k in ("fit_pair_discarded_routing_mass","failed_pair_discarded_routing_mass","successful_pair_retained_routing_mass")) == pytest.approx(1.)
        assert row["routing_renormalization"] == pytest.approx(1/float(retained))
        assert sum(row["image_weights"]) == pytest.approx(1.)
        torch.testing.assert_close(confidence+torch.tensor(row["ot_rejected_mass"]),torch.ones_like(confidence))
    assert hashes(failed_fit) == before


def test_no_successful_route_errors_even_with_mask_bypass(failed_fit,banks,monkeypatch):
    mapper = FeatureMapper.load(failed_fit,allow_failed_pairs=True)
    mapper.config["projection"]["confidence"]["all_invalid_policy"] = "bypass"
    theta = mapper.failed_pair_mask.double()[None]
    monkeypatch.setattr(mapper.image,"pair_weights",lambda *a:theta)
    monkeypatch.setattr(mapper.image,"conditional_weights",lambda *a:theta.sum(1))
    with pytest.raises(ValueError,match="No usable saved-pair routing mass.*failed_pair_discarded_routing_mass=1"):
        mapped(mapper,banks[2],return_metadata=True)


@pytest.mark.parametrize("kind",["missing","bytes","status","constraints"])
def test_registered_success_cannot_be_silently_skipped(failed_fit,kind):
    index = failed_fit/"pairs.jsonl"
    entries = [json.loads(line) for line in index.read_text().splitlines()]
    path = failed_fit/entries[0]["file"]
    if kind == "missing":
        path.unlink()
    elif kind == "bytes":
        path.write_bytes(b"corrupt")
    else:
        state = torch.load(path,weights_only=True)
        if kind == "status":
            state["report"]["status"] = "max_outer_steps"
        else:
            state["plan"] *= .5
            state["r"],state["c"] = state["plan"].double().sum(1),state["plan"].double().sum(0)
        save_tensor(path,state)
        entries[0]["sha256"] = file_hash(path)
        index.write_text("".join(json.dumps(row)+"\n" for row in entries))
        m = json.loads((failed_fit/"manifest.json").read_text())
        m["pair_inventory"]["sha256"] = file_hash(index)
        write_json(failed_fit/"manifest.json",m)
    with pytest.raises(ValueError,match="changed artifact|identity/status|tolerance"):
        FeatureMapper.load(failed_fit,allow_failed_pairs=True)


@pytest.mark.parametrize("kind",["missing_failure","journal_mismatch","active_fit","config_change"])
def test_unknown_missing_pairs_or_changed_fit_are_rejected(failed_fit,kind):
    path = failed_fit/"manifest.json"
    m = json.loads(path.read_text())
    if kind in {"missing_failure","journal_mismatch"}:
        report_path = failed_fit/"pair_batch_report.json"
        report = json.loads(report_path.read_text())
        if kind == "missing_failure":
            report["failures"] = []
        else:
            report["failures"][0]["error"] = "different failure"
        write_json(report_path,report)
    elif kind == "active_fit":
        m["status"] = "fitting"
        write_json(path,m)
    else:
        m["config"]["partial"]["keep_mass"] = .2
        write_json(path,m)
    with pytest.raises(ValueError,match="failed-pair report|failure journal|stopped|fingerprint"):
        FeatureMapper.load(failed_fit,allow_failed_pairs=True)


def test_subset_does_not_break_resume_or_skip_historical_failures(failed_fit,tmp_path,banks):
    subset = FeatureMapper.load(failed_fit,allow_failed_pairs=True)
    successes = {e["file"]:file_hash(failed_fit/e["file"]) for e in subset.pairs.values()}
    fit_mapping(batched_config(),root=tmp_path,resume=failed_fit)
    complete = FeatureMapper.load(failed_fit,allow_failed_pairs=True)
    assert complete.manifest["status"] == "complete" and len(complete.pairs) == 4
    assert "incomplete_fit" not in complete.manifest
    assert complete.manifest["artifact_id"] != subset.manifest["artifact_id"]
    assert (failed_fit/"pair_failures.jsonl").exists()
    assert all(file_hash(failed_fit/p) == sha for p,sha in successes.items())


def test_cli_generation_and_dry_run_record_subset_identity(failed_fit,tmp_path,banks,monkeypatch,capsys):
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    before = hashes(failed_fit)
    checkpoint = tmp_path/"fake.pt"; checkpoint.write_bytes(b"fixed test checkpoint")
    cfg = tmp_path/"dog.yaml"
    cfg.write_text(yaml.safe_dump(dict(domain="dog",data_config="unused.yaml",class_conditioning=dict(null_label=1000))))
    encoder = SimpleNamespace(snapshot_identity=banks[1].representation["encoder"],
        architecture_spec=dict(features=banks[1].representation["layer"]))
    evaluator = SimpleNamespace(branch=SimpleNamespace(encoder=encoder),transformer=None,vae=None,
        device="cpu",model_dtype=torch.float32,checkpoint_path=checkpoint)
    monkeypatch.setattr(stage1a_eval,"load_stage1a_evaluator",lambda *a,**k:evaluator)
    observed = {}
    def sample(branch,transformer,noise,tokens,**kwargs):
        observed.update(tokens=tokens,**kwargs)
        return noise
    monkeypatch.setattr(stage1a_eval,"integrate_pdae_flow",sample)
    monkeypatch.setattr(stage1a_eval,"decode_vae_latents",lambda vae,z,**kw:torch.zeros(len(z),3,8,8))
    monkeypatch.setattr(ground_truth,"load_ground_truth_images",lambda cfg,records:torch.full((len(records),3,8,8),.5))
    output = tmp_path/"test-output"
    args = ["--mapping",str(failed_fit),"--query-bank",str(banks[2].path),"--output-dir",str(output),
        "--allow-failed-pairs","--projection-bandwidth",".35","--top-k-images","0","--generate",
        "--train-config",str(cfg),"--device","cpu","--threads","1"]
    capsys.readouterr()
    assert infoot_test.main(args+["--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["incomplete_fit"]["successful_pair_count"] == 3 and not output.exists()
    assert infoot_test.main(args) == 0
    result,_,m = load_mapped(output)
    report = json.loads((output/"generation_report.json").read_text())
    snapshot = json.loads(next((output/"logs").glob("*/incomplete_fit_snapshot.json")).read_text())
    stats = json.loads(next((output/"logs").glob("*/mapping_report.json")).read_text())
    assert report["incomplete_fit"] == m["incomplete_fit"] == dry["incomplete_fit"]
    assert report["mapper_id"] == dry["mapper_id"] == snapshot["artifact_id"]
    assert snapshot["status"] == "failed" and m["incomplete_fit"]["failed_pair_count"] == 1
    assert stats["failed_pair_discarded_routing_mass"]["mean"] > 0
    assert set(report["failed_pair_discarded_routing_mass"]) == set(banks[2].ids)
    torch.testing.assert_close(observed["tokens"],result.mapped_features.float())
    assert torch.equal(observed["condition_padding_mask"],~result.valid_mask)
    assert report["num_steps"] == 40 and report["guidance_scale"] == 2.
    assert (output/"translation_grid.png").exists()
    assert hashes(failed_fit) == before
