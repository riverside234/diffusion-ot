"""Router failure evidence, independent solver controls and partial-mask tuning."""
import importlib
import json

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, config, cpu_threads
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.mapping import MappingResult, load_mapped, mapping_summary
from infoot_vit.infoot_helper.partial import distance, kernel_state, solve_partial
from infoot_vit.infoot_helper.partial_batch import solve_partial_batch


def test_router_budget_stop_is_reported_before_partial_fitting(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    monkeypatch.setattr(module, "solve_partial", lambda *a, **k: pytest.fail("pair fit before router converged"))
    c = config("grouped_partial")
    c["solver"].update(lam=.2, max_outer_steps=1)
    with pytest.raises(RuntimeError, match="image FusedInfoOT max_outer_steps") as error:
        fit_mapping(c, root=tmp_path)
    assert "plan_delta_l1=" in str(error.value) and "Patch-pair fitting has not started" in str(error.value)
    assert "partial.solver" in str(error.value) and "without --resume" in str(error.value)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    report = json.loads((directory / "image_report.json").read_text())
    attempt = next((directory / "logs").iterdir())
    assert report == json.loads((attempt / "image_report.json").read_text())
    assert report["status"] == "max_outer_steps" and "plan" not in report
    last = report["history"][-1]
    assert last["inner"]["error"] <= c["solver"].get("inner_tolerance", 1e-10)
    assert last["plan_delta_l1"] > report["config"]["outer_tolerance"]
    assert (directory / "plans/image/latest.pt").exists()
    assert (directory / "plans/image.pt").exists()
    assert "image" not in json.loads((directory / "manifest.json").read_text())["models"]
    assert not (directory / "pairs.jsonl").exists()
    assert "image FusedInfoOT 1/1" in (attempt / "stdout.log").read_text()


def test_cli_solver_overrides_do_not_cross_between_stages(tmp_path, monkeypatch):
    from infoot_vit import infoot_fit
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(config("grouped_partial")))
    seen = []
    monkeypatch.setattr(infoot_fit, "fit_mapping", lambda c, **k: seen.append(validate_config(c)) or tmp_path)
    assert infoot_fit.main(["--config", str(path), "--h", ".4", "--reg", ".075", "--lam", ".075",
        "--max-outer-steps", "1200", "--partial-h", ".45", "--partial-reg", ".08", "--partial-lam", ".05",
        "--partial-max-steps", "600", "--keep-mass", ".75"]) == 0
    image, partial = seen[0]["solver"], seen[0]["partial"]["solver"]
    assert (image["h"], image["reg"], image["lam"], image["max_outer_steps"]) == (.4, .075, .075, 1200)
    assert (partial["h"], partial["reg"], partial["lam"], partial["max_outer_steps"]) == (.45, .08, .05, 600)
    assert seen[0]["partial"]["keep_mass"] == .75


def test_revised_partial_recipe_matches_serial_and_preserves_constraints():
    from infoot_vit import infoot_fit
    c = yaml.safe_load((infoot_fit.ROOT / "infoot_vit/configs/grouped_partial.yaml").read_text())
    pc, mass = c["partial"]["solver"], c["partial"]["keep_mass"]
    g = torch.Generator().manual_seed(42)
    x = torch.randn(4, 32, 16, generator=g, dtype=torch.float64)
    y = torch.randn(4, 32, 16, generator=g, dtype=torch.float64) + .3
    cost = torch.stack([distance(a, b) for a, b in zip(x, y)])
    kx = torch.stack([kernel_state(a, pc["h"])[0] for a in x])
    ky = torch.stack([kernel_state(b, pc["h"])[0] for b in y])
    plans, reports = solve_partial_batch(cost, kx, ky, keep_mass=mass, config=pc)
    assert torch.isfinite(plans).all() and (plans >= 0).all()
    assert all(r["status"] == "converged" for r in reports)
    for i, report in enumerate(reports):
        serial, reference = solve_partial(cost[i], kx[i], ky[i], keep_mass=mass, config=pc)
        assert reference["status"] == "converged"
        torch.testing.assert_close(plans[i], serial, atol=1e-10, rtol=1e-7)
        assert report["history"][-1]["objective"] == pytest.approx(reference["history"][-1]["objective"], abs=1e-9)
        assert report["mass_error"] <= pc["mass_tolerance"]
        assert max(report["row_cap_error"], report["column_cap_error"]) <= pc["feasibility_tolerance"]
        assert float(plans[i].sum()) == pytest.approx(mass, abs=pc["mass_tolerance"])


def test_counterfactual_threshold_coverage_does_not_change_mapping():
    confidence = torch.tensor([[0., .07, .11, .20], [.01, .03, .09, .29]], dtype=torch.float64)
    features = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
    mask = confidence >= .10
    result = MappingResult(features, confidence, mask, dict(mode="grouped_partial", seconds=1., queries=[]))
    report = mapping_summary(result, features, features, features,
                             confidence_settings=dict(threshold=.10, mass_floor=1e-12))
    rows = report["confidence_threshold_sweep"]["coverage"]
    assert [r["valid_fraction"] for r in rows] == [5/8, 3/8, 2/8, 0.]
    assert [r["all_invalid_images"] for r in rows] == [0, 0, 0, 2]
    assert report["valid_fraction"] == 3/8
    assert torch.equal(result.valid_mask, mask) and torch.equal(result.mapped_features, features)


def test_threshold_cli_reuses_fit_and_records_override(tmp_path, banks, monkeypatch, capsys):
    from infoot_vit import infoot_test
    c = config("grouped_partial", mass=.75)
    c["projection"]["confidence"]["all_invalid_policy"] = "bypass"  # Explicit all-invalid diagnostic control.
    directory = fit_mapping(c, root=tmp_path)
    original = (directory / "manifest.json").read_bytes()
    monkeypatch.setattr("infoot_vit.infoot_helper.conditional.BalancedModel.fit",
                        lambda *a, **k: pytest.fail("mapping must not fit"))
    results = []
    for threshold in (0., .99):
        output = tmp_path / f"threshold-{threshold}"
        args = ["--mapping", str(directory), "--query-bank", str(banks[2].path), "--output-dir", str(output),
                "--confidence-threshold", str(threshold), "--threads", "1"]
        assert infoot_test.main(args) == 0
        result, _, manifest = load_mapped(output)
        assert manifest["projection"]["confidence"]["threshold"] == threshold
        assert manifest["mapper_id"] == json.loads(original)["artifact_id"]
        report = json.loads(next((output / "logs").glob("*/mapping_report.json")).read_text())
        assert report["confidence_threshold_sweep"]["active_threshold"] == threshold
        results.append(result)
    torch.testing.assert_close(results[0].mapped_features, results[1].mapped_features)
    torch.testing.assert_close(results[0].match_confidence, results[1].match_confidence)
    assert int(results[1].valid_mask.sum()) < int(results[0].valid_mask.sum())
    assert (directory / "manifest.json").read_bytes() == original
    capsys.readouterr()
    assert infoot_test.main(["--mapping", str(directory), "--query-bank", str(banks[2].path),
                            "--dry-run", "--confidence-threshold", ".2"]) == 0
    assert json.loads(capsys.readouterr().out)["projection"]["confidence"]["threshold"] == .2


@pytest.mark.parametrize("threshold", ["-1", "nan", "inf", "1.01"])
def test_invalid_threshold_rejected_before_mapping(tmp_path, threshold):
    from infoot_vit.infoot_test import main
    with pytest.raises(SystemExit):
        main(["--mapping", str(tmp_path / "missing"), "--confidence-threshold", threshold])


def test_balanced_mapping_rejects_irrelevant_threshold_override():
    from infoot_vit.infoot_test import projection_settings
    with pytest.raises(ValueError, match="only to partial"):
        projection_settings(config("grouped_patch"), .1)
