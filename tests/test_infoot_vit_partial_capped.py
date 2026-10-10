"""Saturated-support regression, exact dual blocks and serving-bandwidth selection."""
import importlib
import json
import math

import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads
from infoot_vit.infoot_helper.partial import solver_config, entropy_subproblem
from infoot_vit.infoot_helper.partial_batch import _capped_mass_potentials, entropy_subproblem_batch
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, resource_estimate
from infoot_vit.infoot_helper.mapping import FeatureMapper


@pytest.mark.parametrize("mass", [.01, .75, .999])
def test_capped_mass_projection_extreme_weights_and_nonuniform_capacities(mass):
    weights = torch.tensor([[1000., 0., -1000., 50.], [-200., -100., -50., -500.]], dtype=torch.float64)
    caps = torch.tensor([[.1, .2, .3, .4], [.5, .1, .3, .1]], dtype=torch.float64)
    u, w = _capped_mass_potentials(weights, caps, mass)
    marginal = (weights + u + w[:, None]).exp()
    assert (u <= 0).all() and torch.isfinite(w).all()
    torch.testing.assert_close(marginal.sum(-1), weights.new_full((2,), mass), atol=1e-13, rtol=1e-12)
    assert ((marginal-caps).clamp_min(0) < 1e-13).all()
    assert (u*(marginal-caps)).abs().max() < 1e-10  # KKT complementarity.


@pytest.mark.parametrize("cheap", [145, 146])
def test_saturated_196_patch_problem_matches_analytic_optimum(cheap):
    n, mass, reg = 196, .75, .075
    cost = torch.ones(1, n, n, dtype=torch.float64)
    cost[:, :cheap, :cheap] = .005
    a = b = cost.new_full((1, n), 1/n)
    c = solver_config(dict(reg=reg, max_inner_steps=2000))
    plan, report = entropy_subproblem_batch(a, b, cost, mass, c)
    assert report["ok"].all() and report["iterations"].max() < 1200
    assert report["error"].max() <= 1e-10
    assert report["relative_duality_gap"].max() <= 1e-10
    # Symmetry reduces the optimum to four constant blocks. Cheap rows and
    # columns saturate; other inequality potentials are zero. Solve for exp(u).
    rest, k = n-cheap, math.exp(-.995/reg)
    deficit = mass-cheap/n
    aa, bb, cc = n*deficit*cheap, rest*k*(n*deficit-cheap), -rest**2*k
    exp_u = (-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa)
    exp_w = (1/n)/(cheap*exp_u**2+rest*k*exp_u)
    expected = torch.full((n, n), k*exp_w, dtype=torch.float64)
    expected[:cheap, :] *= exp_u
    expected[:, :cheap] *= exp_u
    expected[:cheap, :cheap] /= k
    torch.testing.assert_close(plan[0], expected, atol=2e-13, rtol=1e-9)
    if cheap == 145:
        # Original POT three-set loop exhibits the same feasible budget stop
        # as the lab logs; do not turn that stop into a successful plan.
        with pytest.raises(RuntimeError, match="did not converge"):
            entropy_subproblem(a[0], b[0], cost[0], mass, dict(c, max_inner_steps=10000))


def test_1024_members_chunk_invariance_and_partial_final_chunk():
    rng = torch.Generator().manual_seed(42)
    cost = torch.rand(1025, 4, 5, dtype=torch.float64, generator=rng)
    a, b = cost.new_full((1025, 4), .25), cost.new_full((1025, 5), .2)
    c = solver_config(dict(reg=.2))
    full, report = entropy_subproblem_batch(a, b, cost, .8, c)
    assert report["ok"].all()
    pieces = []
    for start in [0, 1024]:
        part, info = entropy_subproblem_batch(a[start:start+1024], b[start:start+1024], cost[start:start+1024], .8, c)
        assert info["ok"].all()
        pieces.append(part)
    torch.testing.assert_close(full, torch.cat(pieces), atol=1e-13, rtol=1e-11)
    estimate = resource_estimate((2000,196,768), (2000,196,768), "grouped_partial", 8, 1024)
    assert estimate["estimated_working_gib"] < 48


def test_pair_selection_uses_saved_mapping_bandwidth(tmp_path, banks):
    c = dict(config("grouped_partial"), fit_pair_top_k=1, pair_batch_size=3)
    c["projection"]["bandwidth_multiplier"] = .5
    directory = fit_mapping(c, root=tmp_path)
    mapper = FeatureMapper.load(directory)
    selection = json.loads((directory / "pair_selection.json").read_text())
    assert selection["projection_h"] == mapper.image.h == .35
    assert selection["projection_bandwidth_multiplier"] == .5
    weights = mapper.image.conditional_weights(mapper.image.source)
    for i, row in enumerate(selection["rows"]):
        j = mapper.target_index[row["target_ids"][0]]
        assert row["retained_probability"] == pytest.approx(float(weights[i, j]), abs=1e-8)
        assert j == int(weights[i].argmax())
    report = json.loads((directory / "pair_selection_report.json").read_text())
    assert report["mean_retained_probability"] == pytest.approx(sum(r["retained_probability"] for r in selection["rows"])/2)
    assert (directory / "pair_batch_report.json").exists()


def test_failed_pair_reproducer_and_interrupted_summary_are_saved(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real = module.solve_partial_batch
    def failure(cost, kx, ky, **kwargs):
        callback = kwargs.pop("on_step")
        def interrupt_after_report(plans, reports):
            callback(plans, reports)
            raise KeyboardInterrupt()
        return real(cost, kx, ky, **(kwargs | dict(config=dict(reg=.001, lam=0., max_inner_steps=1))),
                    on_step=interrupt_after_report)
    monkeypatch.setattr(module, "solve_partial_batch", failure)
    c = dict(config("grouped_partial"), fit_pair_top_k=1, pair_batch_size=2)
    with pytest.raises(KeyboardInterrupt):
        fit_mapping(c, root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    attempt = next((directory / "logs").iterdir())
    example = torch.load(attempt / "first_failed_pair.pt", weights_only=True)
    assert example["cost"].dtype == example["kx"].dtype == example["ky"].dtype == torch.float64
    assert example["cost"].shape == (4, 4)
    assert torch.isfinite(example["cost"]).all()
    report = json.loads((directory / "pair_batch_report.json").read_text())
    assert report["status"] == "interrupted_or_failed" and report["failed_pairs"] > 0
    assert report == json.loads((attempt / "pair_batch_report.json").read_text())


def test_optional_reproducer_failure_does_not_cancel_successful_neighbors(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    original_solve, original_save = module.solve_partial_batch, module.save_tensor
    def solve(cost, *args, **kwargs):
        cost = cost.clone()
        cost[0, 0, 0] = float("nan")
        return original_solve(cost, *args, **kwargs)
    def save(path, value):
        if path.name == "first_failed_pair.pt":
            raise OSError("simulated diagnostic write failure")
        return original_save(path, value)
    monkeypatch.setattr(module, "solve_partial_batch", solve)
    monkeypatch.setattr(module, "save_tensor", save)
    c = dict(config("grouped_partial"), fit_pair_top_k=1, pair_batch_size=2)
    with pytest.raises(RuntimeError, match="1 partial pairs failed; 1 successful pairs saved"):
        fit_mapping(c, root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    assert len((directory / "pairs.jsonl").read_text().splitlines()) == 1
    assert len((directory / "pair_failures.jsonl").read_text().splitlines()) == 1
    assert "failure_reproducer_save_failed" in next((directory / "logs").glob("*/events.jsonl")).read_text()
