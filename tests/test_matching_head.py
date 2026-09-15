from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from diffusion_ot.models.matching_head import (
    ResidualMatchingHead, load_matching_head, make_matching_head, matching_head_id, matching_geometry_diagnostics,
)
from diffusion_ot.losses.conditional_structure import conditional_structure_loss
from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.losses.infoot import conditional_variance_decomposition


def test_spectrum_diagnostics_distinguish_spread_from_dimensional_collapse():
    raw = torch.cat([torch.eye(4), -torch.eye(4)])
    rank_one = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]]).repeat(4, 1)
    result = matching_geometry_diagnostics(raw, rank_one)
    assert result["matching_variance"] == pytest.approx(result["raw_normalized_variance"])
    for statistic in ("effective_rank", "participation_rank"):
        assert result[f"raw_normalized_covariance_{statistic}"] == pytest.approx(4)
        assert result[f"matching_covariance_{statistic}"] == pytest.approx(1)
    assert result["matching_covariance_top_eigenvalue_fraction"] == pytest.approx(1)
    assert result["raw_normalized_covariance_top_eigenvalue_fraction"] == pytest.approx(.25)
    collapsed = matching_geometry_diagnostics(raw, torch.ones_like(raw))
    assert collapsed["matching_covariance_effective_rank"] == 0
    assert collapsed["matching_covariance_participation_rank"] == 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = matching_geometry_diagnostics(raw, rank_one)
    assert mixed == pytest.approx(result)


def test_matching_head_starts_at_identity_without_changing_sampling_rng():
    torch.manual_seed(917)
    codes = torch.randn(9, 6)
    before = torch.get_rng_state().clone()
    head = make_matching_head({"input_dim": 6, "hidden_dim": 8}, device="cpu", seed=400)
    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.testing.assert_close(head(codes), F.normalize(codes, dim=1))
    assert not torch.equal(head.residual[1].weight, torch.zeros_like(head.residual[1].weight))


def test_checkpoint_requires_matching_raw_or_ema_head_and_preserves_weights():
    head = ResidualMatchingHead(4, 7)
    with torch.no_grad():
        head.residual[-1].weight.normal_()
    checkpoint = {"config": {"matching_head": {"enabled": True, "input_dim": 4, "hidden_dim": 7}},
                  "matching_heads": {"cat": head.state_dict()},
                  "matching_head_ema": {"cat": deepcopy(head.state_dict())}}
    loaded = load_matching_head(checkpoint, "cat", weights="ema", device="cpu")
    codes = torch.randn(11, 4)
    torch.testing.assert_close(loaded(codes), head(codes))
    assert matching_head_id(loaded) == matching_head_id(head)
    del checkpoint["matching_head_ema"]
    with pytest.raises(ValueError, match="cannot silently"):
        load_matching_head(checkpoint, "cat", weights="ema", device="cpu")
    with pytest.raises(ValueError, match="matching_head"):
        validate_prior_resume({}, checkpoint["config"])
    assert load_matching_head({"config": {}}, "cat", weights="raw", device="cpu") is None


def test_conditional_geometry_is_independent_of_decoder_values_when_explicit():
    torch.manual_seed(918)
    references = {d: torch.randn(6, 8, requires_grad=True) for d in ("cat", "dog")}
    queries = {d: torch.randn(3, 8, requires_grad=True) for d in references}
    ref_features = {d: torch.randn(6, 4, requires_grad=True) for d in references}
    query_features = {d: torch.randn(3, 4, requires_grad=True) for d in references}
    ref_structure = {d: torch.randn(6, 3) for d in references}
    query_structure = {d: torch.randn(3, 3) for d in references}
    plan = torch.eye(6) / 6
    def evaluate(values):
        return conditional_structure_loss(values, queries, ref_structure, query_structure, plan,
                                          bandwidth=.3, cost_scale=1., reference_matching=ref_features,
                                          query_matching=query_features)
    result = evaluate(references)
    changed = evaluate({d: 4 * z + 2 for d, z in references.items()})
    torch.testing.assert_close(result.loss, changed.loss)
    for direction in result.weights:
        torch.testing.assert_close(result.weights[direction], changed.weights[direction])
    result.loss.backward()
    assert all(x.grad is not None and x.grad.norm() > 0 for x in (*ref_features.values(), *query_features.values()))
    assert all(x.grad is None for x in (*references.values(), *queries.values()))


def test_matching_head_learns_conditional_structure_with_fixed_decoder_codes():
    """Mechanism check with a fixed known plan; re-solving cannot undo the test.

    Structure is present in raw codes but nuisance coordinates dominate their
    original distances. The heads can learn the readout while values stay fixed.
    This synthetic fitting test is not an AFHQ generalization claim.
    """
    torch.manual_seed(920)
    ref_labels = torch.arange(24) % 3
    query_labels = torch.arange(12) % 3
    ref_structure = {d: F.one_hot(ref_labels, 3).float() for d in ("cat", "dog")}
    query_structure = {d: F.one_hot(query_labels, 3).float() for d in ref_structure}
    references = {d: torch.cat((s, 2 * torch.randn(len(s), 5)), 1) for d, s in ref_structure.items()}
    queries = {d: torch.cat((s, 2 * torch.randn(len(s), 5)), 1) for d, s in query_structure.items()}
    original = {d: z.clone() for d, z in references.items()}
    plan = (ref_labels[:, None] == ref_labels[None, :]).float() / (24 * 8)
    heads = {d: ResidualMatchingHead(8, 24) for d in references}
    optimizer = torch.optim.Adam([p for h in heads.values() for p in h.parameters()], lr=.01)
    def objective():
        return conditional_structure_loss(
            references, queries, ref_structure, query_structure, plan, bandwidth=.25,
            cost_scale=1., teacher_temperature=.15,
            reference_matching={d: heads[d](x) for d, x in references.items()},
            query_matching={d: heads[d](x) for d, x in queries.items()},
        )
    initial = objective()
    for _ in range(80):
        optimizer.zero_grad()
        objective().loss.backward()
        optimizer.step()
    final = objective()
    assert final.loss < .5 * initial.loss, (float(initial.loss.detach()), float(final.loss.detach()))
    for direction in final.metrics:
        assert final.metrics[direction]["expected_structure_cost"] < initial.metrics[direction]["expected_structure_cost"]
    for d in references:
        torch.testing.assert_close(references[d], original[d])


def test_banks_reject_changed_matching_geometry_and_keep_raw_distance_controls(tmp_path):
    from diffusion_ot.evaluation.stage1b_eval import (
        LatentBank, _direction_projection_evaluation, load_latent_bank,
        save_latent_bank, validate_bank_compatibility,
    )
    raw = torch.eye(3)
    # Different dimensions make mixing matching and raw spaces an immediate error.
    geometry = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    def bank(domain, split):
        return LatentBank(domain, split, raw, geometry, [f"{domain}_{split}_{i}" for i in range(3)],
                          [{}] * 3, f"{domain}_checkpoint", matching_id="head_a")
    source, query, target = bank("cat", "train"), bank("cat", "val"), bank("dog", "train")
    changed = deepcopy(query)
    changed.matching_id = "head_b"
    with pytest.raises(ValueError, match="matching heads"):
        validate_bank_compatibility(source, changed)
    save_latent_bank(tmp_path / "bank.pt", target)
    assert load_latent_bank(tmp_path / "bank.pt").matching_id == "head_a"
    report, tensors = _direction_projection_evaluation(
        source, target, query, torch.eye(3)/3, source_scale=1., target_scale=1.,
        bandwidth=.01, eps=1e-8,
    )
    assert report["mean_nearest_target_distance"] < 1e-6
    torch.testing.assert_close(tensors["conditional_codes"], raw)
    torch.testing.assert_close(tensors["barycentric_codes"], raw)


def test_conditional_means_can_be_correct_without_target_variance_recovery():
    codes = torch.tensor([[-2.], [2.]], dtype=torch.float64)
    uniform = conditional_variance_decomposition(torch.full((3, 2), .5), codes)
    assert uniform["mixture_variance"] == pytest.approx(4)
    assert uniform["mean_variance"] == 0  # Correct conditional mean is constant.
    assert uniform["conditional_variance_fraction"] == pytest.approx(1)
    deterministic = conditional_variance_decomposition(torch.eye(2), codes)
    assert deterministic["mean_variance_fraction"] == pytest.approx(1)
    assert deterministic["conditional_variance"] == 0
    torch.manual_seed(924)
    result = conditional_variance_decomposition(torch.randn(7, 5).double().softmax(1), torch.randn(5, 3).double())
    assert result["mixture_variance"] == pytest.approx(result["mean_variance"] + result["conditional_variance"], abs=1e-12)
