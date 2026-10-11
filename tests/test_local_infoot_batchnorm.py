from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_local_infoot_pipeline import local_helpers


def feature_batch(references, queries):
    return {"references": {"v": references}, "queries": {"v": queries}}


def test_training_statistics_use_only_references_and_keep_raw_codes(local_helpers):
    from infoot_helper.batchnorm_matching import add_matching_features

    references = torch.tensor([[1., 3.], [2., 7.], [4., 9.]], dtype=torch.float64)
    queries = torch.tensor([[100., -30.], [200., 50.]], dtype=torch.float64)
    encoded = {"cat": feature_batch(references, queries),
               "dog": feature_batch(references * 4 + 20, queries * 4 + 20)}
    norms = {name: torch.nn.BatchNorm1d(2, momentum=.3).double() for name in encoded}
    for norm in norms.values():
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2., .5]))
            norm.bias.copy_(torch.tensor([.3, -.7]))
    original_norms = deepcopy(norms)
    add_matching_features(encoded, norms)

    for name, batch in encoded.items():
        raw = batch["references"]["v"]
        norm = norms[name]
        mean, variance = raw.mean(0), raw.var(0, unbiased=False)
        for split in ("references", "queries"):
            expected = (batch[split]["v"] - mean) / (variance + norm.eps).sqrt()
            expected = expected * norm.weight + norm.bias
            torch.testing.assert_close(batch[split]["m"], expected)
        torch.testing.assert_close(norm.running_mean, .3 * mean)
        torch.testing.assert_close(norm.running_var, .7 + .3 * raw.var(0, unbiased=True))
        assert norm.num_batches_tracked == 1
    assert encoded["cat"]["references"]["v"] is references
    assert encoded["cat"]["queries"]["v"] is queries
    torch.testing.assert_close(encoded["cat"]["m"], torch.cat([
        encoded["cat"]["queries"]["m"], encoded["cat"]["references"]["m"],
    ]))

    changed_queries = {name: feature_batch(batch["references"]["v"], queries[:1] * 1000)
                       for name, batch in encoded.items()}
    add_matching_features(changed_queries, original_norms)
    for name in norms:
        torch.testing.assert_close(norms[name].state_dict(), original_norms[name].state_dict())
        torch.testing.assert_close(encoded[name]["references"]["m"],
                                   changed_queries[name]["references"]["m"])


def test_query_normalization_retains_gradients_through_reference_statistics(local_helpers):
    from infoot_helper.batchnorm_matching import add_matching_features

    generator = torch.Generator().manual_seed(19)
    references = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    queries = torch.randn(2, 3, generator=generator, dtype=torch.float64, requires_grad=True)

    def normalize(refs, query):
        encoded = {"cat": feature_batch(refs, query)}
        add_matching_features(encoded, {"cat": torch.nn.BatchNorm1d(3, affine=False).double()})
        return encoded["cat"]["queries"]["m"]

    assert torch.autograd.gradcheck(normalize, (references, queries))
    normalize(references, queries).square().sum().backward()
    for raw in (references, queries):
        assert torch.isfinite(raw.grad).all() and raw.grad.norm() > 0


def test_contrastive_features_reuse_target_reference_statistics(local_helpers, monkeypatch):
    from infoot_helper import translation_contrastive as helper
    from infoot_helper.batchnorm_matching import add_matching_features

    reference_v = torch.tensor([[1., 3.], [2., 7.], [4., 9.]], requires_grad=True)
    norm = torch.nn.BatchNorm1d(2)
    norm(reference_v)
    source = {"cat": feature_batch(reference_v * 2 + 10, torch.tensor([[20., 5.], [30., 8.]]))}
    add_matching_features(source, {"cat": torch.nn.BatchNorm1d(2)})
    source = source["cat"]
    source["queries"]["x0"] = torch.zeros(2, 2)
    mapped = torch.tensor([[3., 5.], [5., 11.]], requires_grad=True)
    context = SimpleNamespace(device="cpu", model_dtype=torch.float32, vae=None, transformer=None,
                              branch=SimpleNamespace(encode=lambda x: x),
                              training_config={"class_conditioning": {"null_label": None}})
    monkeypatch.setattr(helper, "integrate_training_flow", lambda *args, **kwargs: args[3])
    monkeypatch.setattr(helper, "decode_training_images", lambda vae, x: x)
    monkeypatch.setattr(helper, "encode_generated_images", lambda vae, x: x)

    def check_contrastive(recovered, positive, bank, **kwargs):
        expected = (mapped - reference_v.mean(0)) / (reference_v.var(0, unbiased=False) + norm.eps).sqrt()
        torch.testing.assert_close(recovered, expected * norm.weight + norm.bias)
        torch.testing.assert_close(positive, source["queries"]["m"])
        torch.testing.assert_close(bank, source["m"])
        return recovered.square().mean(), {}

    monkeypatch.setattr(helper, "source_code_contrastive_loss", check_contrastive)
    loss, _ = helper.translation_contrastive_loss(context, source, mapped, batch_norm=norm, reference_v=reference_v)
    loss.backward()
    assert norm.num_batches_tracked == 1
    for value in (mapped, reference_v, norm.weight):
        assert torch.isfinite(value.grad).all() and value.grad.norm() > 0


def test_alignment_and_mapping_train_encoders_and_affine_batchnorm(local_helpers):
    from infoot_helper.batchnorm_matching import add_matching_features

    infoot, helpers = local_helpers
    torch.manual_seed(25)
    encoders = torch.nn.ModuleDict({name: torch.nn.Linear(5, 3) for name in ("cat", "dog")})
    norms = torch.nn.ModuleDict({name: torch.nn.BatchNorm1d(3) for name in encoders})
    optimizer = torch.optim.Adam([*encoders.parameters(), *norms.parameters()], lr=.01)
    encoded = {}
    for name, encoder in encoders.items():
        raw = encoder(torch.randn(8, 5))
        encoded[name] = feature_batch(raw[2:], raw[:2])
    add_matching_features(encoded, norms)
    cat, dog = encoded["cat"], encoded["dog"]
    source, target = cat["references"]["m"], dog["references"]["m"]
    plan = helpers.fit_transport(source, target, h=.8, reg=.5, iterations=3)
    assert not plan.requires_grad
    plan.requires_grad_()
    alignment = helpers.alignment_loss(source, target, plan, h=.8, reg=.5)
    solver = infoot.FusedInfoOT(source, target, h=.8, lam=.1, reg=.5)
    torch.testing.assert_close(alignment, infoot.fitting_loss(
        plan, solver.Ks, solver.Kt, .5, C=solver.C, mi_weight=.1, eps=1e-5,
    ))
    for parameter in (encoders["cat"].weight, encoders["dog"].weight,
                      norms["cat"].weight, norms["dog"].weight):
        gradient, = torch.autograd.grad(alignment, parameter, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0

    query = cat["queries"]["m"]
    raw_target = dog["references"]["v"]
    mapped = helpers.conditional_mapping(query, source, target, plan, h=.8)
    solver = infoot.InfoOT(source, target, h=.8)
    solver.P = plan.detach()
    scores = solver.conditional_score(query)
    torch.testing.assert_close(mapped, infoot.projection(scores, target))
    assert not torch.allclose(mapped, infoot.projection(scores, raw_target))
    for features in (query, source, target, raw_target):
        gradient, = torch.autograd.grad(mapped.square().mean(), features, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0

    before = {name: norm.weight.detach().clone() for name, norm in norms.items()}
    (alignment + mapped.square().mean()).backward()
    assert plan.grad is None
    optimizer.step()
    for name, norm in norms.items():
        assert not torch.equal(norm.weight, before[name])


def test_covariance_penalizes_diagonal_and_off_diagonal_with_explicit_reduction(local_helpers):
    from infoot_helper.batchnorm_matching import covariance_loss

    basis = torch.eye(2, dtype=torch.float64) * 2 ** .5
    white = torch.cat([basis, -basis])
    correlated = torch.tensor([[1., 1.], [-1., -1.]] * 2, dtype=torch.float64)
    assert covariance_loss(white).item() == pytest.approx(0)
    assert covariance_loss(2 * white).item() == pytest.approx(9)
    assert covariance_loss(correlated).item() == pytest.approx(1)
    assert covariance_loss(torch.ones_like(white)).item() == pytest.approx(1)
    torch.testing.assert_close(covariance_loss(correlated + 10), covariance_loss(correlated))

    # Two references in three dimensions have rank <= 1: the minimum is 2/3.
    low_rank = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
    assert covariance_loss(low_rank).item() == pytest.approx(2 / 3)


def test_covariance_gradients_only_reach_raw_references_and_accumulate_in_fp32(local_helpers):
    from infoot_helper.batchnorm_matching import covariance_loss

    torch.manual_seed(27)
    raw = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(covariance_loss, (raw,))
    centered = raw[2:] - raw[2:].mean(0)
    expected = ((centered.T @ centered / len(centered) - torch.eye(4)).square()).sum() / 4
    torch.testing.assert_close(covariance_loss(raw[2:]), expected)
    covariance_loss(raw[2:]).backward()
    assert raw.grad[:2].count_nonzero() == 0
    assert torch.isfinite(raw.grad[2:]).all() and raw.grad[2:].norm() > 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = covariance_loss(raw.float())
    torch.testing.assert_close(actual, covariance_loss(raw.float()), rtol=0, atol=0)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("affine", [False, True])
def test_export_roundtrip_and_eval_mapping_are_independent_of_query_batch(local_helpers, tmp_path, affine):
    from infoot_helper.batchnorm_matching import load_batchnorm

    _, helpers = local_helpers
    torch.manual_seed(29)
    references = {"cat": torch.randn(7, 3), "dog": torch.randn(7, 3) * 3 + 10}
    norms = {name: torch.nn.BatchNorm1d(3, affine=affine, eps=.002, momentum=.4)
             for name in references}
    domains = {name: SimpleNamespace(training_config={},
                                    branch=SimpleNamespace(pdae_state_dict=lambda: {}))
               for name in references}
    for name, norm in norms.items():
        norm(references[name])
        norm(references[name] * 1.2)
        if affine:
            with torch.no_grad():
                norm.weight.copy_(torch.tensor([.7, 1.2, 1.5]))
                norm.bias.copy_(torch.tensor([1., 2., 3.]))
        norm.eval()
    helpers.save_domain_checkpoints(domains, tmp_path, 200, norms)
    loaded = {name: load_batchnorm(tmp_path / f"{name}_step_000200.pt",
                                   domain=name, step=200, device="cpu") for name in norms}
    for name, norm in loaded.items():
        assert not norm.training and norm.affine == affine
        assert norm.eps == .002 and norm.momentum == .4
        torch.testing.assert_close(norm.state_dict(), norms[name].state_dict(), rtol=0, atol=0)
    original_state = {name: deepcopy(norm.state_dict()) for name, norm in loaded.items()}
    queries = references["cat"][:3] + .2
    plan = torch.eye(7) / 7

    def map_queries(query, batch_norms):
        with torch.no_grad():
            return helpers.conditional_mapping(
                batch_norms["cat"](query), batch_norms["cat"](references["cat"]),
                batch_norms["dog"](references["dog"]), plan, h=.8,
            )

    together = map_queries(queries, loaded)
    torch.testing.assert_close(together, map_queries(queries, norms), rtol=0, atol=0)
    separately = torch.cat([map_queries(row[None], loaded) for row in queries])
    torch.testing.assert_close(together, separately)
    torch.testing.assert_close(together.flip(0), map_queries(queries.flip(0), loaded))
    expanded = map_queries(torch.cat([queries, queries[:1] + 3]), loaded)
    torch.testing.assert_close(together, expanded[:len(queries)])
    for name, norm in loaded.items():
        torch.testing.assert_close(norm.state_dict(), original_state[name], rtol=0, atol=0)


def test_old_checkpoints_cannot_silently_use_fresh_batchnorm(local_helpers, tmp_path):
    from infoot_helper.batchnorm_matching import batchnorm_checkpoint, load_batchnorm

    path = tmp_path / "cat.pt"
    torch.save({"domain": "cat", "step": 200}, path)
    with pytest.raises(ValueError, match="no matching BatchNorm"):
        load_batchnorm(path, domain="cat", step=200, device="cpu")
    norm = torch.nn.BatchNorm1d(3)
    torch.save({"domain": "cat", "step": 200, "matching_batchnorm": batchnorm_checkpoint(norm)}, path)
    with pytest.raises(ValueError, match="not observed"):
        load_batchnorm(path, domain="cat", step=200, device="cpu")
    with pytest.raises(ValueError, match="does not match"):
        load_batchnorm(path, domain="dog", step=200, device="cpu")
