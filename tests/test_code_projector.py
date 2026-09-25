"""Global MLP InfoNCE routing, negative filtering, and checkpoint integrity."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.contrastive import detached_key_contrastive_loss
from diffusion_ot.models.code_projector import CodeProjectionMLP, code_projector_options, load_code_projector
from diffusion_ot.models.patch_sampler import PatchSampleMLP
from diffusion_ot.training.decoded_translation import self_supervised_translation_options
from diffusion_ot.training.self_supervised_translation import SelfSupervisedDecoderTraining, source_code_contrastive_loss
from test_self_supervised_translation import config, domains, batch


def mlp_config():
    cfg = config()
    cfg["decoded_translation"]["source_contrastive_projector"] = {
        "kind": "mlp", "projection_dim": 16, "lr": 2e-4, "grad_clip_norm": 1.}
    return cfg


def test_code_mlp_reuses_cut_transform_preserves_rng_and_has_finite_autocast_gradients():
    torch.manual_seed(15)
    rng = torch.get_rng_state().clone()
    head = CodeProjectionMLP(8, 16, seed=3)
    cut = PatchSampleMLP((8,), 16, seed=3)
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn(4, 8, requires_grad=True)
    torch.testing.assert_close(head(x), cut(x, 0), rtol=0, atol=0)
    assert sum(isinstance(m, torch.nn.Linear) for m in head.modules()) == 2
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = head(x)
    assert output.dtype == torch.float32
    output.square().mean().backward()
    assert x.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match="Global projector"):
        head(torch.zeros(4, 9))


def test_projected_infonce_matches_explicit_logits_and_detaches_entire_key_path():
    torch.manual_seed(31)
    query = torch.randn(4, 8, requires_grad=True)
    keys = torch.eye(8)[:4].requires_grad_()
    bank = torch.eye(8).requires_grad_()
    query_head, key_head = (CodeProjectionMLP(8, 16, seed=s) for s in (1, 2))
    loss, metrics = source_code_contrastive_loss(query, keys, bank,
        query_projector=query_head, key_projector=key_head)
    q, k, b = [F.normalize(x, dim=-1) for x in (query_head(query), key_head(keys), key_head(bank))]
    logits = torch.cat(((q * k).sum(-1, keepdim=True),
                        (q @ b.T).masked_fill(torch.eye(8, dtype=torch.bool)[:4], -torch.inf)), dim=1) / .2
    expected = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert query.grad.norm() > 0 and keys.grad is None and bank.grad is None
    assert all(p.grad is not None and p.grad.norm() > 0 for p in query_head.parameters())
    assert all(p.grad is None for p in key_head.parameters())
    assert metrics["comparison_space"] == "global_mlp"
    assert metrics["mean_usable_negatives"] == 7


def test_collapsed_mlp_cannot_mask_distinct_raw_code_negatives_as_duplicates():
    head = CodeProjectionMLP(4, 8)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.mlp_0[-1].bias.fill_(1.)
    keys = torch.eye(4)
    loss, metrics = source_code_contrastive_loss(keys, keys, keys,
        query_projector=head, key_projector=deepcopy(head))
    assert metrics["usable_samples"] == 4
    assert metrics["mean_usable_negatives"] == 3
    assert metrics["retrieval_top1"] == metrics["retrieval_chance"] == .25
    assert float(loss.detach()) == pytest.approx(torch.tensor(4.).log().item())
    assert metrics["negative_filter_space"] == "original_source_encoder_code"


def test_mlp_runtime_trains_both_domain_heads_without_changing_source_keys(tmp_path):
    torch.manual_seed(25)
    runtime = SelfSupervisedDecoderTraining(mlp_config(), domains(), None, tmp_path, seed=25)
    weights, refs, keys, logits, latents = batch()
    loss, metrics = runtime.loss(weights, refs, latents, latents, step=2, source_query_codes=keys)
    loss.backward()
    assert runtime.code_projectors["cat"] is not runtime.code_projectors["dog"]
    for source, target in (("cat", "dog"), ("dog", "cat")):
        assert all(p.grad is not None and p.grad.norm() > 0 for p in runtime.code_projectors[target].parameters())
        assert keys[source].grad is None
        assert refs[target].grad.norm() > 0 and logits[f"{source}_to_{target}"].grad.norm() > 0
        retrieval = metrics[f"{source}_to_{target}"]["source_contrastive"]
        assert retrieval["query_projector_domain"] == target and retrieval["key_projector_domain"] == source
    group = next(g for g in runtime.parameter_groups() if g["name"] == "code_projectors")
    assert {id(p) for p in group["params"]} == {id(p) for p in runtime.code_projector_parameters}
    assert not runtime.patch_projectors


def test_mlp_checkpoint_restores_raw_ema_and_rejects_missing_or_incompatible_heads(tmp_path):
    runtime = SelfSupervisedDecoderTraining(mlp_config(), domains(), None, tmp_path, seed=1)
    with torch.no_grad():
        for p in runtime.code_projector_parameters:
            p.add_(.03)
    runtime.code_projector_ema.update(runtime.code_projectors)
    state = deepcopy(runtime.checkpoint_state())
    state.update(config=mlp_config(), format_version=4)
    restored = SelfSupervisedDecoderTraining(mlp_config(), domains(), None, tmp_path, seed=91)
    restored.load_checkpoint(state)
    assert restored.code_projector_ema.num_updates == 1
    for domain in ("cat", "dog"):
        for key, actual in restored.code_projectors[domain].state_dict().items():
            torch.testing.assert_close(actual, state["code_projectors"][domain][key], rtol=0, atol=0)
        for weights, field in (("raw", "code_projectors"), ("ema", "code_projector_ema")):
            actual = load_code_projector(6, runtime.code_options, state, domain, weights=weights, device="cpu")
            for key, value in actual.state_dict().items():
                torch.testing.assert_close(value, state[field][domain][key], rtol=0, atol=0)
            assert not actual.training and all(not p.requires_grad for p in actual.parameters())
            with pytest.raises(ValueError, match="Checkpoint has no"):
                load_code_projector(6, runtime.code_options, {}, domain, weights=weights, device="cpu")
    for field, match in (("decoded_code_projector_options", "protocol"),
                         ("code_projectors", "no code_projectors"),
                         ("code_projector_ema_state", "EMA state")):
        invalid = {k: v for k, v in state.items() if k != field}
        with pytest.raises(ValueError, match=match):
            restored.load_checkpoint(invalid)
    without_mlp = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=1)
    with pytest.raises(ValueError, match="projector protocol"):
        without_mlp.load_checkpoint(state)


@pytest.mark.parametrize("change", [{"projection_dim": 0}, {"projection_dim": True}, {"lr": 0},
                                    {"grad_clip_norm": float("nan")}, {"kind": "linear"}, {"typo": 1}])
def test_invalid_mlp_configuration_is_rejected(change):
    with pytest.raises(ValueError, match="projector"):
        code_projector_options({**mlp_config()["decoded_translation"]["source_contrastive_projector"], **change})


def test_mlp_settings_cannot_be_silently_ignored_by_patchnce_or_checkpoint_evaluation():
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint
    image = mlp_config()["decoded_translation"]
    with pytest.raises(ValueError, match="source_infonce"):
        self_supervised_translation_options({**image, "objective": "patchnce"})
    for current, saved in ((mlp_config(), config()), (config(), mlp_config())):
        with pytest.raises(ValueError, match="projector protocol"):
            _validate_self_supervised_checkpoint(current, {"config": saved})
    _validate_self_supervised_checkpoint(mlp_config(), {"config": mlp_config()})


def test_explicit_negative_mask_requires_boolean_matrix():
    for mask in (torch.ones(3, 3), torch.ones(3, 2, dtype=torch.bool)):
        with pytest.raises(ValueError, match="negative_mask"):
            detached_key_contrastive_loss(torch.eye(3), torch.eye(3), torch.eye(3), negative_mask=mask)
