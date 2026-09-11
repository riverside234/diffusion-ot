from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_legacy_alignment_config_uses_non_cfg_stage1a_models():
    from pathlib import Path

    from diffusion_ot.integrations.hf_snapshot import load_yaml_config

    root = Path(__file__).resolve().parents[1]
    alignment = load_yaml_config(
        root / "configs" / "stage1b_infoot" / "plain_sit_b2_nocfg.yaml"
    )
    for domain in ("cat", "dog"):
        domain_config = alignment["stage1a"][domain]
        stage1a = load_yaml_config(root / domain_config["config"])
        assert stage1a["semantic_cfg"]["enabled"] is False
        assert "_cfg/" not in domain_config["checkpoint"]


def test_cfg_alignment_config_uses_cfg_stage1a_checkpoints():
    from pathlib import Path

    from diffusion_ot.integrations.hf_snapshot import load_yaml_config

    root = Path(__file__).resolve().parents[1]
    alignment = load_yaml_config(
        root / "configs" / "stage1b_infoot" / "plain_sit_b2.yaml"
    )
    for domain in ("cat", "dog"):
        domain_config = alignment["stage1a"][domain]
        stage1a = load_yaml_config(root / domain_config["config"])
        assert stage1a["semantic_cfg"]["enabled"] is True
        assert "_cfg/" in domain_config["checkpoint"]


def _bank(domain: str, split: str, ids: list[str], checkpoint: str = "same"):
    from diffusion_ot.evaluation.stage1b_eval import LatentBank

    raw = torch.arange(len(ids) * 3, dtype=torch.float32).reshape(len(ids), 3) + 1
    return LatentBank(
        domain=domain,
        split=split,
        raw_codes=raw,
        matching_features=torch.nn.functional.normalize(raw, dim=1),
        sample_ids=ids,
        metadata=[{"sample_id": value} for value in ids],
        checkpoint_id=checkpoint,
    )


def test_deterministic_indices_are_reproducible_and_bounded():
    from diffusion_ot.evaluation.stage1b_eval import deterministic_indices

    first = deterministic_indices(20, 7, seed=3)
    second = deterministic_indices(20, 7, seed=3)
    assert first == second
    assert len(first) == len(set(first)) == 7
    assert min(first) >= 0 and max(first) < 20


def test_full_projection_bank_keeps_stable_dataset_order():
    from diffusion_ot.evaluation.stage1b_eval import deterministic_indices

    assert deterministic_indices(5, None, seed=9) == [0, 1, 2, 3, 4]
    assert deterministic_indices(5, 10, seed=9) == [0, 1, 2, 3, 4]


def test_bank_compatibility_rejects_split_leakage():
    from diffusion_ot.evaluation.stage1b_eval import validate_bank_compatibility

    reference = _bank("cat", "train", ["a", "b"])
    query = _bank("cat", "val", ["b", "c"])
    with pytest.raises(ValueError, match="overlap"):
        validate_bank_compatibility(reference, query)


def test_bank_compatibility_rejects_stale_encoder_bank():
    from diffusion_ot.evaluation.stage1b_eval import validate_bank_compatibility

    reference = _bank("dog", "train", ["a"], checkpoint="old")
    query = _bank("dog", "val", ["b"], checkpoint="new")
    with pytest.raises(ValueError, match="different encoders"):
        validate_bank_compatibility(reference, query)


def test_latent_bank_roundtrip_keeps_raw_and_matching_features_separate(tmp_path):
    from diffusion_ot.evaluation.stage1b_eval import load_latent_bank, save_latent_bank

    bank = _bank("cat", "train", ["a", "b", "c"])
    path = tmp_path / "bank.pt"
    save_latent_bank(path, bank)
    loaded = load_latent_bank(path)

    torch.testing.assert_close(loaded.raw_codes, bank.raw_codes)
    torch.testing.assert_close(loaded.matching_features, bank.matching_features)
    assert not torch.equal(loaded.raw_codes, loaded.matching_features)


def test_precision_at_k_uses_only_requested_proxy_attribute():
    from diffusion_ot.evaluation.stage1b_eval import precision_at_k

    rankings = torch.tensor([[0, 1, 2], [2, 1, 0]])
    labels = {
        "q0": {"viewpoint": "front", "framing": "close"},
        "q1": {"viewpoint": "side", "framing": "close"},
        "g0": {"viewpoint": "front", "framing": "wide"},
        "g1": {"viewpoint": "front", "framing": "wide"},
        "g2": {"viewpoint": "side", "framing": "wide"},
    }
    result = precision_at_k(
        rankings,
        ["q0", "q1"],
        ["g0", "g1", "g2"],
        labels,
        attribute="viewpoint",
        ks=[1, 2],
    )

    assert result["precision_at_1"] == pytest.approx(1.0)
    assert result["precision_at_2"] == pytest.approx(0.75)
    assert result["query_coverage"] == pytest.approx(1.0)


def test_direction_evaluation_keeps_query_and_gallery_protocols_distinct():
    from diffusion_ot.evaluation.stage1b_eval import _direction_evaluation

    source_reference = _bank("cat", "train", ["cr0", "cr1"])
    target_reference = _bank("dog", "train", ["dr0", "dr1", "dr2"])
    source_query = _bank("cat", "val", ["cq0", "cq1"])
    target_gallery = _bank("dog", "val", ["dg0", "dg1"])
    coupling = torch.tensor([[0.2, 0.1, 0.2], [0.1, 0.25, 0.15]])
    coupling = coupling / coupling.sum()
    labels = {sample_id: {"viewpoint": "front"} for sample_id in [
        *source_query.sample_ids, *target_reference.sample_ids, *target_gallery.sample_ids
    ]}

    report, tensors = _direction_evaluation(
        source_reference,
        target_reference,
        source_query,
        target_gallery,
        coupling,
        source_scale=1.0,
        target_scale=1.0,
        bandwidth=1.0,
        labels=labels,
        attributes=["viewpoint"],
        ks=[1],
        seed=5,
        eps=1.0e-8,
    )

    assert all(target_id.startswith("dg") for ids in report["conditional_top_ids"].values() for target_id in ids)
    assert tensors["conditional_codes"].shape == (2, 3)
    assert tensors["barycentric_codes"].shape == (2, 3)


def test_direction_evaluation_projects_over_bank_larger_than_fit_references():
    from diffusion_ot.evaluation.stage1b_eval import _direction_evaluation
    from diffusion_ot.losses.infoot import conditional_projection_weights

    source_reference = _bank("cat", "train", ["cr0", "cr1"])
    target_reference = _bank("dog", "train", ["dr0", "dr1", "dr2"])
    target_projection = _bank("dog", "train", ["dp0", "dp1", "dp2", "dp3"])
    source_query = _bank("cat", "val", ["cq0", "cq1"])
    target_gallery = _bank("dog", "val", ["dg0", "dg1"])
    target_projection.raw_codes += 50
    coupling = torch.tensor([[0.2, 0.1, 0.2], [0.1, 0.25, 0.15]])
    coupling /= coupling.sum()

    report, tensors = _direction_evaluation(
        source_reference,
        target_reference,
        source_query,
        target_gallery,
        coupling,
        target_projection=target_projection,
        source_scale=1.0,
        target_scale=1.0,
        bandwidth=1.0,
        labels={},
        attributes=[],
        ks=[1],
        seed=5,
        eps=1.0e-8,
    )
    expected_weights = conditional_projection_weights(
        source_query.matching_features,
        target_projection.matching_features,
        source_reference.matching_features,
        target_reference.matching_features,
        coupling,
    )

    assert report["projection_support"] == "target_training_projection_bank"
    assert report["projection_target_count"] == 4
    torch.testing.assert_close(tensors["conditional_weights"], expected_weights)
    torch.testing.assert_close(
        tensors["conditional_codes"], expected_weights @ target_projection.raw_codes
    )


def test_protocol_identifier_versions_projection_bank_overrides():
    from diffusion_ot.evaluation.stage1b_eval import _protocol_identifier

    config = {"data": {"projection_samples_per_domain": "all"}, "seed": 4}
    full = _protocol_identifier(
        config, max_reference=None, max_projection=None, max_query=None
    )
    bounded = _protocol_identifier(
        config, max_reference=None, max_projection=32, max_query=None
    )

    assert full != bounded
    assert full == _protocol_identifier(
        config, max_reference=None, max_projection=None, max_query=None
    )


def test_checkpoint_selection_uses_full_mean_metrics():
    from diffusion_ot.evaluation.stage1b_eval import _checkpoint_selection_summary

    summary = _checkpoint_selection_summary(
        {
            "cat_to_dog": {
                "projection_method": "infoot_eq7_conditional_expectation",
                "projection_support": "target_training_projection_bank",
                "projection_target_count": 40,
                "mean_conditional_effective_target_count": 12.5,
                "mean_nearest_target_distance": 0.3,
                "mean_projected_to_target_norm_ratio": 0.9,
                "precision": {"conditional": {"viewpoint": {"precision_at_1": 0.6}}},
            }
        },
        {"cat": {"latent_mse": 0.1}},
        {"status": "compared"},
    )

    assert summary["primary_readout"] == "infoot_eq7_conditional_expectation"
    assert summary["directions"]["cat_to_dog"]["projection_target_count"] == 40
    assert summary["directions"]["cat_to_dog"]["mean_nearest_target_distance"] == 0.3
    assert summary["baseline_comparison_status"] == "compared"


@pytest.mark.parametrize("reverse", [False, True])
def test_decoded_grid_receives_full_equation7_mean(monkeypatch, tmp_path, reverse):
    from types import SimpleNamespace

    import diffusion_ot.evaluation.stage1a_eval as stage1a
    import diffusion_ot.evaluation.stage1b_eval as stage1b
    import torchvision.utils

    cat = _bank("cat", "train", ["cr0", "cr1"])
    dog = _bank("dog", "train", ["dr0", "dr1", "dr2"])
    cat.matching_features = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    dog.matching_features = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.4, 0.0], [2.0, 1.4, 0.0]])
    coupling = torch.tensor([[0.30, 0.15, 0.05], [1 / 3 - 0.30, 1 / 3 - 0.15, 1 / 3 - 0.05]])
    source, target = (dog, cat) if reverse else (cat, dog)
    coupling = coupling.T if reverse else coupling
    query = _bank(source.domain, "val", ["q0", "q1"])
    query.matching_features = source.matching_features[:2] + 0.3
    gallery = _bank(target.domain, "val", ["g0", "g1"])
    # Make accidental averaging of validation-gallery codes easy to detect.
    gallery.raw_codes += 1000
    bandwidth, source_scale, target_scale = 0.7, 0.8, 1.3
    report, tensors = stage1b._direction_evaluation(
        source, target, query, gallery, coupling,
        source_scale=source_scale, target_scale=target_scale, bandwidth=bandwidth,
        labels={}, attributes=[], ks=[1], seed=5, eps=1.0e-8,
    )
    kx = torch.exp(-torch.cdist(query.matching_features, source.matching_features).square()
                   / (2 * (bandwidth * source_scale) ** 2))
    ky = torch.exp(-torch.cdist(target.matching_features, target.matching_features).square()
                   / (2 * (bandwidth * target_scale) ** 2))
    a = torch.full((len(source.sample_ids),), 1 / len(source.sample_ids))
    b = torch.full((len(target.sample_ids),), 1 / len(target.sample_ids))
    expected_weights = (kx @ coupling @ ky.T) / ((kx @ a)[:, None] * (ky @ b)[None, :])
    expected_weights *= b
    expected_weights /= expected_weights.sum(1, keepdim=True)
    expected_mean = expected_weights @ target.raw_codes
    torch.testing.assert_close(tensors["conditional_weights"], expected_weights)
    torch.testing.assert_close(tensors["conditional_codes"], expected_mean)
    legacy = kx @ coupling
    legacy /= legacy.sum(1, keepdim=True)
    assert not torch.allclose(tensors["conditional_codes"], legacy @ target.raw_codes)
    assert report["projection_method"] == "infoot_eq7_conditional_expectation"
    assert report["projection_target_count"] == len(target.sample_ids)

    decoder_codes, decoder_noise = [], []

    def fake_integrate(branch, transformer, noise, codes, **kwargs):
        decoder_codes.append(codes.clone())
        decoder_noise.append(noise.clone())
        return noise

    monkeypatch.setattr(stage1a, "integrate_pdae_flow", fake_integrate)
    monkeypatch.setattr(stage1a, "decode_vae_latents", lambda vae, latent: latent)
    monkeypatch.setattr(stage1b, "_load_latents_from_bank", lambda bank, count: torch.zeros(count, 3, 2, 2))
    monkeypatch.setattr(torchvision.utils, "save_image", lambda *args, **kwargs: None)
    context = SimpleNamespace(
        device="cpu", dtype=torch.float32, vae=None, branch=None, transformer=None, training_config={}
    )
    stage1b._save_translation_grid(
        context, context, query, target, tensors["conditional_weights"],
        count=2, num_steps=2, guidance_scale=1.0, temperature=0.5, seed=7,
        output_path=tmp_path / "grid.png",
    )
    # Sampling temperature must not change the full-mean decoder input.
    torch.testing.assert_close(decoder_codes[0], expected_mean)
    assert len(decoder_codes) == 3
    torch.testing.assert_close(decoder_noise[0], decoder_noise[1])
    torch.testing.assert_close(decoder_noise[0], decoder_noise[2])
