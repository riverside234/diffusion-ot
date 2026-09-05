from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


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
