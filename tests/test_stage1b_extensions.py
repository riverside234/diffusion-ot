"""CPU integration coverage of the additive Stage 1B translation objectives."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.losses.semantic_prior import descriptor_digest
from diffusion_ot.models.generator_adaptation import generator_parameter_view
from test_decoded_translation import config, domains, tiny_features
from test_infoot_semantic_prior import save_prior


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    import diffusion_ot.data.latent_dataset as data
    import diffusion_ot.training.train_joint_infoot as train
    import diffusion_ot.training.decoded_translation as decoded

    torch.manual_seed(40)
    originals = domains()
    features = tiny_features()
    latest = {}
    monkeypatch.setattr(decoded, "load_image_features", lambda *a, **kw: deepcopy(features))

    class Dataset:
        def __init__(self, path, domain, split="train", **kwargs):
            self.domain, self.split = domain, split
            self.records = [{"sample_id": f"{domain}_{split}_{i}", "latent_path": "unused"} for i in range(8)]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            gen = torch.Generator().manual_seed(index + (100 if self.domain == "dog" else 0)
                                                + (20 if self.split == "val" else 0))
            return {"x0_latent": torch.randn(4, 8, 8, generator=gen),
                    "sample_id": self.records[index]["sample_id"], "domain": self.domain,
                    "split": self.split, "latent_path": "unused", "metadata": self.records[index]}

    def load_domain(cfg, root, domain, **kwargs):
        value = deepcopy(originals[domain])
        value.checkpoint_path, value.checkpoint_step = tmp_path / f"{domain}.pt", 100
        value.data_config_path = tmp_path / "data.yaml"
        train.freeze_generator_train_encoder(value.branch)
        latest[domain] = value
        return value

    monkeypatch.setattr(data, "CachedLatentDataset", Dataset)
    monkeypatch.setattr(train, "_load_training_domain", load_domain)
    payload = save_prior(tmp_path / "prior.pt")
    # Match the tiny DINO's 2x2 off-diagonal self-similarity descriptor size.
    payload["features"] = F.normalize(torch.randn(len(payload["sample_ids"]), 12), dim=1)
    payload["fingerprint"] = descriptor_digest(payload)
    torch.save(payload, tmp_path / "prior.pt")
    for domain in originals:
        (tmp_path / f"{domain}.yaml").write_text(yaml.safe_dump({"data_config": "data.yaml"}))
    cfg = config()
    cfg["decoded_translation"].update(structure_contrastive_weight=.01,
                                      structure_contrastive_temperature=.2,
                                      code_consistency_weight=.02, code_consistency_mode="contrastive",
                                      # The tiny fixture uses only two nearly
                                      # overlapping query means. Keep distinct
                                      # directions as negatives in this test.
                                      code_consistency_negative_similarity_threshold=1.)
    cfg.update(
        project_root=str(tmp_path), output_dir="out", seed=42,
        stage1a={d: {"config": f"{d}.yaml"} for d in originals},
        data={"transport_batch_size": 8, "reconstruction_batch_size": 4, "num_workers": 0,
              "random_horizontal_flip": 0, "pin_memory": False},
        matching={"bandwidth_multiplier": .55, "calibration_samples": 8, "distance_scale_gradient": "full"},
        matching_regularization={"enabled": True, "std_target": .7,
                                 "variance_weight": .05, "covariance_weight": .01},
        matching_contrastive={"enabled": True, "neighborhood_weight": .01,
                              "conditional_weight": .005, "positive_count": 2, "temperature": .2},
        decoded_encoder_balance={"enabled": True, "target_ratio": 2., "min_scale": .25, "max_scale": 4.},
        infoot={"variant": "fused", "inner_iterations": 300, "entropy_epsilon": .2,
                "mi_weight": .1, "outer_tolerance": 1e-5,
                "strict_convergence": True, "require_outer_convergence": True},
        semantic_prior={"path": "prior.pt", "neighborhood_geometry": "rms_distance"},
        loss_weights={"semantic_neighborhood": .05, "conditional_structure": .05},
        train={"max_steps": 2, "log_every": 1, "save_every": 1, "validation_every": 1,
               "validation_samples": 2, "gradient_diagnostics_every": 1, "lr_encoder": .001},
        gradient_guard={"enabled": False},
    )

    def run(output, *, steps=2, resume=False, modify=None):
        current = deepcopy(cfg)
        current["output_dir"] = output
        if modify:
            modify(current)
        path = tmp_path / f"{output}.yaml"
        path.write_text(yaml.safe_dump(current))
        report = train.train_joint_infoot(path, max_steps=steps, resume_from="latest" if resume else None)
        checkpoint = train._load_checkpoint(Path(report.checkpoint_path))
        logs = {
            name: [json.loads(row) for row in (tmp_path / output / "logs" / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation")
        }
        return checkpoint, logs

    return run, originals, latest


def assert_finite_numbers(value):
    if isinstance(value, dict):
        for nested in value.values():
            assert_finite_numbers(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_finite_numbers(nested)
    elif isinstance(value, float):
        assert math.isfinite(value)


def assert_tensor_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tensor_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tensor_tree_equal(a, b)
    else:
        assert first == second


def test_actual_training_extensions_validate_full_support_and_resume_compatibly(experiment):
    run, originals, latest = experiment
    complete, logs = run("complete")
    assert complete["step"] == complete["decoded_discriminator_updates"] == 2
    assert [row["step"] for row in logs["validation"]] == [0, 1, 2]
    for row in logs["train"]:
        assert_finite_numbers(row)
        contrastive = row["matching_contrastive"]
        assert contrastive["neighborhood_loss"] > 0 and contrastive["conditional_loss"] > 0
        assert contrastive["weighted_loss"] == pytest.approx(
            .01 * contrastive["neighborhood_loss"] + .005 * contrastive["conditional_loss"])
        assert row["weighted_matching_contrastive_encoder_gradient_norm"] > 0
        assert row["weighted_matching_contrastive_matching_head_gradient_norm"] > 0
        assert row["weighted_code_consistency_generator_gradient_norm"] > 0
        assert row["applied_code_consistency_encoder_gradient_norm"] == 0
        assert row["applied_code_consistency_matching_head_gradient_norm"] == 0
        decoded = row["decoded_translation"]
        assert decoded["structure_contrastive_loss"] > 0
        assert decoded["code_consistency_loss"] > 0
        assert decoded["code_consistency_mode"] == "contrastive"
        assert decoded["code_consistency_gradient_routing"] == "generator_only"
        assert row["code_consistency_loss"] == pytest.approx(decoded["code_consistency_effective_weighted_loss"])
        balance = row["decoded_encoder_balance"]
        assert balance["enabled"] and balance["applied"]
        assert -1 <= balance["cosine"] <= 1
        requested = balance["effective_target_ratio"] / balance["prebalanced_ratio"]
        assert balance["scale"] == pytest.approx(min(4., max(.25, requested)))
        assert balance["effective_ratio"] == pytest.approx(balance["prebalanced_ratio"] * balance["scale"])
        for direction in ("cat_to_dog", "dog_to_cat"):
            assert contrastive["conditional"][direction]["valid_candidates"] == 6
            assert decoded[direction]["reference_targets"] == 6
            assert decoded[direction]["code_consistency"]["valid_conditions"] == 2
            assert decoded[direction]["code_consistency"]["condition_bank_size"] == 2
            assert decoded[direction]["structure_contrastive"]["negative_bank_size"] == 6
        for domain in ("cat", "dog"):
            assert contrastive["neighborhood"][domain]["valid_candidates"] == 5
            assert row["infoot_reference_counts"][domain] == 6
            assert row["matching_regularization"][domain]["samples"] == 6
    for row in logs["validation"]:
        assert_finite_numbers(row)
        assert row["matching_contrastive"]["conditional"]["cat_to_dog"]["valid_candidates"] == 6
        assert row["decoded_translation"]["cat_to_dog"]["reference_targets"] == 6
        assert row["decoded_translation"]["cat_to_dog"]["code_consistency"]["valid_conditions"] == 2
        assert len(row["decoded_translation"]["code_condition_bank_ids"]["cat"]) == 2
        assert "structure_contrastive_loss" in row["decoded_translation"]
    for domain, original in originals.items():
        current = latest[domain].branch
        assert any(not torch.equal(value, original.branch.encoder.state_dict()[key])
                   for key, value in current.encoder.state_dict().items())
        for group in ("adapters", "lora"):
            before = generator_parameter_view(original.branch)[group].state_dict()
            assert any(not torch.equal(value, before[key])
                       for key, value in generator_parameter_view(current)[group].state_dict().items())
    run("resumed", steps=1)
    resumed, resumed_logs = run("resumed", resume=True)
    assert complete["step"] == resumed["step"] == resumed["decoded_discriminator_updates"] == 2
    assert [row["step"] for row in resumed_logs["train"]] == [1, 2]
    # Dedicated diffusion noise and native RNG streams restore exactly. The
    # existing shuffled DataLoader saves its generator but not its permutation
    # and cursor, so resume does not promise the uninterrupted batch order or
    # bitwise-identical model weights. This test does not conceal that limit.
    for key in ("decoded_noise_states", "rng_state"):
        assert_tensor_tree_equal(complete[key], resumed[key])
    for key in ("matching_contrastive", "decoded_encoder_balance", "decoded_translation"):
        assert complete["config"][key] == resumed["config"][key]
        assert key in resumed_logs["train"][-1]
    assert_finite_numbers(resumed_logs["train"][-1])
    for key in ("encoders", "matching_heads", "encoder_ema", "matching_head_ema", "generators",
                "generator_ema", "optimizer", "decoded_discriminators", "decoded_discriminator_optimizer"):
        assert resumed[key]


def test_generator_code_consistency_does_not_change_encoder_or_head_updates(experiment):
    run, _, _ = experiment
    with_code, _ = run("with_code", steps=1)
    without_code, _ = run("without_code", steps=1,
                          modify=lambda cfg: cfg["decoded_translation"].update(code_consistency_weight=0))
    for key in ("encoders", "matching_heads", "decoded_discriminators"):
        assert_tensor_tree_equal(with_code[key], without_code[key])
    with pytest.raises(AssertionError):
        assert_tensor_tree_equal(with_code["generators"], without_code["generators"])


def test_resume_rejects_changed_new_objectives_and_tiny_reference_banks(experiment):
    run, _, _ = experiment
    run("resume_guard", steps=1)
    for key, change in (("matching_contrastive", {"neighborhood_weight": .02}),
                        ("decoded_encoder_balance", {"target_ratio": 1.5}),
                        ("decoded_translation", {"code_consistency_weight": .04}),
                        ("decoded_translation", {"code_consistency_mode": "cosine"}),
                        ("decoded_translation", {"code_consistency_temperature": .1}),
                        ("decoded_translation", {"code_consistency_negative_similarity_threshold": .95})):
        with pytest.raises(ValueError, match=f"Resume cannot change {key}"):
            run("resume_guard", resume=True, modify=lambda cfg: cfg[key].update(change))
    # Six references leave five non-self candidates: k=5 supplies no negatives.
    with pytest.raises(ValueError, match="enough OT references"):
        run("too_few_negatives", steps=1,
            modify=lambda cfg: cfg["matching_contrastive"].update(positive_count=5))
