from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusion_ot.losses.semantic_prior import (
    SemanticPriorBank, descriptor_digest, neighborhood_distillation_loss,
    patch_structure_descriptor, validate_prior_resume,
)


def save_prior(path, count=8):
    generator = torch.Generator().manual_seed(71)
    payload = {"format_version": 1, "sample_ids": [], "domains": [], "splits": [],
               "features": [], "metadata": {"cost_scale": 1.0}}
    for domain in ("cat", "dog"):
        for split in ("train", "val"):
            for i in range(count):
                payload["sample_ids"].append(f"{domain}_{split}_{i}")
                payload["domains"].append(domain)
                payload["splits"].append(split)
                payload["features"].append(torch.randn(4, generator=generator))
    payload["features"] = torch.nn.functional.normalize(torch.stack(payload["features"]), dim=-1)
    payload["fingerprint"] = descriptor_digest(payload)
    torch.save(payload, path)
    return payload


def test_structure_descriptor_is_channel_rotation_invariant_but_spatially_sensitive():
    torch.manual_seed(7)
    tokens = torch.randn(3, 16, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    expected = patch_structure_descriptor(tokens)
    torch.testing.assert_close(patch_structure_descriptor(tokens @ rotation), expected)
    assert not torch.allclose(patch_structure_descriptor(tokens[:, torch.randperm(16)]), expected)


def test_neighborhood_loss_excludes_self_pairs_and_detaches_teacher():
    torch.manual_seed(3)
    teacher = torch.randn(8, 4, requires_grad=True)
    current = torch.randn(8, 5, requires_grad=True)
    loss = neighborhood_distillation_loss(current, teacher)
    loss.backward()
    assert current.grad.norm() > 0
    assert teacher.grad is None
    assert abs(float(neighborhood_distillation_loss(teacher.detach(), teacher))) < 1e-6
    assert float(neighborhood_distillation_loss(teacher.detach().roll(1, 0), teacher)) > .01


def test_bank_looks_up_ids_not_row_positions_and_rejects_split_leakage(tmp_path):
    payload = save_prior(tmp_path / "prior.pt")
    bank = SemanticPriorBank(tmp_path / "prior.pt")
    torch.testing.assert_close(bank.lookup(["cat_train_2", "cat_train_0"], "cat", "train"), payload["features"][[2, 0]])
    with pytest.raises(ValueError, match="domain/split"):
        bank.lookup(["cat_val_0"], "cat", "train")
    with pytest.raises(ValueError, match="missing"):
        bank.lookup(["not_present"], "cat", "train")
    payload["features"][0, 0] += 1
    torch.save(payload, tmp_path / "prior.pt")
    with pytest.raises(ValueError, match="fingerprint"):
        SemanticPriorBank(tmp_path / "prior.pt")


def test_resume_rejects_changed_prior_or_variant():
    config = {"infoot": {"variant": "fused"}, "semantic_prior": {"fingerprint": "a"}}
    validate_prior_resume(config, config)
    with pytest.raises(ValueError, match="Resume cannot change"):
        validate_prior_resume({}, config)
    changed = deepcopy(config)
    changed["semantic_prior"]["fingerprint"] = "b"
    with pytest.raises(ValueError, match="Resume cannot change"):
        validate_prior_resume(config, changed)


def test_fused_solver_matches_official_cost_plus_mi_recurrence():
    ot = pytest.importorskip("ot")
    import numpy as np
    from diffusion_ot.losses.infoot import solve_infoot, gaussian_kernel
    torch.manual_seed(21)
    x, y = torch.randn(4, 3, dtype=torch.float64), torch.randn(6, 5, dtype=torch.float64)
    cost = torch.rand(4, 6, dtype=torch.float64)
    kx, ky = gaussian_kernel(x).numpy(), gaussian_kernel(y).numpy()
    a, b = np.ones(4) / 4, np.ones(6) / 6
    p = np.outer(a, b)
    for _ in range(4):
        joint = kx @ p @ ky.T
        gradient = np.log(joint / np.outer(kx @ a, ky @ b)) + kx.T @ (p / joint) @ ky
        p = ot.sinkhorn(a, b, .7 * cost.numpy() - .3 * gradient, .2,
                        numItermax=2000, stopThr=1e-12)
    actual = solve_infoot(x, y, cross_cost=cost, cross_cost_weight=.7, mi_weight=.3,
                          entropy_epsilon=.2, inner_iterations=4,
                          projection_iterations=2000, projection_tolerance=1e-12)
    torch.testing.assert_close(actual.coupling, torch.from_numpy(p), atol=1e-8, rtol=1e-7)
    assert actual.objective == pytest.approx(float((actual.coupling * cost * .7).sum())
                                           - .3 * actual.mutual_information - .2 * actual.entropy)


def test_fused_prior_breaks_ambiguous_geometry_and_zero_cost_recovers_plain():
    from diffusion_ot.losses.infoot import solve_infoot, solve_plain_infoot, transport_diagnostics
    x = torch.zeros(3, 2, dtype=torch.float64)
    cost = torch.ones(3, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    cost[torch.arange(3), permutation] = 0
    settings = dict(inner_iterations=3, entropy_epsilon=.05, projection_tolerance=1e-12)
    fused = solve_infoot(x, x, cross_cost=cost, **settings)
    plain = solve_plain_infoot(x, x, **settings)
    assert torch.equal(fused.coupling.argmax(1), permutation)
    assert float((fused.coupling * cost).sum()) < .01
    torch.testing.assert_close(solve_infoot(x, x, cross_cost=cost, cross_cost_weight=0, **settings).coupling,
                               plain.coupling)
    assert transport_diagnostics(plain.coupling)["mean_row_effective_targets"] == pytest.approx(3)
    assert transport_diagnostics(fused.coupling)["mean_row_effective_targets"] < 1.01


def test_fixed_validation_is_repeatable_and_preserves_training_rng(monkeypatch):
    import diffusion_ot.training.train_joint_infoot as train
    encoder = torch.nn.Linear(3, 2)
    domain = SimpleNamespace(branch=SimpleNamespace(encoder=encoder), device="cpu", dtype=torch.float32)
    monkeypatch.setattr(train, "_reconstruction_loss", lambda domain, x, z: (z * torch.rand_like(z)).square().mean())
    inputs = {"cat": torch.randn(4, 3)}
    rng = torch.get_rng_state().clone()
    args = ({"cat": domain}, {"cat": deepcopy(encoder)}, inputs)
    first = train.fixed_reconstruction_probe(*args, seed=10)
    assert first == train.fixed_reconstruction_probe(*args, seed=10)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert encoder.training
    assert first["cat_raw_reconstruction"] == first["cat_stage1a_reconstruction"]


def test_structure_cache_builder_uses_training_only_calibration(monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys
    import yaml
    from PIL import Image
    import numpy as np
    import diffusion_ot.data.afhq as afhq

    class Batch(dict):
        def to(self, device):
            return Batch({k: v.to(device) for k, v in self.items()})

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to_dict(self):
            return {"test_processor": True, "resample": Image.Resampling.BICUBIC}
        def __call__(self, images, **kwargs):
            assert kwargs["do_rescale"] is False
            return Batch(pixel_values=torch.stack(images))

    class Model(torch.nn.Module):
        config = SimpleNamespace(_commit_hash="test_revision")
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def forward(self, pixel_values):
            tokens = pixel_values.flatten(2).transpose(1, 2)
            return SimpleNamespace(last_hidden_state=torch.cat((tokens[:, :1], tokens), 1))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoImageProcessor=Processor, AutoModel=Model))
    images = []
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    val_indices = []
    for domain in ("cat", "dog"):
        for split in ("train", "val"):
            records = []
            for i in range(4):
                index = len(images)
                pixels = np.random.default_rng(index).integers(0, 255, (4, 4, 3), dtype=np.uint8)
                images.append({"image": Image.fromarray(pixels)})
                records.append({"hf_index": index, "sample_id": f"{domain}_{split}_{i}", "image_column": "image"})
                if split == "val":
                    val_indices.append(index)
            (manifests / f"{domain}_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    monkeypatch.setattr(afhq, "load_afhq_dataset", lambda path: images)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump({"project_root": str(tmp_path), "manifest_dir": "manifests", "image_size": 4}))
    script_path = Path(__file__).resolve().parents[1] / "scripts/cache_infoot_structure.py"
    spec = importlib.util.spec_from_file_location("cache_structure_test", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for index, output in enumerate(("first.pt", "second.pt")):
        if index:
            for i in val_indices:
                images[i]["image"] = images[i]["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        monkeypatch.setattr(sys, "argv", [str(script_path), "--data-config", str(config_path),
                                          "--output", output, "--device", "cpu"])
        assert script.main() == 0
    first, second = SemanticPriorBank(tmp_path / "first.pt"), SemanticPriorBank(tmp_path / "second.pt")
    assert first.cost_scale == second.cost_scale
    assert first.fingerprint != second.fingerprint
    assert first.metadata["resolved_revision"] == "test_revision"


def test_fused_training_resume_and_evaluation_pipeline(monkeypatch, tmp_path):
    """Exercise real orchestration and autograd, replacing only datasets/SiT."""
    import yaml
    import diffusion_ot.data.latent_dataset as data
    import diffusion_ot.training.train_joint_infoot as train
    import diffusion_ot.evaluation.stage1b_eval as evaluation

    class Dataset:
        def __init__(self, config, domain, split="train", **kwargs):
            self.domain, self.split = domain, split
            self.records = [{"sample_id": f"{domain}_{split}_{i}", "latent_path": "unused"} for i in range(8)]
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            return dict(x0_latent=torch.tensor([float(i) / 8, (i % 3) / 3, .5]),
                        sample_id=self.records[i]["sample_id"], domain=self.domain,
                        split=self.split, latent_path="unused", metadata=self.records[i])

    class Generator(torch.nn.Linear):
        def trainable_state_dict(self):
            return self.state_dict()

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)
            self.semantic_transformer = Generator(2, 3)

    originals = {d: Branch() for d in ("cat", "dog")}
    latest_domains = {}
    def load_domain(config, root, domain, **kwargs):
        branch = deepcopy(originals[domain])
        train.freeze_generator_train_encoder(branch)
        value = SimpleNamespace(branch=branch, transformer=branch.semantic_transformer,
                                training_config={}, device="cpu", dtype=torch.float32,
                                checkpoint_path=tmp_path / f"{domain}.pt", checkpoint_step=100,
                                stage1a_architecture={"semantic_cfg_enabled": True,
                                                     "attention_lora": {"enabled": True}},
                                data_config_path=tmp_path / "data.yaml")
        latest_domains[domain] = value
        return value

    monkeypatch.setattr(data, "CachedLatentDataset", Dataset)
    monkeypatch.setattr(train, "_load_training_domain", load_domain)
    monkeypatch.setattr(train, "_reconstruction_loss", lambda d, x, z: (d.branch.semantic_transformer(z)-x).square().mean())
    save_prior(tmp_path / "prior.pt")
    for d in originals:
        (tmp_path / f"{d}.yaml").write_text(yaml.safe_dump({"data_config": "data.yaml"}))
    config = {"stage": "stage1b_fused_infoot", "project_root": str(tmp_path), "output_dir": "out",
              "stage1a": {d: {"config": f"{d}.yaml"} for d in originals},
              "data": {"transport_batch_size": 8, "reconstruction_batch_size": 4, "num_workers": 0,
                       "random_horizontal_flip": 0, "pin_memory": False},
              "infoot": {"variant": "fused", "inner_iterations": 2, "entropy_epsilon": .1},
              "matching": {"bandwidth_multiplier": .7, "calibration_samples": 8},
              "loss_weights": {"semantic_neighborhood": .05},
              "semantic_prior": {"path": "prior.pt"}, "ema": {"decay": .995},
              "train": {"max_steps": 2, "log_every": 1, "save_every": 1, "validation_every": 1,
                        "keep_step_checkpoints": True, "gradient_diagnostics_every": 1}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    report = train.train_joint_infoot(path)
    for d in originals:
        assert not torch.equal(latest_domains[d].branch.encoder.weight, originals[d].encoder.weight)
        torch.testing.assert_close(latest_domains[d].branch.semantic_transformer.weight, originals[d].semantic_transformer.weight)
    payload = train._load_checkpoint(Path(report.checkpoint_path))
    assert payload["stage"] == "stage1b_fused_infoot"
    assert payload["config"]["semantic_prior"]["fingerprint"]
    assert (tmp_path / "out/checkpoints/step_000001.pt").is_file()
    resumed = train.train_joint_infoot(path, max_steps=3, resume_from="latest")
    assert resumed.initial_step == 2 and resumed.final_step == 3
    payload = train._load_checkpoint(Path(resumed.checkpoint_path))
    assert payload["ema_state"]["num_updates"] == 3

    def eval_context(config, root, domain, **kwargs):
        context = latest_domains[domain]
        context.checkpoint_path = Path(resumed.checkpoint_path)
        context.checkpoint_step = 3
        context.weights = "raw"
        context.stage1a_checkpoint_path = tmp_path / f"{domain}.pt"
        context.stage1a_checkpoint_step = 100
        context.stage1a_weights = "ema"
        return context
    monkeypatch.setattr(evaluation, "_load_domain_context", eval_context)
    monkeypatch.setattr(evaluation, "_dataset", lambda ctx, split, root: Dataset(None, "cat" if ctx is latest_domains["cat"] else "dog", split))
    eval_config = {"project_root": str(tmp_path), "output_dir": "eval", "alignment_device": "cpu",
                   "data": {"reference_samples_per_domain": 4, "query_samples_per_domain": 4,
                            "gallery_samples_per_domain": 4}, "matching": {"bandwidth_multiplier": .7},
                   "infoot": {"variant": "fused", "inner_iterations": 2},
                   "comparison": {"require_stage1a_baseline": False},
                   "reconstruction": {"enabled": False}, "translation": {"enabled": False},
                   "visualization": {"enabled": False}}
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config))
    result = evaluation.run_stage1b_evaluation(path, eval_path, checkpoint_path=resumed.checkpoint_path)
    assert result.solver["variant"] == "fused"
    assert result.solver["semantic_prior_fingerprint"] == payload["config"]["semantic_prior"]["fingerprint"]
    assert result.baseline_comparison["status"] == "missing"
    assert result.retrieval["dog_to_cat"]["projection_target_count"] == 8
