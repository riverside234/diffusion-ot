from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

import pytest
import torch
import yaml

from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from test_local_infoot_pipeline import local_helpers
from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool


@pytest.fixture
def cnn_project(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(52)
    models, checkpoints = {}, {}
    for domain in ("cat", "dog"):
        config = dict(tiny_config(), domain=domain, project_root=str(tmp_path),
                      model_config="model.yaml", data_config="data.yaml",
                      class_conditioning={"null_label": None})
        config["encoder"]["output_batchnorm"] = True
        branch = build_pdae_sit_branch(FakeSiT(), stage_config=config)
        with torch.no_grad():
            branch.encode(torch.randn(12, 4, 8, 8))
        models[domain] = branch.eval()
        path = tmp_path / f"outputs/pdae_cnn_{domain}/checkpoints/latest.pt"
        path.parent.mkdir(parents=True)
        torch.save({"model": branch.pdae_state_dict(), "config": config,
                    "step": 2000, "domain": domain, "optimizer": {}}, path)
        checkpoints[domain] = path
        for split, count in (("train", 5), ("val", 16)):
            folder = tmp_path / "data/latents/afhq_sit_b2_256" / f"{domain}_{split}"
            folder.mkdir(parents=True)
            for index in range(count):
                latent = torch.randn(4, 8, 8)
                torch.save({"x0_latent": latent[None] if split == "train" else latent},
                           folder / f"{index:03d}.pt")
    return tmp_path, models, checkpoints


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_bank_cli_restores_batchnorm_and_pins_checkpoint(local_helpers, cnn_project, monkeypatch, domain):
    from diffusion_ot.data.latent_dataset import load_latent_tensor

    root, models, checkpoints = cnn_project
    original = Path(__file__).resolve().parents[1] / f"infoot/bank/bank_{domain}.py"
    script = root / f"infoot/bank/bank_{domain}.py"
    script.parent.mkdir(parents=True)
    script.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script), "--checkpoint", str(checkpoints[domain]),
                                     "--batch-size", "2"])
    runpy.run_path(str(script), run_name="__main__")
    bank = torch.load(root / f"data/infoot_test/{domain}_bank.pt", weights_only=True)
    with torch.no_grad():
        expected = torch.cat([
            models[domain].encode(load_latent_tensor(path)) for path in bank["latent_paths"]
        ])
    torch.testing.assert_close(bank["v_bank"], expected, rtol=1e-4, atol=1e-5)
    saved = torch.load(bank["checkpoint_path"], weights_only=True)
    assert Path(bank["checkpoint_path"]) != checkpoints[domain]
    assert bank["source_checkpoint_path"] == str(checkpoints[domain])
    assert "optimizer" not in saved
    assert yaml.safe_load(Path(bank["config_path"]).read_text())["encoder"]["output_batchnorm"]
    torch.testing.assert_close(saved["model"]["encoder"], models[domain].encoder.state_dict(), rtol=0, atol=0)
    assert saved["model"]["encoder"]["output_norm.num_batches_tracked"].item() == 1


def test_new_cnn_bank_fit_and_test_pipeline(local_helpers, cnn_project, monkeypatch):
    from diffusion_ot.data import ground_truth, manifests, latent_dataset
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.integrations import sit_diffusers
    from bank import create_bank
    from infoot_helper import umap_plot
    import torchvision.utils

    root, models, checkpoints = cnn_project
    for domain, path in checkpoints.items():
        create_bank(root, domain, path, batch_size=3)
        changed = torch.load(path, weights_only=True)
        changed["model"]["encoder"]["output_norm.running_mean"].add_(100)
        torch.save(changed, path)
    (root / "model.yaml").write_text("pretrained: pretrained.yaml\n")
    evaluation = root / "configs/stage1a_eval/residual_sit_b2_256.yaml"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(yaml.safe_dump({"project_root": str(root),
                                         "sampling": {"variants": ["correct_z"]}}))
    infoot, _ = local_helpers
    bank_dir = root / "data/infoot_test"
    banks = {name: torch.load(bank_dir / f"{name}_bank.pt", weights_only=True) for name in models}
    captured = {}
    integrate_flow = stage1a_eval.integrate_pdae_flow

    def integrate(branch, transformer, noise, v, **kwargs):
        captured["mapped"] = v
        torch.testing.assert_close(branch.encoder.output_norm.running_mean,
                                   models["dog"].encoder.output_norm.running_mean)
        assert not branch.encoder.training and noise.shape == (len(v), 4, 32, 32)
        torch.testing.assert_close(noise, noise[:1].expand_as(noise), rtol=0, atol=0)
        return integrate_flow(branch, transformer, noise, v, **kwargs)

    def save_grid(grid, path, **kwargs):
        assert grid.shape == (32, 3, 32, 32) and kwargs["nrow"] == 16
        assert Path(path).name == "cat_to_dog_test_cnn_bn.png"
        captured["grid"] = True

    def save_umap(source, target, queries, mapped, path, **kwargs):
        torch.testing.assert_close(source, banks["cat"]["v_bank"])
        torch.testing.assert_close(target, banks["dog"]["v_bank"])
        paths = sorted((root / "data/latents/afhq_sit_b2_256/cat_val").glob("*.pt"))
        x0 = torch.stack([latent_dataset.load_latent_tensor(path) for path in paths])
        with torch.no_grad():
            torch.testing.assert_close(queries, models["cat"].encode(x0))
        torch.testing.assert_close(mapped, captured["mapped"])
        captured["umap"] = True
        return path

    def load_components(*args, **kwargs):
        transformer = FakeSiT()
        transformer.pos_embed = torch.nn.Parameter(torch.zeros(1, 256, 8))
        return SimpleNamespace(transformer=transformer, vae=torch.nn.Identity())

    monkeypatch.setattr(sit_diffusers, "load_sit_components", load_components)
    monkeypatch.setattr(latent_dataset, "CachedLatentDataset", lambda **kwargs: [])
    monkeypatch.setattr(stage1a_eval, "integrate_pdae_flow", integrate)
    monkeypatch.setattr(stage1a_eval, "decode_vae_latents", lambda vae, x: x[:, :3])
    monkeypatch.setattr(torchvision.utils, "save_image", save_grid)
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda *a, **k: torch.ones(16, 3, 32, 32))
    monkeypatch.setattr(manifests, "read_jsonl", lambda path: [{"sample_id": f"{i:03d}"} for i in range(16)])
    monkeypatch.setattr(umap_plot, "save_umap", save_umap)
    project = Path(__file__).resolve().parents[1]
    for filename, arguments in (
        ("infoot_fit.py", ["--h", "0.8", "--reg", "0.5", "--lam", "0.1"]),
        ("infoot_test.py", ["--save", "cnn_bn"]),
    ):
        script = root / "infoot" / filename
        script.parent.mkdir(exist_ok=True)
        script.write_text((project / "infoot" / filename).read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [str(script), *arguments])
        runpy.run_path(str(script), run_name="__main__")
    plan = torch.load(bank_dir / "cat_to_dog_plan.pt", weights_only=True)
    assert plan["feature_space"] == "raw" and torch.isfinite(plan["P"]).all()
    assert captured["grid"] and captured["umap"]
