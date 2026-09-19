"""Pinned Clean-FID and original-source SSIM; galleries never enter metrics."""
from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from diffusion_ot.evaluation.offline_artifacts import file_hash, fingerprint


def metric_versions():
    required = {"clean-fid": "0.1.35", "scikit-image": "0.25.2", "scipy": "1.15.3"}
    for name, expected in required.items():
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f"Stage 4 requires {name}=={expected}, found {actual}; "
                               "install requirements-stage4.txt")
    return {name: version(name) for name in (*required, "numpy", "pillow", "torch", "torchvision")}


def image_name(sample_id):
    # Source IDs are metadata, not arbitrary filesystem paths.
    return fingerprint(str(sample_id))[:24] + ".png"


def save_rgb(path, pixels):
    if pixels.ndim != 3 or pixels.shape[0] != 3 or not torch.isfinite(pixels).all():
        raise ValueError("Expected finite CHW RGB image")
    array = pixels.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    Image.fromarray(array).save(temporary, format="PNG")
    temporary.replace(path)


def source_ssim(translation, source):
    from skimage.metrics import structural_similarity
    with Image.open(translation) as image:
        fake = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    with Image.open(source) as image:
        real = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    if fake.shape != real.shape or min(real.shape[:2]) < 11:
        raise ValueError("SSIM requires matching RGB image sizes of at least 11x11")
    value = float(structural_similarity(real, fake, data_range=1.0, channel_axis=-1,
                                        gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False))
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite SSIM")
    return value


def exact_image_files(folder, ids):
    paths = [Path(folder) / image_name(key) for key in ids]
    if len(set(paths)) != len(paths) or len(paths) < 2:
        raise ValueError("Metrics require at least two distinct source IDs")
    if set(Path(folder).iterdir()) != set(paths):
        raise ValueError(f"Incomplete or contaminated metric image folder: {folder}")
    return paths


class CleanFID:
    def __init__(self, cache_dir, *, device, batch_size=32, num_workers=4):
        from cleanfid.inception_torchscript import InceptionV3W
        self.versions = metric_versions()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.batch_size, self.num_workers = batch_size, num_workers
        # Same wrapper used by Clean-FID's clean feature extractor, explicit cache.
        self.model = InceptionV3W(str(self.cache_dir), download=True, resize_inside=False).eval().to(self.device)
        self.weight_hash = file_hash(self.cache_dir / "inception-2015-12-05.pt")

    def statistics(self, paths):
        from cleanfid import fid
        key = fingerprint({"files": [(p.name, file_hash(p)) for p in paths],
                           "versions": self.versions, "weights": self.weight_hash, "mode": "clean"})
        cache = self.cache_dir / f"stats_{key}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                return data["mu"], data["cov"]
        features = fid.get_files_features([str(p) for p in paths], model=self.model,
                                         device=self.device, mode="clean", batch_size=self.batch_size,
                                         num_workers=self.num_workers).astype(np.float64)
        if len(features) != len(paths) or not np.isfinite(features).all():
            raise ValueError("Invalid FID features")
        mu, cov = features.mean(0), np.cov(features, rowvar=False)
        temporary = cache.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, mu=mu, cov=cov)
        temporary.replace(cache)
        return mu, cov

    def compute(self, generated, real):
        from cleanfid import fid
        if min(len(generated), len(real)) < 2:
            raise ValueError("FID requires at least two images per distribution")
        value = float(fid.frechet_distance(*self.statistics(generated), *self.statistics(real)))
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite FID")
        return value


def save_gallery(output, direction, ids, source_folder, generated_folder):
    """Sixteen unresized 256px pairs, plus a two-column contact sheet."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    for key in ids:
        name = image_name(key)
        with Image.open(Path(source_folder) / name) as im:
            source = im.convert("RGB")
        with Image.open(Path(generated_folder) / name) as im:
            translated = im.convert("RGB")
        if source.size != translated.size:
            raise ValueError("Gallery source/translation dimensions differ")
        w, h = source.size
        panel = Image.new("RGB", (w * 2, h + 40), "white")
        panel.paste(source, (0, 40))
        panel.paste(translated, (w, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 3), f"{direction} | {key}", fill="black")
        draw.text((6, 22), "Source", fill="black")
        draw.text((w + 6, 22), "Translation", fill="black")
        panel.save(output / name)
        source.save(output / f"{Path(name).stem}_source.png")
        translated.save(output / f"{Path(name).stem}_translation.png")
        panels.append(panel)
    if not panels:
        raise ValueError("Gallery selection is empty")
    sheet = Image.new("RGB", (panels[0].width * 2, panels[0].height * math.ceil(len(panels)/2)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * panel.width, (i // 2) * panel.height))
    sheet.save(output / "contact_sheet.png")
