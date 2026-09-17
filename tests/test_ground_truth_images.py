from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from diffusion_ot.data.ground_truth import load_ground_truth_images


@pytest.fixture
def original_data(monkeypatch, tmp_path):
    import diffusion_ot.data.ground_truth as ground_truth

    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, 2:6] = [64, 128, 192]
    dataset = [{"image": Image.fromarray(pixels), "label": "cat"},
               {"image": Image.new("RGB", (4, 4), "white"), "label": "dog"}]
    monkeypatch.setattr(ground_truth, "load_afhq_dataset", lambda _: dataset)
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({"dataset_id": "test/AFHQ", "split": "train",
                                     "image_size": 4, "center_crop": True}))
    records = [{"dataset_id": "test/AFHQ", "dataset_split": "train", "hf_index": i,
                "image_column": "image", "domain": domain, "sample_id": f"afhq_{domain}_{i:06d}"}
               for i, domain in enumerate(("cat", "dog"))]
    return config, records


def test_original_images_use_cache_crop_and_preserve_record_order(original_data):
    config, records = original_data
    images = load_ground_truth_images(config, list(reversed(records)))
    assert images.shape == (2, 3, 4, 4)
    torch.testing.assert_close(images[0], torch.ones(3, 4, 4))
    expected = (torch.tensor([64, 128, 192]) / 255).view(3, 1, 1).expand(3, 4, 4)
    torch.testing.assert_close(images[1], expected)


@pytest.mark.parametrize("change,pattern", [
    ({"hf_index": -1}, "index out of range"),
    ({"hf_index": 2}, "index out of range"),
    ({"sample_id": "afhq_cat_000099"}, "identity mismatch"),
    ({"domain": "dog"}, "identity mismatch"),
    ({"dataset_id": "another/dataset"}, "dataset mismatch"),
    ({"dataset_split": "test"}, "dataset mismatch"),
])
def test_wrong_original_image_references_fail(original_data, change, pattern):
    config, records = original_data
    records = deepcopy(records)
    records[0].update(change)
    with pytest.raises(ValueError, match=pattern):
        load_ground_truth_images(config, records)


def test_missing_metadata_cannot_silently_use_a_vae_image(original_data):
    config, records = original_data
    del records[0]["hf_index"]
    with pytest.raises(ValueError, match="Original-image metadata missing"):
        load_ground_truth_images(config, records)
