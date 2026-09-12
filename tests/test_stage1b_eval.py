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
    assert alignment["matching"]["bandwidth_multiplier"] == pytest.approx(0.55)
    assert alignment["matching"]["distance_scale"] == "infoot_rms"
    assert alignment["infoot"]["algorithm"] == "official_projected_sinkhorn"
    assert alignment["infoot"]["initialization"] == "independent_product"
    assert alignment["infoot"]["entropy_epsilon"] == pytest.approx(0.02)


def test_cfg_alignment_config_uses_cfg_lora_stage1a_checkpoints():
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
        assert stage1a["semantic_cfg"]["dropout_probability"] == pytest.approx(0.10)
        assert stage1a["adapter"]["lora"] is True
        assert stage1a["adapter"]["lora_rank"] == 64
        assert stage1a["adapter"]["lora_alpha"] == 64
        assert stage1a["adapter"]["injection_layers"] == list(range(12))
        assert stage1a["adapter"]["lora_layers"] == list(range(4, 12))
        assert stage1a["train"]["max_steps"] == 40000
        assert stage1a["train"]["lr_lora"] == pytest.approx(0.000025)
        assert "_cfg_adaln_all_lora_r64/" in domain_config["checkpoint"]
    assert alignment["stage1a"]["require_semantic_cfg"] is True
    assert alignment["stage1a"]["require_attention_lora"] is True
    assert alignment["stage1a"]["require_attention_lora_rank"] == 64
    assert alignment["stage1a"]["require_attention_lora_alpha"] == 64
    assert alignment["matching"]["bandwidth_multiplier"] == pytest.approx(0.55)
    assert alignment["matching"]["distance_scale"] == "infoot_rms"
    assert alignment["infoot"]["algorithm"] == "official_projected_sinkhorn"
    assert alignment["infoot"]["initialization"] == "independent_product"
    assert alignment["infoot"]["entropy_epsilon"] == pytest.approx(0.02)
    assert alignment["trainable"]["attention_lora"] is False


def test_quick_evaluation_uses_the_training_infoot_kernel_and_entropy():
    from pathlib import Path

    from diffusion_ot.integrations.hf_snapshot import load_yaml_config

    root = Path(__file__).resolve().parents[1]
    alignment = load_yaml_config(
        root / "configs" / "stage1b_infoot" / "plain_sit_b2.yaml"
    )
    evaluation = load_yaml_config(
        root / "configs" / "stage1b_eval" / "quick_sit_b2.yaml"
    )

    assert evaluation["matching"]["bandwidth_multiplier"] == pytest.approx(
        alignment["matching"]["bandwidth_multiplier"]
    )
    assert evaluation["matching"]["distance_scale"] == alignment["matching"][
        "distance_scale"
    ]
    assert evaluation["infoot"]["algorithm"] == alignment["infoot"]["algorithm"]
    assert evaluation["infoot"]["initialization"] == alignment["infoot"][
        "initialization"
    ]
    assert evaluation["infoot"]["mi_weight"] == pytest.approx(
        alignment["infoot"]["mi_weight"]
    )
    assert evaluation["infoot"]["entropy_epsilon"] == pytest.approx(
        alignment["infoot"]["entropy_epsilon"]
    )
    assert evaluation["output_dir"].endswith("_cfg_adaln_all_lora_r64")
    assert evaluation["proxy_labels"]["attributes"] == ["viewpoint", "framing"]
    assert evaluation["visualization"]["target_alpha"] == pytest.approx(0.18)
    assert evaluation["visualization"]["projection_alpha"] == pytest.approx(0.55)


def test_proxy_precision_caption_includes_every_rule_attribute_and_k():
    from diffusion_ot.evaluation.stage1b_eval import _proxy_precision_caption

    precision = {
        rule: {
            attribute: {
                "precision_at_1": 0.5,
                "precision_at_5": 0.4,
                "precision_at_15": None,
                "query_coverage": 0.75,
            }
            for attribute in ("viewpoint", "framing")
        }
        for rule in ("random", "conditional", "nn_plan_row", "nn_barycentric")
    }

    caption = _proxy_precision_caption(precision, ["viewpoint", "framing"])

    for rule in precision:
        assert f"{rule}:" in caption
    assert caption.count("viewpoint:") == 4
    assert caption.count("framing:") == 4
    assert caption.count("P@1=50.0%") == 8
    assert caption.count("P@5=40.0%") == 8
    assert caption.count("P@15=n/a") == 8
    assert caption.count("coverage=75.0%") == 8


def test_umap_visualization_writes_separate_labeled_readout_images(
    monkeypatch, tmp_path
):
    import sys
    from types import ModuleType, SimpleNamespace

    numpy = pytest.importorskip("numpy")
    from diffusion_ot.evaluation.stage1b_eval import _save_umap_visualizations

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, values):
            positions = numpy.arange(len(values), dtype=float)
            return numpy.column_stack((positions, positions * 0.5))

        def transform(self, values):
            positions = numpy.arange(len(values), dtype=float) + 0.25
            return numpy.column_stack((positions, positions * 0.5))

    class FakeAxis:
        def __init__(self):
            self.scatter_calls = []

        def scatter(self, *args, **kwargs):
            self.scatter_calls.append(kwargs)

        def legend(self, **kwargs):
            self.legend_kwargs = kwargs

        def set_title(self, title):
            self.title = title

        def set_xticks(self, ticks):
            self.xticks = ticks

        def set_yticks(self, ticks):
            self.yticks = ticks

    class FakeFigure:
        def __init__(self, axes):
            self.axes = axes
            self.caption = ""
            self.title = ""

        def suptitle(self, title, **kwargs):
            self.title = title

        def text(self, *args, **kwargs):
            self.caption = args[2]

        def tight_layout(self, **kwargs):
            self.layout_kwargs = kwargs

        def savefig(self, path, **kwargs):
            path.write_text(self.title + "\n" + self.caption, encoding="utf-8")

    figures = []
    pyplot = ModuleType("matplotlib.pyplot")

    def fake_subplots(rows, columns, **kwargs):
        axes = [FakeAxis() for _ in range(columns)]
        figure = FakeFigure(axes)
        figures.append(figure)
        return figure, numpy.asarray([axes], dtype=object)

    pyplot.subplots = fake_subplots
    pyplot.get_cmap = lambda name, count: lambda index: (
        index / max(count, 1), 0.2, 0.8, 1.0
    )
    pyplot.close = lambda figure: None
    lines = ModuleType("matplotlib.lines")
    lines.Line2D = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib.lines", lines)
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    target = _bank("cat", "train", ["t0", "t1", "t2"])
    source_ids = ["q0", "q1"]
    labels = {
        "t0": {"viewpoint": "front", "framing": "close"},
        "t1": {"viewpoint": "side", "framing": "wide"},
        "t2": {"viewpoint": "front", "framing": "wide"},
        "q0": {"viewpoint": "front", "framing": "close"},
        "q1": {"viewpoint": "side", "framing": "wide"},
    }
    precision = {
        "conditional": {
            attribute: {"precision_at_1": 0.5, "query_coverage": 1.0}
            for attribute in ("viewpoint", "framing")
        },
        "nn_barycentric": {
            attribute: {"precision_at_1": 0.5, "query_coverage": 1.0}
            for attribute in ("viewpoint", "framing")
        },
    }
    paths = {
        "conditional": tmp_path / "conditional.png",
        "barycentric": tmp_path / "barycentric.png",
    }

    _save_umap_visualizations(
        target,
        source_ids,
        torch.randn(2, 3),
        torch.randn(2, 3),
        labels=labels,
        attributes=["viewpoint", "framing"],
        precision=precision,
        direction="dog_to_cat",
        random_state=7,
        n_jobs=1,
        target_alpha=0.18,
        projection_alpha=0.55,
        output_paths=paths,
    )

    assert paths["conditional"].is_file()
    assert paths["barycentric"].is_file()
    assert paths["conditional"].read_bytes() != paths["barycentric"].read_bytes()
    assert len(figures) == 2
    assert all(
        [axis.title for axis in figure.axes] == ["Viewpoint", "Framing"]
        for figure in figures
    )
    assert all("P@1=50.0%" in figure.caption for figure in figures)
    plotted_alphas = {
        call["alpha"]
        for figure in figures
        for axis in figure.axes
        for call in axis.scatter_calls
    }
    assert plotted_alphas == {0.18, 0.55}


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


def test_direction_evaluation_uses_official_cross_matrix_kernel_scales():
    from diffusion_ot.evaluation.stage1b_eval import _direction_evaluation

    source_reference = _bank("cat", "train", ["cr0", "cr1"])
    target_reference = _bank("dog", "train", ["dr0", "dr1", "dr2"])
    source_query = _bank("cat", "val", ["cq0", "cq1"])
    target_gallery = _bank("dog", "val", ["dg0", "dg1"])
    target_projection = _bank("dog", "train", ["dp0", "dp1", "dp2", "dp3"])
    coupling = torch.tensor([[0.2, 0.1, 0.2], [0.1, 0.25, 0.15]])
    coupling /= coupling.sum()

    report, _ = _direction_evaluation(
        source_reference,
        target_reference,
        source_query,
        target_gallery,
        coupling,
        target_projection=target_projection,
        source_scale=99.0,
        target_scale=98.0,
        bandwidth=0.55,
        labels={},
        attributes=[],
        ks=[1],
        seed=5,
        eps=1.0e-8,
        distance_scale_mode="infoot_rms",
    )

    def official_scale(left, right):
        return float((torch.cdist(left, right).square().mean() / 2.0).sqrt())

    assert report["conditional_distance_scales"] == pytest.approx(
        {
            "query_source": official_scale(
                source_query.matching_features, source_reference.matching_features
            ),
            "gallery_target": official_scale(
                target_gallery.matching_features, target_reference.matching_features
            ),
            "projection_target": official_scale(
                target_projection.matching_features,
                target_reference.matching_features,
            ),
        }
    )


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


@pytest.mark.parametrize("configured,override,expected", [
    (None, None, 0.55), (0.4, None, 0.4), (0.4, 0.3, 0.3),
])
def test_projection_bandwidth_resolution(configured, override, expected):
    from diffusion_ot.evaluation.stage1b_eval import _evaluation_bandwidths

    config = {"bandwidth_multiplier": 0.55,
              "projection_bandwidth_multiplier": configured}
    assert _evaluation_bandwidths(config, override) == (0.55, expected)
    assert _evaluation_bandwidths({"bandwidth_multiplier": 0.55}) == (0.55, 0.55)


@pytest.mark.parametrize("invalid", [0, -0.1, float("nan"), float("inf")])
def test_projection_bandwidth_rejects_invalid_values(invalid):
    from diffusion_ot.evaluation.stage1b_eval import _evaluation_bandwidths

    with pytest.raises(ValueError, match="finite and positive"):
        _evaluation_bandwidths({"projection_bandwidth_multiplier": invalid})
    with pytest.raises(ValueError, match="finite and positive"):
        _evaluation_bandwidths({}, invalid)
    with pytest.raises(ValueError, match="finite and positive"):
        _evaluation_bandwidths({"bandwidth_multiplier": invalid})


def test_protocol_identifier_versions_projection_bandwidth():
    from diffusion_ot.evaluation.stage1b_eval import _protocol_identifier

    def protocol(width):
        return _protocol_identifier(
            {"matching": {"bandwidth_multiplier": 0.55,
                          "projection_bandwidth_multiplier": width}},
            max_reference=None, max_projection=None, max_query=None,
        )

    assert protocol(0.4) != protocol(0.55)
    assert protocol(0.4) == protocol(0.4)


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
@pytest.mark.parametrize("projection_bandwidth", [None, 0.35])
def test_decoded_grid_receives_full_equation7_mean(
    monkeypatch, tmp_path, reverse, projection_bandwidth,
):
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
        projection_bandwidth=projection_bandwidth,
    )
    effective_bandwidth = bandwidth if projection_bandwidth is None else projection_bandwidth
    kx = torch.exp(-torch.cdist(query.matching_features, source.matching_features).square()
                   / (2 * (effective_bandwidth * source_scale) ** 2))
    ky = torch.exp(-torch.cdist(target.matching_features, target.matching_features).square()
                   / (2 * (effective_bandwidth * target_scale) ** 2))
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
    assert report["fit_bandwidth_multiplier"] == bandwidth
    assert report["projection_bandwidth_multiplier"] == effective_bandwidth
    # The same coupling gives the same barycentric control for either width.
    default_report, default_tensors = stage1b._direction_evaluation(
        source, target, query, gallery, coupling,
        source_scale=source_scale, target_scale=target_scale, bandwidth=bandwidth,
        labels={}, attributes=[], ks=[1], seed=5, eps=1.0e-8,
    )
    torch.testing.assert_close(tensors["barycentric_codes"], default_tensors["barycentric_codes"])
    if projection_bandwidth is not None:
        assert not torch.allclose(tensors["conditional_codes"], default_tensors["conditional_codes"])

    # Retrieval uses the same projection bandwidth, evaluated on gallery points.
    kg = torch.exp(-torch.cdist(gallery.matching_features, target.matching_features).square()
                   / (2 * (effective_bandwidth * target_scale) ** 2))
    scores = (kx @ coupling @ kg.T) / ((kx @ a)[:, None] * (kg @ b)[None, :])
    expected_top = scores.argmax(dim=1).tolist()
    assert report["conditional_top_ids"] == {
        sample_id: [gallery.sample_ids[index]]
        for sample_id, index in zip(query.sample_ids, expected_top)
    }

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
