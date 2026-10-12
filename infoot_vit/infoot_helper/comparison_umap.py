"""One PNG for a matched projection comparison; coordinates live in its report."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import time

import torch

from .feature_bank import digest, file_hash
from .umap_plot import ARROW_STYLE, POINT_STYLES, _sample_points, _source_points, draw_umap_panel


MODES = ("barycentric", "conditional_displacement", "conditional_edge_mean")
TITLES = ("Barycentric reference", "Displacement: residual retained", "Edge mean: residual removed")


def require_dependencies():
    """Fail before mapping/decoding if the required UMAP cannot be produced."""
    try:
        from umap import UMAP
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise RuntimeError("The residual experiment requires umap-learn and matplotlib. "
                           "Install with: python -m pip install umap-learn matplotlib") from exc
    return UMAP, Figure, FigureCanvasAgg


def _point_hash(points):
    return hashlib.sha256(points.contiguous().numpy().tobytes()).hexdigest()


def save_comparison_umap(mapper, query_bank, results, output, *, mapped_ids,
                         max_points=1000, seed=42, n_neighbors=15, min_dist=.1):
    """Training-only reducers, shared per level by every estimator, no refitting OT.

    Image means are display readouts, never pooled transport inputs. Patch panels
    use exactly the same valid query token positions for all estimators. UMAPs
    for image means and patches are separate spaces and cannot be compared by xy.
    """
    if (type(max_points) is not int or max_points < 2 or type(seed) is not int
            or not 0 <= seed < 2**32 or type(n_neighbors) is not int or n_neighbors < 2
            or not 0 <= min_dist <= 1):
        raise ValueError("Invalid comparison UMAP point limit, seed, neighbors or min_dist.")
    if set(results) != set(MODES) or set(mapped_ids) != set(MODES):
        raise ValueError("Comparison UMAP requires all three matched projection modes.")
    if mapper.source.manifest["split"] != "train" or mapper.target.manifest["split"] != "train":
        raise ValueError("Comparison UMAP fitting requires training banks.")
    ids = query_bank.ids
    if (query_bank.manifest["split"] not in {"val", "test"}
            or set(ids) & (set(mapper.source.ids) | set(mapper.target.ids))):
        raise ValueError("Comparison UMAP queries must be held out from the training banks.")
    valid = results[MODES[0]].valid_mask.detach().cpu()
    if (mapper.source.representation != mapper.target.representation
            or query_bank.representation != mapper.source.representation
            or mapper.source.features.shape[1:] != query_bank.features.shape[1:]
            or mapper.target.features.shape[1:] != query_bank.features.shape[1:]):
        raise ValueError("Comparison UMAP representation or token shapes differ.")
    for result in results.values():
        if (result.mapped_features.shape != query_bank.features.shape
                or not result.mapped_features.is_floating_point()
                or not torch.equal(result.valid_mask.detach().cpu(), valid)
                or [r["query_id"] for r in result.diagnostics["queries"]] != ids):
            raise ValueError("Comparison UMAP requires identical ordered queries and masks.")
    path = Path(output) / "umap.png"
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing UMAP: {path}")
    UMAP, Figure, FigureCanvasAgg = require_dependencies()
    started = time.perf_counter()
    figure = Figure(figsize=(16, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, sharex="row", sharey="row")
    panels = {}
    for row, level in enumerate(("image", "patch")):
        print(f"UMAP: {level} readout, fit real training banks; transform all three methods", flush=True)
        groups = {}
        for offset, (name, bank) in enumerate((("source_training", mapper.source), ("target_training", mapper.target))):
            groups[name] = _sample_points(bank.features, bank.ids, level=level, limit=max_points, seed=seed + offset)
        # Include every valid held-out image in the image-mean display, even when
        # the training/patch sampling cap is smaller than the query count.
        groups["source_query"] = _sample_points(query_bank.features, ids, mask=valid,
            level=level, limit=len(ids) if level == "image" else max_points, seed=seed + 2)
        records = groups["source_query"][1]
        for mode in MODES:
            points = _source_points(results[mode].mapped_features, ids, records, valid)
            if not torch.isfinite(points).all():
                raise ValueError(f"Nonfinite UMAP features in {mode}.")
            groups[mode] = (points, records)
        training = torch.cat([groups[name][0] for name in ("source_training", "target_training")])
        if len(training) < 3:
            raise ValueError("Comparison UMAP requires at least three training points.")
        kwargs = dict(n_components=2, n_neighbors=min(n_neighbors, len(training) - 1),
            min_dist=min_dist, metric="euclidean", init="random", random_state=seed,
            transform_seed=seed, n_jobs=1)
        reducer = UMAP(**kwargs)
        embedded = torch.as_tensor(reducer.fit_transform(training.numpy().copy())).float()
        coordinates = dict(zip(("source_training", "target_training"),
            embedded.split([len(groups[name][0]) for name in ("source_training", "target_training")])))
        query_groups = ("source_query", *MODES)
        queries = torch.cat([groups[name][0] for name in query_groups])
        transformed = (torch.as_tensor(reducer.transform(queries.numpy().copy())).float()
                       if len(queries) else torch.empty((0, 2)))
        coordinates.update(zip(query_groups, transformed.split([len(groups[name][0]) for name in query_groups])))
        if any(xy.shape != (len(groups[name][0]), 2) or not torch.isfinite(xy).all()
               for name, xy in coordinates.items()):
            raise RuntimeError("UMAP returned nonfinite or incorrectly shaped coordinates.")
        for col, (mode, title) in enumerate(zip(MODES, TITLES)):
            # Reuse the same drawing code as infoot_test.py --umap.
            panel_coordinates = {name: coordinates[name] for name in
                                 ("source_training", "target_training", "source_query")}
            panel_coordinates["mapped_query"] = coordinates[mode]
            draw_umap_panel(axes[row, col], panel_coordinates, records, ids, level=level,
                title=f"{title}\n{'Valid-token image means' if level == 'image' else 'Matched individual patches'}",
                ylabel="UMAP 2" if col == 0 else "", legend_fontsize=8)
        panels[level] = dict(reducer=kwargs, fit_groups=["source_training", "target_training"],
            transform_groups=list(query_groups), training_points_sha256=_point_hash(training),
            groups={name: dict(features_sha256=_point_hash(points),
                points=[dict(record, xy=xy.tolist()) for record, xy in zip(records_, coordinates[name])])
                for name, (points, records_) in groups.items()})
    figure.suptitle("Residual ablation: one training-fitted UMAP per row", fontsize=15)
    figure.supxlabel("Columns share coordinates within each row. Image and patch spaces differ. "
                    "Numbers match image-grid columns.\nTraining anchors only fit the display; "
                    "all query variants use transform. Proximity is not a translation-quality score.", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    return dict(schema="infoot_projection_comparison_umap_v1", file=path.name, sha256=file_hash(path),
        mapper_id=mapper.manifest["artifact_id"], mapped_artifact_ids=mapped_ids,
        query_bank_id=query_bank.artifact_id, query_ids=ids,
        training_banks={name: dict(artifact_id=bank.artifact_id, selected_ids_sha256=digest(bank.ids))
                        for name, bank in (("source", mapper.source), ("target", mapper.target))},
        versions={name: importlib.metadata.version(name) for name in
                  ("umap-learn", "matplotlib", "numpy", "scikit-learn", "numba")},
        feature_space="raw encoder patch tokens; valid-token means for image display only",
        style=dict(source="infoot_vit.infoot_helper.umap_plot", point_styles=POINT_STYLES, arrow_style=ARROW_STYLE),
        feature_normalization="none", max_points_per_training_group_and_patch_variant=max_points,
        all_invalid_query_ids=[sid for sid, active in zip(ids, valid.any(1).tolist()) if not active],
        panels=panels, seconds=time.perf_counter() - started,
        interpretation="No query or mapped points fit either reducer. Matched query records/masks across methods. "
            "Image means are display-only; transport uses unchanged patch tokens. UMAP is not a semantic score, "
            "and coordinates cannot be compared between rows or separately fitted runs.")
