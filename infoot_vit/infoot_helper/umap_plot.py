"""Optional training-bank-fitted UMAP; never changes transport or conditioning."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import random
import time

import torch

from .feature_bank import write_json


def _sample_points(features, ids, *, level, limit, seed, mask=None):
    """Sample by stable image ID and patch index, independently of bank order."""
    if (features.ndim != 3 or len(features) != len(ids) or min(features.shape) < 1
            or not features.is_floating_point() or len(set(ids)) != len(ids)):
        raise ValueError("UMAP needs floating-point [N,P,D] maps and unique ordered IDs.")
    valid = (torch.ones(features.shape[:2], dtype=torch.bool) if mask is None
             else mask.detach().cpu())
    if valid.dtype != torch.bool or valid.shape != features.shape[:2]:
        raise ValueError("UMAP token mask must be boolean [N,P].")
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    candidates = (valid[order].any(1).nonzero().flatten().tolist() if level == "image"
                  else valid[order].flatten().nonzero().flatten().tolist())
    chosen = sorted(random.Random(seed).sample(candidates, min(limit, len(candidates))))
    records, values = [], []
    for index in chosen:
        row, patch = (order[index], None) if level == "image" else (order[index // features.shape[1]], index % features.shape[1])
        if patch is None:
            value = features[row, valid[row].to(features.device)].detach().float().mean(0)
        else:
            value = features[row, patch].detach().float()
        values.append(value.cpu())
        records.append(dict(sample_id=ids[row], patch_index=patch, valid_tokens=int(valid[row].sum())))
    points = torch.stack(values) if values else torch.empty((0, features.shape[-1]))
    if not torch.isfinite(points).all():
        raise ValueError("UMAP input points contain nonfinite features.")
    return points, records


def _source_points(features, ids, records, mask):
    """Use the same query IDs/token positions as the mapped readout."""
    index = {sid: i for i, sid in enumerate(ids)}
    values = []
    for record in records:
        row, patch = index[record["sample_id"]], record["patch_index"]
        value = (features[row, mask[row].to(features.device)].detach().float().mean(0)
                 if patch is None else features[row, patch].detach().float())
        values.append(value.cpu())
    return torch.stack(values) if values else torch.empty((0, features.shape[-1]))


def save_umap(mapper, query_bank, result, output, *, level="image", max_points=1000,
              seed=42, n_neighbors=15, min_dist=.1, metric="euclidean", mapped_id=None):
    """Fit one display reducer on real training points, then transform queries.

    Image view averages only valid patches for display; patch view uses individual
    unchanged 768-D tokens. Source query means use the same valid positions as the
    mapped means. All-invalid queries have no plotted readout and are listed.
    """
    if (level not in {"image", "patch"} or type(max_points) is not int or max_points < 2
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(n_neighbors) is not int or n_neighbors < 2
            or not 0 <= min_dist <= 1 or metric not in {"euclidean", "cosine"}):
        raise ValueError("Invalid UMAP level, point limit, seed, neighbors, minimum distance or metric.")
    start = time.perf_counter()
    count = len(result.mapped_features)
    query_ids = query_bank.ids[:count]
    if (count < 1 or count > len(query_bank.ids)
            or result.mapped_features.shape != query_bank.features[:count].shape
            or mapper.source.features.shape[1:] != mapper.target.features.shape[1:]
            or mapper.source.features.shape[1:] != result.mapped_features.shape[1:]
            or [row["query_id"] for row in result.diagnostics["queries"]] != query_ids):
        raise ValueError("UMAP features must match the ordered mapping query IDs and representation.")
    if mapper.source.manifest["split"] != "train" or mapper.target.manifest["split"] != "train":
        raise ValueError("UMAP fitting uses real training banks only.")
    if query_bank.manifest["split"] not in {"val", "test"} or set(query_ids) & (set(mapper.source.ids) | set(mapper.target.ids)):
        raise ValueError("UMAP queries must be held out from both training banks.")
    try:
        from umap import UMAP
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise RuntimeError("Optional UMAP requires umap-learn and matplotlib. Install with: python -m pip install umap-learn matplotlib") from exc

    groups = {}
    for offset, (name, bank) in enumerate((("source_training", mapper.source), ("target_training", mapper.target))):
        groups[name] = _sample_points(bank.features, bank.ids, level=level, limit=max_points, seed=seed + offset)
    groups["mapped_query"] = _sample_points(result.mapped_features, query_ids, level=level,
        limit=max_points, seed=seed + 2, mask=result.valid_mask)
    groups["source_query"] = (_source_points(query_bank.features[:count], query_ids,
        groups["mapped_query"][1], result.valid_mask.detach().cpu()), groups["mapped_query"][1])
    training = torch.cat([groups[name][0] for name in ("source_training", "target_training")])
    if len(training) < 3:
        raise ValueError("UMAP needs at least three valid sampled training points.")
    kwargs = dict(n_components=2, n_neighbors=min(n_neighbors, len(training) - 1), min_dist=min_dist,
                  metric=metric, init="random", random_state=seed, transform_seed=seed, n_jobs=1)
    reducer = UMAP(**kwargs)
    embedded = torch.from_numpy(reducer.fit_transform(training.numpy().copy())).float()
    sizes = [len(groups[name][0]) for name in ("source_training", "target_training")]
    coordinates = dict(zip(("source_training", "target_training"), embedded.split(sizes)))
    # Transform the paired query groups together; their presence cannot alter the fit.
    queries = torch.cat([groups[name][0] for name in ("source_query", "mapped_query")])
    transformed = (torch.from_numpy(reducer.transform(queries.numpy().copy())).float()
                   if len(queries) else torch.empty((0, 2)))
    coordinates["source_query"], coordinates["mapped_query"] = transformed.split(len(queries) // 2) if len(queries) else (transformed, transformed)
    if any(v.shape != (len(groups[name][0]), 2) or not torch.isfinite(v).all() for name, v in coordinates.items()):
        raise RuntimeError("UMAP returned nonfinite or incorrectly shaped coordinates.")

    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    styles = dict(source_training=("#2563eb", "o", "Real source training", 12, .3),
                  target_training=("#f59e0b", "o", "Real target training", 12, .3),
                  source_query=("#1d4ed8", "o", "Source queries (valid positions)", 45, .9),
                  mapped_query=("#16a34a", "X", "Mapped queries", 65, .95))
    for name, values in coordinates.items():
        color, marker, label, size, alpha = styles[name]
        axis.scatter(values[:, 0], values[:, 1], c=color, marker=marker, s=size, alpha=alpha, label=label)
    if level == "image":
        query_index = {sid: i + 1 for i, sid in enumerate(query_ids)}
        for record, before, after in zip(groups["mapped_query"][1], coordinates["source_query"], coordinates["mapped_query"]):
            axis.annotate("", xy=after.tolist(), xytext=before.tolist(),
                          arrowprops=dict(arrowstyle="->", color="#64748b", alpha=.45, lw=.8))
            axis.annotate(str(query_index[record["sample_id"]]), after.tolist(), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"{mapper.mode}: {'valid-token image means' if level == 'image' else 'individual patch features'}")
    axis.legend(loc="best")
    figure.supxlabel("Fit: real training banks. Queries: transform only. Feature geometry is not an image-quality score.", fontsize=9)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "umap.png", dpi=160)
    report = dict(schema="infoot_umap_v1", mapper_id=mapper.manifest["artifact_id"],
        mapped_artifact_id=mapped_id, query_bank_id=query_bank.artifact_id, mode=mapper.mode,
        source_domain=mapper.source.manifest["domain"], target_domain=mapper.target.manifest["domain"],
        level=level, max_points_per_group=max_points, reducer=kwargs,
        versions={name: importlib.metadata.version(name) for name in ("umap-learn", "matplotlib")},
        fit_groups=["source_training", "target_training"], feature_normalization="none",
        projection=mapper.config["projection"], query_ids=query_ids,
        all_invalid_query_ids=[sid for sid, active in zip(query_ids, result.valid_mask.any(1).tolist()) if not active],
        groups={name: dict(points=[dict(record, xy=xy.tolist()) for record, xy in zip(groups[name][1], values)])
                for name, values in coordinates.items()},
        seconds=time.perf_counter() - start,
        interpretation="Display only. Image means use valid positions shared by source/mapped queries; patch view samples individual valid tokens. Sampling is without replacement by stable ID. No InfoOT refit, token pooling in transport, or conditioning changes. UMAP coordinates across separately fitted runs are not directly comparable.")
    write_json(output / "umap.json", report)
    return report
