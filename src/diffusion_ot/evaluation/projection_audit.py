"""Reproducible controls for interpreting projected decoder codes, not quality scores."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import textwrap
from typing import Any

import torch
import torch.nn.functional as F


AUDIT_VERSION = "projection_audit_v1"


def distribution(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().double().flatten().cpu()
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError("Audit distributions require nonempty finite values.")
    return {"count": values.numel(), "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)), "min": float(values.min()),
            "median": float(values.quantile(.5)), "p90": float(values.quantile(.9)),
            "max": float(values.max())}


def code_geometry(codes: torch.Tensor, targets: torch.Tensor, *,
                  exclude_self: bool = False) -> tuple[dict, dict]:
    """Chunked high-dimensional NN controls; target-bank rows exclude themselves."""
    codes, targets = codes.detach().float().cpu(), targets.detach().float().cpu()
    if codes.ndim != 2 or targets.ndim != 2 or not len(codes) or not len(targets):
        raise ValueError("Audit codes and target bank must be nonempty matrices.")
    if not torch.isfinite(codes).all() or not torch.isfinite(targets).all():
        raise ValueError("Audit codes must be finite.")
    if exclude_self and (len(targets) < 2 or not torch.equal(codes, targets)):
        raise ValueError("Self exclusion requires the identical ordered bank with at least two rows.")
    centered = codes - codes.mean(0)
    energy = torch.linalg.svdvals(centered).square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    rank = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp())
    stats = {"norm": distribution(codes.norm(dim=1)),
             "total_variance": float(centered.square().sum(1).mean()),
             "effective_rank": rank if float(energy.sum()) > 1e-12 else 0.,
             "zero_norm_rows": int((codes.norm(dim=1) <= 1e-12).sum()),
             "nn_excludes_self": exclude_self}
    tensors = {"norm": codes.norm(dim=1)}
    normalized_targets = F.normalize(targets, dim=1)
    for metric in ("euclidean", "cosine"):
        distances, indices = [], []
        for start in range(0, len(codes), 128):
            batch = codes[start:start + 128]
            costs = (torch.cdist(batch, targets, compute_mode="donot_use_mm_for_euclid_dist")
                     if metric == "euclidean" else
                     (1 - F.normalize(batch, dim=1) @ normalized_targets.T).clamp(0, 2))
            if exclude_self:
                costs[torch.arange(len(batch)), torch.arange(start, start + len(batch))] = torch.inf
            values, nearest = costs.min(1)
            distances.append(values)
            indices.append(nearest)
        tensors[f"nn_{metric}"] = torch.cat(distances)
        tensors[f"nn_{metric}_indices"] = torch.cat(indices)
        stats[f"nn_{metric}"] = distribution(tensors[f"nn_{metric}"])
    return stats, tensors


@torch.inference_mode()
def decoder_representations(z_proj: torch.nn.Module, groups: dict[str, torch.Tensor], *,
                            batch_size: int = 128) -> tuple[dict, dict]:
    """Use the selected checkpoint's affine LayerNorm/MLP without altering it."""
    if not isinstance(z_proj, torch.nn.Sequential) or not isinstance(z_proj[0], torch.nn.LayerNorm):
        raise ValueError("Projection audit requires the decoder's actual z_proj starting with LayerNorm.")
    if z_proj.training:
        raise ValueError("Projection audit requires z_proj in evaluation mode.")
    parameter = next(z_proj.parameters())
    result = {"raw": {}, "post_layernorm": {}, "post_z_proj": {}}
    for name, values in groups.items():
        result["raw"][name] = values.detach().float().cpu()
        normalized, projected = [], []
        for chunk in values.split(batch_size):
            chunk = chunk.to(device=parameter.device, dtype=parameter.dtype)
            normalized.append(z_proj[0](chunk).float().cpu())
            projected.append(z_proj(chunk).float().cpu())
        result["post_layernorm"][name] = torch.cat(normalized)
        result["post_z_proj"][name] = torch.cat(projected)
    layer = z_proj[0]
    state = {"layernorm_eps": layer.eps, "normalized_shape": list(layer.normalized_shape),
             "elementwise_affine": layer.elementwise_affine,
             "compute_dtype": str(parameter.dtype),
             "z_proj_state": {k: v.detach().cpu().clone() for k, v in z_proj.state_dict().items()}}
    return result, state


def target_fitted_pca(groups: dict[str, torch.Tensor], *, normalize: bool) -> dict:
    values = {k: (F.normalize(v.double(), dim=1) if normalize else v.double())
              for k, v in groups.items()}
    target = values["target_bank"]
    mean = target.mean(0)
    _, singular, vh = torch.linalg.svd(target - mean, full_matrices=False)
    components = vh[:2]
    # Fix the arbitrary SVD sign for stable artifacts.
    signs = components[torch.arange(len(components)), components.abs().argmax(1)].sign()
    components = components * signs[:, None]
    embedding = {k: (v - mean) @ components.T for k, v in values.items()}
    embedding = {k: F.pad(v, (0, 2 - v.shape[1])).float() for k, v in embedding.items()}
    energy = singular.square()
    return {"fit_group": "target_bank", "normalization": "l2" if normalize else "none",
            "mean": mean, "components": components,
            "explained_variance_ratio": energy[:2] / energy.sum().clamp_min(1e-12),
            "embeddings": embedding}


def save_reducer(path: Path, reducer: Any) -> dict:
    """Pickle is a local reproducibility artifact; never load untrusted reducers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        version = importlib.metadata.version("umap-learn")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {"umap_version": version, "parameters": reducer.get_params(deep=False),
            "reducer_path": str(path), "reducer_format": "trusted_local_pickle"}


def _plot_controls(embeddings: dict[str, torch.Tensor], *, title: str, footer: str,
                   path: Path, fit_subset_indices: torch.Tensor | None = None) -> None:
    import matplotlib.pyplot as plt

    styles = {"conditional": ("#D55E00", "X", "Cross-domain conditional mean"),
              "barycentric": ("#009E73", "P", "Nearest-reference plan readout"),
              "real_target_query": ("#7C3AED", "^", "Held-out real targets"),
              "same_domain_conditional": ("#E69F00", "s", "Same-domain identity-plan readout"),
              "bank_subset": ("#CC79A7", "x", "Shuffled bank subset transformed"),
              "selected_targets": ("#332288", "+", "Exact selected targets transformed")}
    names = [k for k in styles if k in embeddings]
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
    target = embeddings["target_bank"].numpy()
    all_points = torch.cat(list(embeddings.values())).numpy()
    low, high = all_points.min(0), all_points.max(0)
    pad = (high - low).clip(min=.1) * .06
    for axis, name in zip(axes.flat, names):
        color, marker, label = styles[name]
        value = embeddings[name].numpy()
        axis.scatter(target[:, 0], target[:, 1], s=10, c="#2563EB", alpha=.30, label="Fitted target bank")
        axis.scatter(value[:, 0], value[:, 1], s=26, c=color, marker=marker, alpha=.85, label=label)
        if name == "bank_subset" and fit_subset_indices is not None:
            original = target[fit_subset_indices.numpy()]
            for a, b in zip(original, value):
                axis.plot([a[0], b[0]], [a[1], b[1]], color="#777777", linewidth=.4, alpha=.5)
        axis.set_title(label, fontsize=10)
        axis.set_xlim(low[0] - pad[0], high[0] + pad[0])
        axis.set_ylim(low[1] - pad[1], high[1] + pad[1])
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=13)
    figure.text(.5, .015, textwrap.fill(footer, width=155), ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, .09, 1, .96))
    figure.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.15)
    plt.close(figure)


def save_projection_audit(*, output_dir: Path, direction: str, groups: dict[str, torch.Tensor],
                          group_ids: dict[str, list[str]], z_proj: torch.nn.Module,
                          tensors: dict[str, torch.Tensor], protocol: dict,
                          visualize: bool, random_state: int, n_neighbors: int = 15,
                          min_dist: float = .1, n_jobs: int = 1) -> tuple[dict, dict[str, str]]:
    """Persist full numeric controls even when plot generation is disabled."""
    if n_neighbors < 2 or not math.isfinite(min_dist) or not 0 <= min_dist <= 1:
        raise ValueError("Audit UMAP requires n_neighbors >= 2 and min_dist in [0, 1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(groups["target_bank"]) < 3:
        raise ValueError("Projection audit requires at least three target training codes.")
    if set(group_ids["target_bank"]) & set(group_ids["real_target_query"]):
        raise ValueError("Real-target controls must be disjoint from the target training bank.")
    if set(groups) != set(group_ids) or any(len(v) != len(group_ids[k]) for k, v in groups.items()):
        raise ValueError("Audit code groups and ordered sample IDs disagree.")
    representations, decoder_state = decoder_representations(z_proj, groups)
    report = {"version": AUDIT_VERSION, "direction": direction, "protocol": protocol,
              "representations": {}, "umap": {},
              "interpretation": "Geometry and transform controls only; not semantic correctness or checkpoint-selection scores."}
    payload = {"version": AUDIT_VERSION, "protocol": protocol, "group_ids": group_ids,
               "representations": representations, "decoder": decoder_state,
               "projection_tensors": {k: v.detach().cpu() for k, v in tensors.items()},
               "geometry": {}, "pca": {}, "umap": {}}
    paths = {}
    generator = torch.Generator().manual_seed(random_state)
    # A strict subset bypasses UMAP's full-array identity/hash shortcut.
    subset = torch.randperm(len(groups["target_bank"]), generator=generator)[:min(64, len(groups["target_bank"]) - 1)]
    payload["bank_subset_indices"] = subset
    selected = tensors["conditional_weights"].argmax(1)
    payload["selected_target_indices"] = selected
    payload["bank_subset_ids"] = [group_ids["target_bank"][i] for i in subset.tolist()]
    payload["selected_target_ids"] = [group_ids["target_bank"][i] for i in selected.tolist()]

    for space, values in representations.items():
        report["representations"][space], payload["geometry"][space] = {}, {}
        for name, codes in values.items():
            stats, samples = code_geometry(codes, values["target_bank"], exclude_self=name == "target_bank")
            report["representations"][space][name] = stats
            payload["geometry"][space][name] = samples
        plot_groups = {**values, "bank_subset": values["target_bank"][subset],
                       "selected_targets": values["target_bank"][selected]}
        for metric in ("euclidean", "cosine"):
            key = f"{space}_{metric}"
            pca = target_fitted_pca(plot_groups, normalize=metric == "cosine")
            payload["pca"][key] = pca
            if not visualize:
                continue
            path = output_dir / f"{key}_pca.png"
            _plot_controls(pca["embeddings"], title=f"{direction}: target-fitted PCA, {space}, {metric}",
                           footer="PCA fit uses only training target codes. Cosine view applies L2 normalization before linear PCA. All panels share axes; overlap is not a semantic score.", path=path)
            paths[f"{key}_pca"] = str(path)
        if not visualize:
            continue
        import umap
        for metric in ("euclidean", "cosine"):
            key = f"{space}_{metric}"
            fit_values = {k: (F.normalize(v, dim=1) if metric == "cosine" else v) for k, v in plot_groups.items()}
            kwargs = dict(n_components=2, n_neighbors=min(n_neighbors, len(values["target_bank"]) - 1),
                          min_dist=min_dist, metric=metric, init="random", random_state=random_state,
                          transform_seed=random_state, n_jobs=n_jobs)
            reducer = umap.UMAP(**kwargs)
            embedded = {"target_bank": torch.from_numpy(reducer.fit_transform(fit_values["target_bank"].numpy().copy())).clone()}
            embedded.update({k: torch.from_numpy(reducer.transform(v.numpy().copy())).clone()
                             for k, v in fit_values.items() if k != "target_bank"})
            details = save_reducer(output_dir / f"{key}_target_reducer.pkl", reducer)
            details.update(fit_ids=group_ids["target_bank"], fit_group="target_bank",
                           normalization="l2" if metric == "cosine" else "none")
            displacement = {"bank_subset": (embedded["bank_subset"] - embedded["target_bank"][subset]).norm(dim=1),
                            "selected_targets": (embedded["selected_targets"] - embedded["target_bank"][selected]).norm(dim=1)}
            report["umap"][key] = {**details, "fit_transform_displacement": {k: distribution(v) for k, v in displacement.items()}}
            payload["umap"][key] = {"embeddings": embedded, "displacement": displacement}
            path = output_dir / f"{key}_umap.png"
            _plot_controls(embedded, title=f"{direction}: target-fitted UMAP, {space}, {metric}",
                           footer="Only target-bank codes participate in fitting. Other groups use transform. Lines join shuffled bank IDs to their fitted locations; displacement measures a transform artifact, not semantic error.",
                           path=path, fit_subset_indices=subset)
            paths[f"{key}_umap"] = str(path)

    # Joint UMAP is deliberately limited to raw-code views and explicitly transductive.
    if visualize:
        for metric in ("euclidean", "cosine"):
            values = representations["raw"]
            names = list(values)
            merged = torch.cat([values[k] for k in names])
            if metric == "cosine":
                merged = F.normalize(merged, dim=1)
            reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(merged) - 1),
                                min_dist=min_dist, metric=metric, init="random", random_state=random_state,
                                transform_seed=random_state, n_jobs=n_jobs)
            joint = torch.from_numpy(reducer.fit_transform(merged.numpy().copy())).clone()
            chunks = joint.split([len(values[k]) for k in names])
            embedded = dict(zip(names, chunks))
            key = f"raw_{metric}_joint"
            details = save_reducer(output_dir / f"{key}_reducer.pkl", reducer)
            report["umap"][key] = {**details, "transductive": True, "use_for_checkpoint_selection": False,
                                    "fit_groups": names}
            payload["umap"][key] = {"embeddings": embedded}
            path = output_dir / f"{key}_umap.png"
            _plot_controls(embedded, title=f"{direction}: JOINT exploratory UMAP, raw, {metric}",
                           footer="TRANSDUCTIVE: real and projected codes all participate in fitting. This layout is exploratory only and must never be used for checkpoint selection.", path=path)
            paths[f"{key}_umap"] = str(path)
    tensor_path = output_dir / "projection_audit.pt"
    report["tensor_path"] = str(tensor_path)
    report["decoder"] = {k: v for k, v in decoder_state.items() if k != "z_proj_state"}
    payload["report"] = report
    torch.save(payload, tensor_path)
    (output_dir / "projection_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, paths
