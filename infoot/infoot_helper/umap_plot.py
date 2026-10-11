from pathlib import Path

import torch


def save_umap(cat_bank, dog_bank, queries, mapped, output_path, *,
              title="InfoOT UMAP", max_points=2000, seed=0):
    from umap import UMAP
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if len(queries) != len(mapped) or not len(queries):
        raise ValueError("UMAP needs paired, nonempty query and mapped features.")
    generator = torch.Generator().manual_seed(seed)
    banks = [v.detach().float().cpu() for v in (cat_bank, dog_bank)]
    banks = [v[torch.randperm(len(v), generator=generator)[:max_points]] for v in banks]
    training = torch.cat(banks)
    if len(training) < 3:
        raise ValueError("UMAP needs at least three training features.")
    reducer = UMAP(n_components=2, n_neighbors=min(15, len(training) - 1),
                   min_dist=0.1, metric="euclidean", init="random",
                   random_state=seed, transform_seed=seed, n_jobs=1)
    reference_xy = reducer.fit_transform(training.numpy())
    query_xy = reducer.transform(torch.cat([
        v.detach().float().cpu() for v in (queries, mapped)
    ]).numpy())
    groups = (reference_xy[:len(banks[0])], reference_xy[len(banks[0]):],
              query_xy[:len(queries)], query_xy[len(queries):])
    styles = (("Cat bank", "#0072B2", "o", 10, 0.25),
              ("Dog bank", "#E69F00", "o", 10, 0.25),
              ("Cat validation", "#0072B2", "^", 55, 1),
              ("Mapped to dog", "#D55E00", "X", 65, 1))
    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for xy, (label, color, marker, size, alpha) in zip(groups, styles):
        axis.scatter(xy[:, 0], xy[:, 1], label=label, color=color,
                     marker=marker, s=size, alpha=alpha)
    for index, (source, target) in enumerate(zip(groups[2], groups[3]), 1):
        axis.plot([source[0], target[0]], [source[1], target[1]],
                  color="0.5", alpha=0.3, linewidth=0.7, zorder=0)
        axis.annotate(str(index), target, xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    return output_path
