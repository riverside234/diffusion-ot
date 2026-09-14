"""Summarize fixed InfoOT validation probes and sampled training diagnostics.

Uses only the standard library unless --plot is requested (matplotlib).
No checkpoints, datasets, or GPU are required; input logs are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def read_log(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty log: {path}")
    # Resumed runs can repeat the checkpoint-step probe. Report this and use
    # the last observation at that step; repeated rows are not replications.
    by_step = {int(row["step"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)], {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows), "duplicate_steps": len(rows) - len(by_step),
    }


def validation_row(row: dict) -> dict:
    result = {"step": row["step"], "conditional_kl": row.get("conditional_structure_loss"),
              "support_loss": row.get("projection_support_loss")}
    for domain in ("cat", "dog"):
        current, baseline = row.get(f"{domain}_raw_reconstruction"), row.get(f"{domain}_stage1a_reconstruction")
        result[f"{domain}_rec"] = current
        result[f"{domain}_rec_drift_pct"] = 100 * (current / baseline - 1) if baseline and current is not None else None
    for direction in ("cat_to_dog", "dog_to_cat"):
        metrics = row.get("conditional_structure", {}).get(direction, {})
        for field in ("kl", "expected_structure_cost", "projected_to_target_variance_ratio",
                      "projected_to_target_norm_ratio", "effective_targets"):
            result[f"{direction}.{field}"] = metrics.get(field)
        uniform, teacher, current = (metrics.get(k) for k in
                                     ("uniform_structure_cost", "teacher_structure_cost", "expected_structure_cost"))
        gap = uniform - teacher if uniform is not None and teacher is not None else 0
        result[f"{direction}.teacher_gain_pct"] = 100 * (uniform-current)/gap if gap > 1e-12 and current is not None else None
    return result


def summarize(folder: Path, warmup_steps: int) -> dict:
    training, training_source = read_log(folder / "train.jsonl")
    validation, validation_source = read_log(folder / "validation.jsonl")
    post_warmup = [row for row in training if row["step"] >= warmup_steps]
    guards = {}
    for domain in ("cat", "dog"):
        rows = [row["gradient_guard"][domain] for row in post_warmup if domain in row.get("gradient_guard", {})]
        if rows:
            guards[domain] = {"logged_observations": len(rows),
                              "mean_auxiliary_scale": mean(r["auxiliary_scale"] for r in rows),
                              "mean_cosine_before": mean(r["cosine_before"] for r in rows),
                              "conflict_fraction": mean(r["conflict_projected"] for r in rows),
                              "scaled_fraction": mean(r["auxiliary_scale"] < .99999 for r in rows),
                              "mean_ratio_after": mean(r["auxiliary_ratio_after"] for r in rows)}
    probes = [row.get("projection_probe", {}) for row in validation]
    ids = [probe["sample_ids"] for probe in probes if "sample_ids" in probe]
    return {
        "sources": {"training": training_source, "validation": validation_source},
        "warmup_steps": warmup_steps,
        "train_steps": [training[0]["step"], training[-1]["step"]],
        "validation_uses_same_ids": all(item == ids[0] for item in ids) if ids else None,
        "all_logged_training_solves_converged": all(
            r.get("infoot_sinkhorn_converged") and r.get("infoot_outer_converged") for r in training
        ),
        "post_warmup_gradient_guard": guards,
        "validation": [validation_row(row) for row in validation],
    }


def table(summary: dict) -> str:
    columns = ["step", "cat_rec_drift_pct", "dog_rec_drift_pct", "conditional_kl", "support_loss",
               "cat_to_dog.expected_structure_cost", "dog_to_cat.expected_structure_cost",
               "cat_to_dog.projected_to_target_variance_ratio", "dog_to_cat.projected_to_target_variance_ratio"]
    labels = ["Step", "Cat rec drift %", "Dog rec drift %", "KL", "Support",
              "C2D cost", "D2C cost", "C2D variance ratio", "D2C variance ratio"]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---:"] * len(labels)) + " |"]
    for row in summary["validation"]:
        values = [str(row[key]) if key == "step" else "n/a" if row[key] is None else f"{row[key]:.5f}" for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def plot(summary: dict, folder: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["validation"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = ("#2563eb", "#db6a1f")
    panels = [
        ("Fixed reconstruction drift", "% from Stage 1A", [("cat_rec_drift_pct", "Cat"), ("dog_rec_drift_pct", "Dog")]),
        ("Conditional structure KL", "KL (lower is better)", [("conditional_kl", "Bidirectional")]),
        ("Fraction of teacher cost gain", "% of uniform-to-teacher improvement",
         [("cat_to_dog.teacher_gain_pct", "Cat to Dog"), ("dog_to_cat.teacher_gain_pct", "Dog to Cat")]),
        ("Projected / target centered variance", "Variance ratio",
         [("cat_to_dog.projected_to_target_variance_ratio", "Cat to Dog"), ("dog_to_cat.projected_to_target_variance_ratio", "Dog to Cat")]),
        ("Projection-support divergence", "Divergence (lower is better)", [("support_loss", "Bidirectional")]),
    ]
    for ax, (title, ylabel, series) in zip(axes.flat, panels):
        for color, (key, label) in zip(colors, series):
            observed = [(r["step"], r[key]) for r in rows if r[key] is not None]
            if observed:
                ax.plot(*zip(*observed), marker="o", markersize=3, color=color, label=label)
        ax.set(title=title, ylabel=ylabel)
        if ax.lines:
            ax.legend(frameon=False, fontsize=9)
    training, _ = read_log(folder / "train.jsonl")
    ax = axes.flat[-1]
    for domain, color in zip(("cat", "dog"), colors):
        observed = [(r["step"], r["gradient_guard"][domain]["auxiliary_scale"]) for r in training
                    if domain in r.get("gradient_guard", {})]
        if observed:
            ax.plot(*zip(*observed), color=color, label=domain.title(), alpha=.85)
    ax.set(title="Auxiliary gradient retained by guard", ylabel="Scale after conflict projection", ylim=(0, 1.05))
    if ax.lines:
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.axvline(summary["warmup_steps"], color="#9ca3af", linestyle="--", linewidth=1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Training update")
    figure.suptitle(f"InfoOT co-training: fixed validation probes through {rows[-1]['step']:,} updates\n"
                     "Dashed line: end of alignment warmup. Teacher metrics do not measure decoded image quality.", fontsize=13)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot and args.output_dir is None:
        parser.error("--plot requires --output-dir")
    summary = summarize(args.log_dir, args.warmup_steps)
    print(table(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "validation"}, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "validation.md").write_text(table(summary), encoding="utf-8")
        if args.plot:
            plot(summary, args.log_dir, args.output_dir / "trends.png")


if __name__ == "__main__":
    main()
