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


def matching_regularization_fields(row: dict) -> dict:
    """Preserve protection diagnostics; absence in older logs stays explicit."""
    protection = row.get("matching_regularization", {})
    result = {"matching_regularization_loss": row.get("matching_regularization_loss")}
    for key in ("variance_loss", "covariance_loss", "weighted_variance_loss", "weighted_covariance_loss", "std_target"):
        result[f"matching_regularization.{key}"] = protection.get(key)
    for domain in ("cat", "dog"):
        for key in ("scaled_std_mean", "scaled_std_min", "fraction_below_std_target", "covariance_loss"):
            result[f"matching_regularization.{domain}.{key}"] = protection.get(domain, {}).get(key)
    return result


def validation_row(row: dict) -> dict:
    probe, decoded = row.get("projection_probe", {}), row.get("decoded_translation", {})
    result = {"step": row["step"], "conditional_kl": row.get("conditional_structure_loss"),
              "support_loss": row.get("projection_support_loss"),
              "decoded_structure": decoded.get("structure_loss"),
              "outer_converged": probe.get("outer_converged"),
              "sinkhorn_converged": probe.get("sinkhorn_converged"),
              "solver_iterations": probe.get("iterations"),
              "solver_plan_delta_l1": probe.get("plan_delta_l1"),
              "solver_row_residual": probe.get("row_residual"),
              "solver_column_residual": probe.get("column_residual"),
              "decoded_outer_converged": decoded.get("solver", {}).get("outer_converged")}
    for domain in ("cat", "dog"):
        current, baseline = row.get(f"{domain}_raw_reconstruction"), row.get(f"{domain}_stage1a_reconstruction")
        result[f"{domain}_rec"] = current
        result[f"{domain}_rec_drift_pct"] = 100 * (current / baseline - 1) if baseline and current is not None else None
        current_null, baseline_null = row.get(f"{domain}_null_reconstruction"), row.get(f"{domain}_stage1a_null_reconstruction")
        result[f"{domain}_null_rec"] = current_null
        result[f"{domain}_null_rec_drift_pct"] = 100 * (current_null / baseline_null - 1) if baseline_null and current_null is not None else None
        result[f"{domain}_matching_variance"] = probe.get("matching_feature_variance", {}).get(domain)
        for key in ("stage1a_encoder_current_generator_reconstruction", "current_encoder_stage1a_generator_reconstruction"):
            result[f"{domain}_{key}"] = row.get(f"{domain}_{key}")
    for direction in ("cat_to_dog", "dog_to_cat"):
        metrics = row.get("conditional_structure", {}).get(direction, {})
        for field in ("kl", "expected_structure_cost", "projected_to_target_variance_ratio",
                      "projected_to_target_norm_ratio", "effective_targets", "teacher_effective_targets",
                      "teacher_projected_to_target_variance_ratio", "teacher_conditional_variance_fraction"):
            result[f"{direction}.{field}"] = metrics.get(field)
        result[f"{direction}.decoded_structure"] = decoded.get(direction, {}).get("structure_loss")
        uniform, teacher, current = (metrics.get(k) for k in
                                     ("uniform_structure_cost", "teacher_structure_cost", "expected_structure_cost"))
        gap = uniform - teacher if uniform is not None and teacher is not None else 0
        result[f"{direction}.teacher_gain_pct"] = 100 * (uniform-current)/gap if gap > 1e-12 and current is not None else None
    result.update(matching_regularization_fields(row))
    return result


def training_row(row: dict) -> dict:
    """Keep rotating-batch observations separate from fixed validation probes."""
    window = row.get("window_mean", {})
    image = row.get("decoded_translation", {})
    result = {"step": row["step"], "window_updates": row.get("window_updates"),
              "window_loss": window.get("loss"),
              "window_conditional_kl": window.get("conditional_structure_loss"),
              "window_mi": window.get("infoot_mutual_information"),
              "window_neighborhood_loss": window.get("semantic_neighborhood_loss"),
              "window_decoded_loss": window.get("decoded_translation_loss"),
              "window_matching_regularization_loss": window.get("matching_regularization_loss"),
              "window_matching_variance_loss": window.get("matching_variance_loss"),
              "window_matching_covariance_loss": window.get("matching_covariance_loss"),
              "decoded_structure": image.get("structure_loss"),
              "decoded_adversarial": image.get("adversarial_loss"),
              "discriminator_loss": image.get("discriminator_loss"),
              "discriminator_gradient_norm": image.get("discriminator_gradient_norm"),
              "decoded_ramp": image.get("ramp")}
    cat, dog = (window.get(f"{d}_reconstruction_loss") for d in ("cat", "dog"))
    result["window_reconstruction"] = cat + dog if cat is not None and dog is not None else None
    for key in ("conditional_structure_to_reconstruction_gradient_ratio",
                "alignment_to_reconstruction_gradient_ratio", "total_grad_norm_pre_clip",
                "matching_regularization_to_reconstruction_encoder_gradient_ratio",
                "weighted_matching_regularization_encoder_gradient_norm",
                "weighted_matching_regularization_matching_head_gradient_norm",
                "generator_gradient_norm_pre_clip", "seconds_per_step", "training_seconds_total"):
        result[key] = row.get(key)
    for domain in ("cat", "dog"):
        result[f"{domain}_matching_variance"] = row.get("matching_feature_variance", {}).get(domain)
        result[f"{domain}_distance_scale"] = row.get(f"{domain}_distance_scale")
        result[f"{domain}_anchor_loss"] = row.get(f"{domain}_anchor_loss")
    for direction in ("cat_to_dog", "dog_to_cat"):
        for key, value in row.get("conditional_structure", {}).get(direction, {}).items():
            result[f"{direction}.{key}"] = value
        result[f"{direction}.decoded_structure"] = image.get(direction, {}).get("structure_loss")
    result.update(matching_regularization_fields(row))
    return result


def summarize(folder: Path | None, warmup_steps: int, *, train_log: Path | None = None,
              validation_log: Path | None = None) -> dict:
    if train_log is None and folder is None:
        raise ValueError("Supply a log directory or training log.")
    training, training_source = read_log(train_log if train_log is not None else folder / "train.jsonl")
    # A console paste often contains training alone. Absence is not zero drift
    # or evidence that the trainer failed to run its separate validation probes.
    validation_path = validation_log if validation_log is not None else (folder / "validation.jsonl" if folder else None)
    if validation_log is not None or (validation_path is not None and validation_path.exists()):
        validation, validation_source = read_log(validation_path)
    else:
        validation, validation_source = [], None
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
    decoded_ids = [(r["decoded_translation"]["reference_ids"], r["decoded_translation"]["query_ids"])
                   for r in validation if all(k in r.get("decoded_translation", {}) for k in ("reference_ids", "query_ids"))]
    return {
        "sources": {"training": training_source, "validation": validation_source},
        "validation_status": "available" if validation else "not_supplied",
        "training_observation_note": "Rotating minibatches; window means cover only each reported window, not all intervening updates.",
        "warmup_steps": warmup_steps,
        "train_steps": [training[0]["step"], training[-1]["step"]],
        "validation_uses_same_ids": all(item == ids[0] for item in ids) if ids and len(ids) == len(validation) else None,
        "validation_uses_same_decoded_ids": all(item == decoded_ids[0] for item in decoded_ids) if decoded_ids and len(decoded_ids) == len(validation) else None,
        "validation_steps_failed_outer_convergence": [r["step"] for r in validation if r.get("projection_probe", {}).get("outer_converged") is False],
        "validation_steps_missing_outer_convergence": [r["step"] for r in validation if r.get("projection_probe", {}).get("outer_converged") is None],
        "all_logged_training_solves_converged": all(
            r.get("infoot_sinkhorn_converged") and r.get("infoot_outer_converged") for r in training
        ),
        "post_warmup_gradient_guard": guards,
        "training": [training_row(row) for row in training],
        "validation": [validation_row(row) for row in validation],
    }


def table(summary: dict) -> str:
    if not summary["validation"]:
        return "No fixed validation log supplied; reconstruction drift and held-out translation quality cannot be inferred.\n"
    columns = ["step", "cat_rec_drift_pct", "dog_rec_drift_pct", "conditional_kl", "support_loss",
               "cat_to_dog.expected_structure_cost", "dog_to_cat.expected_structure_cost",
               "cat_to_dog.projected_to_target_variance_ratio", "dog_to_cat.projected_to_target_variance_ratio"]
    labels = ["Step", "Cat rec drift %", "Dog rec drift %", "KL", "Support",
              "C2D cost", "D2C cost", "C2D variance ratio", "D2C variance ratio"]
    if any(row["decoded_structure"] is not None for row in summary["validation"]):
        columns = ["step", "cat_rec_drift_pct", "dog_rec_drift_pct", "conditional_kl",
                   "cat_to_dog.decoded_structure", "dog_to_cat.decoded_structure",
                   "cat_to_dog.teacher_gain_pct", "dog_to_cat.teacher_gain_pct", "outer_converged"]
        labels = ["Step", "Cat rec drift %", "Dog rec drift %", "KL", "C2D decoded structure",
                  "D2C decoded structure", "C2D teacher gain %", "D2C teacher gain %", "Outer converged"]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---:"] * len(labels)) + " |"]
    for row in summary["validation"]:
        values = ["n/a" if row[key] is None else str(row[key]) if key in {"step", "outer_converged"} else f"{row[key]:.5f}" for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def training_table(summary: dict) -> str:
    columns = ["step", "window_reconstruction", "window_conditional_kl", "window_mi",
               "cat_matching_variance", "dog_matching_variance", "decoded_structure",
               "conditional_structure_to_reconstruction_gradient_ratio"]
    labels = ["Step", "Window reconstruction", "Window KL", "Window MI", "Cat matching variance",
              "Dog matching variance", "Decoded structure (batch)", "KL / reconstruction E gradient"]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---:"] * len(labels)) + " |"]
    for row in summary["training"]:
        values = [str(row[key]) if key == "step" else "n/a" if row[key] is None else f"{row[key]:.5f}" for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def plot_training(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["training"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("Same-domain flow loss", "Window mean", [("window_reconstruction", "Cat + Dog")]),
        ("Conditional structure KL", "Window mean", [("window_conditional_kl", "Bidirectional")]),
        ("Matching-feature spread", "Population covariance trace",
         [("cat_matching_variance", "Cat"), ("dog_matching_variance", "Dog")]),
        ("Decoded DINO structure", "Rotating batch (4 images / direction)",
         [("cat_to_dog.decoded_structure", "Cat to Dog"), ("dog_to_cat.decoded_structure", "Dog to Cat")]),
        ("Encoder gradient magnitudes", "Weighted objective / reconstruction",
         [("conditional_structure_to_reconstruction_gradient_ratio", "Conditional KL"),
          ("alignment_to_reconstruction_gradient_ratio", "InfoOT")]),
        ("Feature discriminator", "Hinge loss (not a quality score)", [("discriminator_loss", "Both domains")]),
    ]
    for ax, (title, ylabel, series) in zip(axes.flat, panels):
        for color, (key, label) in zip(("#2563eb", "#d97706"), series):
            observed = [(r["step"], r.get(key)) for r in rows if r.get(key) is not None]
            if observed:
                ax.plot(*zip(*observed), color=color, marker="o", markersize=3, label=label)
        ax.axvline(summary["warmup_steps"], color="#9ca3af", linestyle="--", linewidth=1)
        full_ramp = next((r["step"] for r in rows if r.get("decoded_ramp") == 1), None)
        if full_ramp is not None:
            ax.axvline(full_ramp, color="#9ca3af", linestyle=":", linewidth=1)
        ax.set(title=title, ylabel=ylabel, xlabel="Training update")
        ax.grid(alpha=.18)
        if ax.lines and any(line.get_label()[0] != "_" for line in ax.lines):
            ax.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Experiment D: {len(rows)} training snapshots through step {rows[-1]['step']:,}\n"
                   "Rotating batches, not fixed validation. Dashed: alignment warmup; dotted: first logged full image ramp.", fontsize=12)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot(summary: dict, folder: Path, output: Path) -> None:
    if not summary["validation"]:
        plot_training(summary, output)
        return
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
    decoded_run = any(r["decoded_structure"] is not None for r in rows)
    if decoded_run:
        panels[3:] = [
            ("Decoded DINO structure", "4 fixed images / direction; training feature prior",
             [("cat_to_dog.decoded_structure", "Cat to Dog"), ("dog_to_cat.decoded_structure", "Dog to Cat")]),
            ("Matching-feature spread", "Fixed reference covariance trace",
             [("cat_matching_variance", "Cat"), ("dog_matching_variance", "Dog")]),
            ("Null reconstruction drift", "% from Stage 1A",
             [("cat_null_rec_drift_pct", "Cat"), ("dog_null_rec_drift_pct", "Dog")]),
        ]
    for ax, (title, ylabel, series) in zip(axes.flat, panels):
        for color, (key, label) in zip(colors, series):
            observed = [(r["step"], r[key]) for r in rows if r[key] is not None]
            if observed:
                ax.plot(*zip(*observed), marker="o", markersize=3, color=color, label=label)
        ax.set(title=title, ylabel=ylabel)
        if ax.lines:
            ax.legend(frameon=False, fontsize=9)
    if not decoded_run:
        training, _ = read_log(Path(summary["sources"]["training"]["path"]))
        ax = axes.flat[-1]
        for domain, color in zip(("cat", "dog"), colors):
            observed = [(r["step"], r["gradient_guard"][domain]["auxiliary_scale"]) for r in training
                        if domain in r.get("gradient_guard", {})]
            if observed:
                ax.plot(*zip(*observed), color=color, label=domain.title(), alpha=.85)
        ax.set(title="Auxiliary gradient retained by guard", ylabel="Scale after conflict projection", ylim=(0, 1.05))
        if ax.lines:
            ax.legend(frameon=False, fontsize=9)
    else:
        # The failed solve directly affects KL and teacher gain. Decoded metrics
        # refit a separate plan whose convergence older logs do not report.
        for ax, key in ((axes.flat[1], "conditional_kl"), (axes.flat[2], "cat_to_dog.teacher_gain_pct")):
            failed = [(r["step"], r[key]) for r in rows if r["outer_converged"] is False and r[key] is not None]
            if failed:
                ax.scatter(*zip(*failed), marker="x", color="#dc2626", s=65, zorder=5, label="Outer solve unfinished")
                ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.axvline(summary["warmup_steps"], color="#9ca3af", linestyle="--", linewidth=1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Training update")
    figure.suptitle(f"InfoOT co-training: fixed validation probes through {rows[-1]['step']:,} updates\n"
                     "Dashed: alignment warmup. DINO scores measure teacher agreement; they are not independent quality validation.", fontsize=12)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-dir", type=Path)
    source.add_argument("--train-log", type=Path, help="A training-only JSONL file or console JSONL paste.")
    parser.add_argument("--validation-log", type=Path, help="Optional separate fixed-probe JSONL file.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.plot and args.output_dir is None:
        parser.error("--plot requires --output-dir")
    summary = summarize(args.log_dir, args.warmup_steps, train_log=args.train_log, validation_log=args.validation_log)
    print(table(summary))
    print(training_table(summary))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"validation", "training"}}, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "validation.md").write_text(table(summary), encoding="utf-8")
        (args.output_dir / "training.md").write_text(training_table(summary), encoding="utf-8")
        if args.plot:
            plot(summary, args.log_dir, args.output_dir / "trends.png")
            if summary["validation"]:
                plot_training(summary, args.output_dir / "training_trends.png")


if __name__ == "__main__":
    main()
