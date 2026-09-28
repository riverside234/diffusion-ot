from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Cat/Dog InfoOT alignment, optionally with experiment D generator adaptation and decoded image losses."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device-cat", default=None)
    parser.add_argument("--device-dog", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use train.smoke_steps from the alignment config.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a Stage 1B checkpoint, or latest.pt when no path is supplied.",
    )
    parser.add_argument(
        "--quick-eval",
        nargs="?",
        const="__configured__",
        default=None,
        metavar="CONFIG",
        help=(
            "Evaluate the initial baseline and final Stage 1B checkpoint. Spatial-cost "
            "recipes use their retained calibrated step-0 checkpoint after training; "
            "other recipes evaluate Stage 1A before training. With no value, use quick_evaluation.config."
        ),
    )
    parser.add_argument("--eval-weights", choices=["ema", "raw"], default=None)
    parser.add_argument("--max-reference", type=int, default=None)
    parser.add_argument("--max-projection", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    from diffusion_ot.evaluation.stage1b_eval import run_stage1b_evaluation
    from diffusion_ot.training.train_joint_infoot import train_joint_infoot
    from diffusion_ot.losses.spatial_correlative import spatial_correlative_options

    args = parse_args()
    config_path = repo_path(args.config)
    config = load_yaml_config(config_path)
    spatial_enabled = spatial_correlative_options(config) is not None
    train_config = config.get("train") or {}
    if args.smoke and args.max_steps is not None:
        raise ValueError("Use either --smoke or --max-steps, not both.")
    max_steps = (
        int(train_config.get("smoke_steps", 500)) if args.smoke else args.max_steps
    )

    quick_eval_path = None
    eval_weights = args.eval_weights
    if args.quick_eval is not None:
        quick_config = config.get("quick_evaluation") or {}
        if args.quick_eval == "__configured__":
            if not quick_config.get("config"):
                raise ValueError("The alignment config has no quick_evaluation.config path.")
            quick_eval_path = repo_path(str(quick_config["config"]))
        else:
            quick_eval_path = repo_path(args.quick_eval)
        eval_weights = eval_weights or str(quick_config.get("weights", "ema"))

    baseline_report = None
    if quick_eval_path is not None and not args.dry_run and not spatial_enabled:
        baseline_report = run_stage1b_evaluation(
            config_path,
            quick_eval_path,
            device_cat=args.device_cat,
            device_dog=args.device_dog,
            max_reference=args.max_reference,
            max_projection=args.max_projection,
            max_query=args.max_query,
        )
        print(f"stage1a_baseline_report: {baseline_report.output_dir}")

    report = train_joint_infoot(
        config_path,
        device_cat=args.device_cat,
        device_dog=args.device_dog,
        max_steps=max_steps,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )
    print("stage1b_train_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    if quick_eval_path is not None and not args.dry_run:
        if report.checkpoint_path is None:
            raise RuntimeError("Training completed without a checkpoint to evaluate.")
        if spatial_enabled:
            # Calibration is created by training and frozen in step_000000.pt.
            # Evaluate that retained initial state before the final checkpoint.
            baseline_report = run_stage1b_evaluation(
                config_path, quick_eval_path,
                checkpoint_path=Path(report.checkpoint_path).with_name("step_000000.pt"),
                weights=str(eval_weights), device_cat=args.device_cat, device_dog=args.device_dog,
                max_reference=args.max_reference, max_projection=args.max_projection, max_query=args.max_query)
            print(f"stage1b_initial_calibrated_report: {baseline_report.output_dir}")
        joint_report = run_stage1b_evaluation(
            config_path,
            quick_eval_path,
            checkpoint_path=report.checkpoint_path,
            weights=str(eval_weights),
            device_cat=args.device_cat,
            device_dog=args.device_dog,
            max_reference=args.max_reference,
            max_projection=args.max_projection,
            max_query=args.max_query,
        )
        print(f"stage1b_quick_evaluation_report: {joint_report.output_dir}")
        print(f"  baseline_comparison: {joint_report.baseline_comparison['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
