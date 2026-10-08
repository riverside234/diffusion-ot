from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.config_defaults import STAGE1A_EVAL, stage1a_training_config


def repo_path(path: str) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else ROOT / raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 1A PDAE fixed-noise smoke gate and optional "
            "inferred-noise round-trip gate for one domain. The training "
            "config selects the matching semantic encoder and AdaLN/LoRA architecture."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=["cat", "dog"], help="Use the default residual-cosmap training recipe for this domain.")
    selection.add_argument(
        "--train-config",
        help=(
            "The exact Cat or Dog Stage 1A training config used by the checkpoint, "
            "including its plain or residual encoder and attention-LoRA settings."
        ),
    )
    parser.add_argument(
        "--eval-config",
        default=STAGE1A_EVAL,
        help="Shared Stage 1A evaluation config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--weights",
        choices=["ema", "raw", "both"],
        default=None,
        help="Checkpoint weights override; 'both' saves matched raw/EMA grids and a comparison report.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override; defaults to the training run's latest.pt.",
    )
    parser.add_argument(
        "--roundtrip", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable inferred-noise round-trip evaluation; follows YAML when omitted (PDAE v2 defaults off).",
    )
    parser.add_argument("--solver", choices=["euler", "heun"], default=None, help="Override the velocity ODE solver.")
    parser.add_argument("--num-steps", type=int, default=None, help="Smoke integration steps (Heun uses two field evaluations per step).")
    parser.add_argument("--noise-seed", type=int, default=None, help="Change starting noise while preserving the selected image IDs.")
    parser.add_argument("--output-subdir", default=None, help="Output subdirectory under the training run; sampling overrides otherwise create a distinct subdirectory.")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test, run_stage1a_weight_comparison

    args = parse_args()
    sampling_overrides = dict(solver=args.solver, num_steps=args.num_steps,
                              noise_seed=args.noise_seed, output_subdir=args.output_subdir)
    if args.weights == "both":
        comparison = run_stage1a_weight_comparison(
            repo_path(args.train_config or stage1a_training_config(args.domain)),
            repo_path(args.eval_config), device=args.device, checkpoint_path=args.checkpoint,
            roundtrip=args.roundtrip,
            **sampling_overrides,
        )
        print(json.dumps({"comparison_path": comparison["comparison_path"],
                          "grids": {name: report["grid_path"] for name, report in comparison["reports"].items()},
                          "ema_minus_raw": comparison["ema_minus_raw"]}, indent=2))
        return 0
    report = run_stage1a_smoke_test(
        repo_path(args.train_config or stage1a_training_config(args.domain)),
        repo_path(args.eval_config),
        device=args.device,
        weights=args.weights,
        checkpoint_path=args.checkpoint,
        roundtrip=args.roundtrip,
        **sampling_overrides,
    )
    print("stage1a_smoke_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    if report.extra_reports:
        print("stage1a_extra_reports:")
        for key, value in sorted(report.extra_reports.items()):
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
