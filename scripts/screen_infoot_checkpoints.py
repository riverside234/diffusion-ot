from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P1/E0 cached checkpoint/readout screen (no training or P1a).")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Reuse each checkpoint's models, banks and fitted InfoOT across readouts/bandwidths.")
    run.add_argument("--alignment-config", required=True)
    run.add_argument("--eval-config", required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints", nargs="+", help="Ordered paths; put the saved step-0 checkpoint first.")
    source.add_argument("--checkpoint-dir", help="Directory with step_NNNNNN.pt checkpoints.")
    run.add_argument("--checkpoint-steps", nargs="+", type=int, default=[0, 2500, 5000, 8000])
    run.add_argument("--bandwidths", nargs="+", type=float, default=[.15, .20, .25, .35])
    run.add_argument("--finalist", action="append", metavar="BANDWIDTH:READOUT",
                     help="Repeat for chosen pairs; defaults to 64 images and 20/40 steps. No automatic winner selection.")
    run.add_argument("--num-steps", nargs="+", type=int, default=None)
    run.add_argument("--samples", type=int, default=None, help="Default 16 for screening, 64 for finalists.")
    run.add_argument("--draws", type=int, default=3, help="Fixed categorical/noise draws; same draws across variants.")
    run.add_argument("--bank-seeds", nargs="+", type=int, default=None, help="Optional separate bank robustness screen; query/noise seeds stay fixed.")
    run.add_argument("--weights", choices=["ema", "raw"], default="ema")
    run.add_argument("--device-cat", default=None)
    run.add_argument("--device-dog", default=None)
    run.add_argument("--output-dir", required=True)
    summary = sub.add_parser("summarize", help="Compare copied evaluation JSON reports without checkpoints/dataset.")
    summary.add_argument("--results-dir", required=True)
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--reference-bandwidth", type=float, default=.25)
    return parser.parse_args(argv)


def main(argv=None):
    from diffusion_ot.evaluation.checkpoint_screen import run_checkpoint_screen, summarize_reports
    args = parse_args(argv)
    if args.command == "summarize":
        summarize_reports(repo_path(args.results_dir).rglob("evaluation_report.json"), repo_path(args.output_dir),
                          reference_bandwidth=args.reference_bandwidth)
    else:
        checkpoints = ([repo_path(x) for x in args.checkpoints] if args.checkpoints else
                       [repo_path(args.checkpoint_dir) / f"step_{step:06d}.pt" for step in args.checkpoint_steps])
        finalists = None
        if args.finalist:
            try:
                finalists = [(float(value.split(":")[0]), value.split(":")[1]) for value in args.finalist
                             if value.count(":") == 1]
                if len(finalists) != len(args.finalist):
                    raise ValueError()
            except (ValueError, IndexError):
                raise SystemExit("Finalists must have the form 0.25:conditional_mean.")
        run_checkpoint_screen(repo_path(args.alignment_config), repo_path(args.eval_config), checkpoints,
            repo_path(args.output_dir), weights=args.weights, device_cat=args.device_cat, device_dog=args.device_dog,
            bandwidths=args.bandwidths, finalists=finalists, steps=args.num_steps, samples=args.samples,
            draws=args.draws, bank_seeds=args.bank_seeds)
    print(f"P1 results: {repo_path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
