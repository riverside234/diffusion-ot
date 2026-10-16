"""Train a CNN PDAE branch with the existing Stage 1A recipes."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from diffusion_ot.config_defaults import stage1a_training_config
from diffusion_ot.integrations.hf_snapshot import resolve_project_local_path
from diffusion_ot.training.learning_rate import apply_step_learning_rates, resolve_lr_schedule
from pdae_cnn.helpers.config import load_config


def train(config, *, resume_from=None):
    """Run optimizer updates and return the latest checkpoint path."""
    from pdae_cnn.data import build_loaders, cycle_batches
    from pdae_cnn.helpers.checkpoint import load_checkpoint, save_checkpoint
    from pdae_cnn.initialize import build_training_state
    from pdae_cnn.train_step import train_step
    from pdae_cnn.eval.validation import validate

    options = config.get("train") or {}
    evaluation = config.get("evaluation") or {}
    max_steps = int(options.get("max_steps", 1000))
    log_every, save_every = int(options.get("log_every", 40)), int(options.get("save_every", 1000))
    eval_every = int(evaluation.get("every_steps", 1000))
    if max_steps < 0 or min(log_every, save_every) <= 0:
        raise ValueError("max_steps must be non-negative; logging and saving intervals must be positive.")
    if evaluation.get("enabled", True) and eval_every <= 0:
        raise ValueError("evaluation.every_steps must be positive.")
    schedule = resolve_lr_schedule(options.get("lr_schedule"))
    latest = Path(config["output_dir"]) / "checkpoints/latest.pt"
    requested = resume_from if resume_from is not None else options.get("resume_from")
    resume_path = None
    if requested:
        resume_path = latest if str(requested) == "latest" else resolve_project_local_path(
            requested, config["project_root"], field_name="resume_from",
        )
    if resume_path is None and latest.exists():
        raise FileExistsError("Output already has a checkpoint; use --resume or a new --output-dir.")

    train_loader, val_loader = build_loaders(config)
    state = build_training_state(config)
    initial_step = load_checkpoint(resume_path, state, config, loader=train_loader) if resume_path else 0
    if not 0 <= initial_step <= max_steps:
        raise ValueError("Checkpoint step must be between zero and train.max_steps.")
    base_lrs = [group["lr"] for group in state.optimizer.param_groups]
    batches = cycle_batches(train_loader)

    def save(step):
        save_checkpoint(latest, state, config, step=step, loader=train_loader)
        if options.get("keep_step_checkpoints", True):
            save_checkpoint(latest.with_name(f"step_{step:06d}.pt"), state, config,
                            step=step, loader=train_loader)

    def evaluate(step):
        print(f"validation step={step}: {validate(state, val_loader, config, step=step)}", flush=True)

    if resume_path is None:
        save(initial_step)
    if val_loader is not None and evaluation.get("evaluate_at_start", True):
        evaluate(initial_step)
    for step in range(initial_step + 1, max_steps + 1):
        apply_step_learning_rates(state.optimizer, base_lrs, schedule, step)
        metrics = train_step(state, batches, config, step=step)
        if step == initial_step + 1 or step % log_every == 0 or step == max_steps:
            print(f"step={step}/{max_steps} loss={metrics['loss'].item():.6f}", flush=True)
        if step % save_every == 0 or step == max_steps:
            save(step)
        if val_loader is not None and (step % eval_every == 0 or step == max_steps):
            evaluate(step)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    recipe = parser.add_mutually_exclusive_group(required=True)
    recipe.add_argument("--config", help="Path to a Stage 1A training YAML.")
    recipe.add_argument("--domain", choices=("cat", "dog"), help="Use the domain's default CNN recipe.")
    parser.add_argument("--device", help="Torch device, such as cuda:0 or cpu.")
    parser.add_argument("--project-root", help="Override the training recipe's project root.")
    parser.add_argument("--output-dir", help="Override the checkpoint output directory.")
    parser.add_argument("--max-steps", type=int, help="Override the total optimizer-update limit.")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume latest, or a checkpoint path.")
    args = parser.parse_args()
    path = Path(args.config or stage1a_training_config(args.domain)).expanduser()
    config = load_config(
        path if path.is_absolute() else ROOT / path,
        project_root=args.project_root, device=args.device, output_dir=args.output_dir,
    )
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    print(f"checkpoint: {train(config, resume_from=args.resume)}", flush=True)


if __name__ == "__main__":
    main()
