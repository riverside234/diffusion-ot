from pathlib import Path
import logging
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator
from diffusion_ot.models.generator_adaptation import (
    configure_generator_adaptation,
)
from infoot_helper.cotraining_checkpoint import resume_latest
from infoot_helper.distributed.checkpoint import save_checkpoint
from infoot_helper.distributed.data import make_loader, cycle_batches
from infoot_helper.distributed.model import CoTrainingStep
from infoot_helper.distributed.runtime import (
    setup_distributed, close_distributed, setup_logging,
    wrap_distributed, sync_batchnorm_buffers, mean_metrics,
)

from diffusion_ot.data.afhq import load_afhq_dataset
from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.data.manifests import read_jsonl
from infoot_helper.augmentation import augment_and_encode

def main():
    rank, world_size, device = setup_distributed()
    try:
        train(rank, world_size, device)
    finally:
        close_distributed()


def train(rank, world_size, device):
    torch.manual_seed(42)

    settings = {
        "batch_size": 1024,
        "encode_batch_size": 128,
        "flow_batch_size": 16,
        "query_count": 32,
        "steps": 2000,
        "fit_h": 0.4,
        "projection_h": 0.2,
        "mi_weight": 1.59,
        "reg": 0.74,
        "infoOT_loss_weight": 0.10,
        "fit_iterations": 100,
        "sampling_steps": 20,
        "flow_weight": 1.0,
        "contrastive_weight": 0.05,
        "covariance_weight": 0.3,
        "batchnorm": {"affine": False, "momentum": 0.1, "eps": 1e-5},
        "batchnorm_lr": 1e-5,
    }
    if not 1 <= settings["flow_batch_size"] <= settings["batch_size"]:
        raise ValueError("flow_batch_size must be between 1 and batch_size.")

    output_dir = ROOT / "outputs/infoot_cotraining"
    setup_logging(output_dir, rank)
    logging.info("Training settings: %s", settings)
    logging.info("GPUs/processes=%s global_batch_size_per_domain=%s (batch_size is per GPU)",
                 world_size, world_size * settings["batch_size"])

    domains = {}
    batch_norms = torch.nn.ModuleDict()
    loaders = {}
    parameter_groups = []

    data_config_path = ROOT / "configs/data/afhq_huggan.yaml"
    image_dataset = load_afhq_dataset(data_config_path)

    for name in ("cat", "dog"):
        context = load_stage1a_evaluator(
            ROOT / (
                f"configs/stage1a_pdae/"
                f"{name}_sit_b2_lora_residual_cosmap.yaml"
            ),
            ROOT / "configs/stage1a_eval/residual_sit_b2_256.yaml",
            device=device,
            weights="raw",
        )
        domains[name] = context
        batch_norms[name] = torch.nn.BatchNorm1d(
            context.training_config["encoder"]["z_dim"], **settings["batchnorm"],
        ).to(context.device)
        if batch_norms[name].affine:
            parameter_groups.append({
                "params": batch_norms[name].parameters(), "lr": settings["batchnorm_lr"],
            })

        context.vae.eval().requires_grad_(False)
        generator_view = configure_generator_adaptation(context.branch)

        parameter_groups.extend([
            {
                "params": context.branch.encoder.parameters(), #encoder
                "lr": 1e-5,
            },
            {
                "params": generator_view.parameters(),  #generator
                "lr": 5e-6,
            },
        ])

        records = read_jsonl(
            ROOT / "data/manifests" / f"{name}_train.jsonl"
        )

        loaders[name] = make_loader(records, settings["batch_size"], rank, world_size)

    optimizer = torch.optim.Adam(parameter_groups)
    torch.manual_seed(42 + rank)
    start_step = resume_latest(output_dir, domains, batch_norms, optimizer, device=device)
    if start_step >= settings["steps"]:
        logging.info(f"Already at step {start_step}; increase settings['steps'] to continue.")
        return
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    batches = {name: cycle_batches(loader, start_step) for name, loader in loaders.items()}
    training_step = wrap_distributed(CoTrainingStep(domains, batch_norms, settings), device)

    for step in range(start_step + 1, settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)

        latents = {}

        for name, iterator in batches.items():
            images = load_ground_truth_images(
                data_config_path,
                next(iterator),
                dataset=image_dataset,
            )
            latents[name] = torch.cat([
                augment_and_encode(domains[name], chunk)
                for chunk in images.split(settings["encode_batch_size"])
            ])

        report = step == start_step + 1 or step % 100 == 0
        loss, metrics = training_step(latents, report=report)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        sync_batchnorm_buffers(batch_norms)

        if report:
            metrics = mean_metrics(metrics, device)
            logging.info("step=%s %s", step, " ".join(f"{key}={value:.6g}" for key, value in metrics.items()))

        if step % 200 == 0 or step == settings["steps"]:
            save_checkpoint(output_dir, step, domains, batch_norms, optimizer, settings, device)


if __name__ == "__main__":
    main()
