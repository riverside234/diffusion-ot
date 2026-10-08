from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator
from diffusion_ot.models.generator_adaptation import (
    configure_generator_adaptation,
)
from diffusion_ot.training.train_joint_infoot import _cycle

from infoot_helper.encoding import encode_batches
from infoot_helper.diagnostics import representation_log
from infoot_helper.batchnorm_matching import (
    add_matching_features, batchnorm_checkpoint, covariance_loss,
)
from infoot_helper.infoot_cotraining_helper import (
    fit_transport,
    alignment_loss,
    conditional_mapping,
    save_domain_checkpoints,
)
from infoot_helper.native_flow import native_flow_loss
from infoot_helper.translation_contrastive import (
    translation_contrastive_loss,
)

from diffusion_ot.data.afhq import load_afhq_dataset
from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.data.manifests import read_jsonl
from infoot_helper.augmentation import augment_and_encode

def main():
    torch.manual_seed(42)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Co-training requires two visible GPUs; set CUDA_VISIBLE_DEVICES to a GPU pair.")
    device, infoot_device = "cuda:0", "cuda:1"

    settings = {
        "batch_size": 1024,
        "encode_batch_size": 32,
        "flow_batch_size": 8,
        "query_count": 16,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = {}
    batch_norms = torch.nn.ModuleDict()
    batches = {}
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

        loader = DataLoader(
            records,
            batch_size=settings["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=list,
            num_workers=2,
        )

        batches[name] = _cycle(loader)

    optimizer = torch.optim.Adam(parameter_groups)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    for step in range(1, settings["steps"] + 1):
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

        encoded = encode_batches(
            domains, latents, query_count=settings["query_count"],
            encode_batch_size=settings["encode_batch_size"],
        )
        add_matching_features(encoded, batch_norms)

        flow_losses = {
            name: native_flow_loss(
                domains[name],
                encoded[name]["x0"][:settings["flow_batch_size"]],
                encoded[name]["v"][:settings["flow_batch_size"]],
            )
            for name in domains
        }
        loss_flow = torch.stack(list(flow_losses.values())).mean()

        cat_refs = encoded["cat"]["references"]["v"]
        dog_refs = encoded["dog"]["references"]["v"]
        matching = {
            name: batch["references"]["m"].to(infoot_device)
            for name, batch in encoded.items()
        }
        covariance_losses = {
            name: covariance_loss(batch["references"]["v"])
            for name, batch in encoded.items()
        }
        loss_covariance = torch.stack(list(covariance_losses.values())).mean()

        P = fit_transport(
            matching["cat"],
            matching["dog"],
            h=settings["fit_h"],
            mi_weight=settings["mi_weight"],
            reg=settings["reg"],
            iterations=settings["fit_iterations"],
        )

        loss_infoot = alignment_loss(
            matching["cat"],
            matching["dog"],
            P,
            h=settings["fit_h"],
            mi_weight=settings["mi_weight"],
            reg=settings["reg"],
        ).to(device)

        contrastive_losses = []

        for source, target, plan in (
            ("cat", "dog", P),
            ("dog", "cat", P.T),
        ):
            mapped_v = conditional_mapping(
                encoded[source]["queries"]["m"].to(infoot_device),
                matching[source],
                matching[target],
                plan,
                h=settings["projection_h"],
                target_v=encoded[target]["references"]["v"].to(infoot_device),
            ).to(device)

            loss, metrics = translation_contrastive_loss(
                domains[target],
                encoded[source],
                mapped_v,
                steps=settings["sampling_steps"],
            )

            contrastive_losses.append(loss)

        loss_contrastive = torch.stack(contrastive_losses).mean()

        loss = (
            settings["flow_weight"] * loss_flow
            + settings["infoOT_loss_weight"] * loss_infoot
            + settings["contrastive_weight"] * loss_contrastive
            + settings["covariance_weight"] * loss_covariance
        )


        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()

        if step == 1 or step % 100 == 0:
            geometry = representation_log({"cat": cat_refs, "dog": dog_refs})
            print(
                f"step={step} "
                f"flow_cat={flow_losses['cat'].item():.4f} "
                f"flow_dog={flow_losses['dog'].item():.4f} "
                f"infoot={loss_infoot.item():.4f} "
                f"{geometry} "
                f"cov_cat={covariance_losses['cat'].item():.4f} "
                f"cov_dog={covariance_losses['dog'].item():.4f} "
                f"cov_weighted={settings['covariance_weight'] * loss_covariance.item():.4f} "
                f"contrastive={loss_contrastive.item():.4f} "
                f"total={loss.item():.4f}"
            )

        if step % 200 == 0 or step == settings["steps"]:
            torch.save(
                {
                    "step": step,
                    "settings": settings,
                    "models": {
                        name: context.branch.pdae_state_dict()
                        for name, context in domains.items()
                    },
                    "training_configs": {
                        name: context.training_config
                        for name, context in domains.items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "matching_batchnorm": {
                        name: batchnorm_checkpoint(norm) for name, norm in batch_norms.items()
                    },
                },
                output_dir / f"step_{step:06d}.pt",
            )
            save_domain_checkpoints(domains, output_dir, step, batch_norms)


if __name__ == "__main__":
    main()
