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
from infoot_helper.infoot_cotraining_helper import (
    fit_transport,
    alignment_loss,
    conditional_mapping,
    save_domain_checkpoints,
)
from infoot_helper.native_flow import native_flow_loss
from infoot_helper.reference_rms import ReferenceRMSEMA
from infoot_helper.translation_contrastive import (
    translation_contrastive_loss,
)

from diffusion_ot.data.afhq import load_afhq_dataset
from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.data.manifests import read_jsonl
from infoot_helper.augmentation import augment_and_encode

def main():
    torch.manual_seed(42)
    device = "cuda" 

    settings = {
        "batch_size": 32,
        "query_count": 8,
        "steps": 2000,
        "h": 0.4,
        "mi_weight": 0.10,
        "reg": 0.02,
        "fit_iterations": 1200,
        "sampling_steps": 20,
        "flow_weight": 1.0,
        "contrastive_weight": 0.05,
        "rms_decay": 0.99,
    }

    output_dir = ROOT / "outputs/infoot_cotraining"
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = {}
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
    projection_rms = ReferenceRMSEMA(decay=settings["rms_decay"])
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
            latents[name] = augment_and_encode(domains[name], images)

        encoded = encode_batches(
            domains, latents, query_count=settings["query_count"]
        )

        flow_losses = {
            name: native_flow_loss(
                domains[name],
                encoded[name]["x0"],
                encoded[name]["v"],
            )
            for name in domains
        }
        loss_flow = torch.stack(list(flow_losses.values())).mean()

        cat_refs = encoded["cat"]["references"]["v"]
        dog_refs = encoded["dog"]["references"]["v"]
        projection_scales = projection_rms.update(
            {"cat": cat_refs, "dog": dog_refs}, step=step,
        )

        fit_diagnostics = {}
        P = fit_transport(
            cat_refs,
            dog_refs,
            h=settings["h"],
            mi_weight=settings["mi_weight"],
            reg=settings["reg"],
            iterations=settings["fit_iterations"],
            diagnostics=fit_diagnostics,
        )

        loss_infoot = alignment_loss(
            cat_refs,
            dog_refs,
            P,
            h=settings["h"],
            mi_weight=settings["mi_weight"],
            reg=settings["reg"],
        )

        contrastive_losses = []

        for source, target, plan in (
            ("cat", "dog", P),
            ("dog", "cat", P.T),
        ):
            mapped_v = conditional_mapping(
                encoded[source]["queries"]["v"],
                encoded[source]["references"]["v"],
                encoded[target]["references"]["v"],
                plan,
                h=settings["h"],
                scales=(projection_scales[source], projection_scales[target]),
            )

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
            + loss_infoot
            + settings["contrastive_weight"] * loss_contrastive
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
                f"fit={fit_diagnostics['status']} "
                f"rms_cat={projection_scales['cat']:.4f} "
                f"rms_dog={projection_scales['dog']:.4f} "
                f"{geometry} "
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
                    "transport_fit": fit_diagnostics,
                    "projection_rms": projection_rms.state_dict(),
                },
                output_dir / f"step_{step:06d}.pt",
            )
            save_domain_checkpoints(domains, output_dir, step, projection_rms)


if __name__ == "__main__":
    main()
