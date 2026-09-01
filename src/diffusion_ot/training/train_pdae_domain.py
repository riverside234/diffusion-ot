from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


@dataclass
class PDAETrainReport:
    config_path: str
    domain: str
    split: str
    output_dir: str
    final_step: int
    train_samples: int
    checkpoint_path: str | None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _load_stage_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    path = Path(config_path).resolve()
    config = load_yaml_config(path)
    root = effective_project_root(config, fallback=find_project_root(path.parent))
    return config, path, root


def _resolve_config_path(config: dict[str, Any], root: Path, key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Stage config is missing {key}.")
    return resolve_project_local_path(value, root, field_name=key)


def _torch_dtype_from_model(transformer, fallback):
    for parameter in transformer.parameters():
        return parameter.dtype
    return fallback


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _optimizer_groups(branch, train_config: dict[str, Any]):
    encoder_params = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    ]
    adapter_params = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    groups = []
    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": float(train_config.get("lr_encoder", train_config.get("lr", 1.0e-4))),
            }
        )
    if adapter_params:
        groups.append(
            {
                "params": adapter_params,
                "lr": float(train_config.get("lr_adapter", train_config.get("lr", 1.0e-4))),
            }
        )
    return groups


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_pdae_domain(
    config_path: str | Path,
    device: str | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> PDAETrainReport:
    import torch
    from torch.utils.data import DataLoader

    from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch
    from diffusion_ot.integrations.sit_diffusers import load_sit_components, validate_transformer_config
    from diffusion_ot.losses.pdae_flow import (
        make_linear_flow_target,
        pdae_flow_snr_weight,
        pdae_velocity_gap_loss,
        velocity_gap_target,
    )
    from diffusion_ot.models.pdae_sit import (
        build_pdae_sit_branch,
        make_null_class_labels,
    )

    config, resolved_config_path, root = _load_stage_config(config_path)
    train_config = _nested(config, "train")
    dataloader_config = _nested(config, "dataloader")
    flow_config = _nested(config, "flow")
    class_config = _nested(config, "class_conditioning")
    loss_weight_config = _nested(config, "loss_weighting")

    domain = str(config.get("domain", "")).lower()
    if not domain:
        raise ValueError("Stage config is missing domain.")
    split = str(config.get("split", "train")).lower()

    data_config_path = _resolve_config_path(config, root, "data_config")
    model_config_path = _resolve_config_path(config, root, "model_config")
    model_config = load_yaml_config(model_config_path)
    pretrained_config_path = (
        resolve_project_local_path(config["pretrained_config"], root, field_name="pretrained_config")
        if config.get("pretrained_config")
        else resolve_project_local_path(model_config["pretrained"], root, field_name="model.pretrained")
    )
    output_dir = resolve_project_local_path(
        config.get("output_dir", f"outputs/stage1a_{domain}_sit_b2"),
        root,
        field_name="output_dir",
    )

    seed = int(train_config.get("seed", 20260901))
    torch.manual_seed(seed)
    device = device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    limit = train_config.get("limit")
    dataset = CachedLatentDataset(
        data_config_path=data_config_path,
        domain=domain,
        split=split,
        project_root=root,
        limit=int(limit) if limit is not None else None,
        validate_exists=not dry_run,
    )

    final_step = int(max_steps if max_steps is not None else train_config.get("max_steps", 1000))
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = PDAETrainReport(
            config_path=str(resolved_config_path),
            domain=domain,
            split=split,
            output_dir=str(output_dir),
            final_step=0,
            train_samples=len(dataset),
            checkpoint_path=None,
            dry_run=True,
        )
        _write_json(output_dir / "dry_run_report.json", report.to_dict())
        return report

    components = load_sit_components(
        pretrained_config_path,
        project_root=root,
        device=device,
        torch_dtype=str(config.get("torch_dtype", "float32")),
    )
    transformer = components.transformer
    transformer.eval()

    mismatches = validate_transformer_config(transformer, model_config.get("expected_transformer_config") or {})
    if mismatches:
        raise ValueError("Unexpected SiT transformer config:\n" + "\n".join(f"  - {item}" for item in mismatches))

    branch = build_pdae_sit_branch(transformer, model_config=model_config, stage_config=config)
    model_dtype = _torch_dtype_from_model(transformer, torch.float32)
    branch.to(device=device, dtype=model_dtype)
    branch.train()

    batch_size = int(dataloader_config.get("batch_size", 16))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(dataloader_config.get("num_workers", 4)),
        pin_memory=bool(dataloader_config.get("pin_memory", True)),
        drop_last=bool(dataloader_config.get("drop_last", True)),
        collate_fn=collate_latent_batch,
        generator=generator,
    )
    batch_iter = _cycle(loader)

    optimizer = torch.optim.AdamW(
        _optimizer_groups(branch, train_config),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        betas=tuple(train_config.get("betas", [0.9, 0.999])),
    )
    grad_clip_norm = train_config.get("grad_clip_norm", 1.0)
    flow_direction = str(flow_config.get("direction", "noise_to_data"))
    time_eps = float(flow_config.get("time_eps", 1.0e-5))
    null_label = class_config.get("null_label")
    log_every = int(train_config.get("log_every", 20))
    save_every = int(train_config.get("save_every", 500))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)

    for step in range(1, final_step + 1):
        batch = next(batch_iter)
        x0 = batch["x0_latent"].to(device=device, dtype=model_dtype, non_blocking=True)
        target = make_linear_flow_target(x0, eps=time_eps, direction=flow_direction)
        class_labels = make_null_class_labels(
            transformer,
            batch_size=x0.shape[0],
            device=x0.device,
            null_label=int(null_label) if null_label is not None else None,
        )

        output = branch(
            x0_latent=x0,
            x_t=target.x_t,
            timestep=target.t,
            class_labels=class_labels,
        )
        pred_delta_v = output.delta_sample
        base_v = output.base_sample
        weight = pdae_flow_snr_weight(
            t=target.t.float(),
            direction=flow_direction,
            gamma=float(loss_weight_config.get("gamma", 0.25)),
            normalize_mean_to=loss_weight_config.get("normalize_mean_to", 1.0),
            clamp_min=loss_weight_config.get("clamp_min", 0.05),
            clamp_max=loss_weight_config.get("clamp_max", 5.0),
        )
        loss = pdae_velocity_gap_loss(
            pred_delta_v.float(),
            target.target_v.float(),
            base_v.float(),
            weight=weight.to(device=pred_delta_v.device, dtype=torch.float32),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in branch.parameters() if parameter.requires_grad],
                float(grad_clip_norm),
            )
        optimizer.step()

        if step == 1 or step % log_every == 0:
            target_delta = velocity_gap_target(target.target_v.float(), base_v.float(), detach=True)
            residual_rms = target_delta.square().mean().sqrt().item()
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "target_delta_rms": residual_rms,
                        "weight_mean": float(weight.detach().mean().cpu()),
                    },
                    sort_keys=True,
                )
            )

        if step % save_every == 0 or step == final_step:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "domain": domain,
                    "model": branch.pdae_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )

    return PDAETrainReport(
        config_path=str(resolved_config_path),
        domain=domain,
        split=split,
        output_dir=str(output_dir),
        final_step=final_step,
        train_samples=len(dataset),
        checkpoint_path=str(checkpoint_path),
        dry_run=False,
    )
