from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffusion_ot.integrations.hf_snapshot import load_yaml_config, verify_sit_snapshot


@dataclass
class SiTComponents:
    pipeline: Any
    transformer: Any
    vae: Any
    scheduler: Any
    local_dir: Path


def _torch_dtype(dtype_name: str | None):
    if not dtype_name or dtype_name == "auto":
        return None
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_sit_pipeline(
    pretrained_config_path: str | Path,
    project_root: str | Path | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    from diffusers import DiffusionPipeline

    config = load_yaml_config(pretrained_config_path)
    report = verify_sit_snapshot(
        pretrained_config_path,
        project_root=project_root,
        allow_download=config.get("download_if_missing", False),
        write_report=True,
    )
    if not report.ok:
        missing = "\n".join(f"  - {item}" for item in report.missing_files)
        raise FileNotFoundError(
            f"SiT snapshot is incomplete at {report.local_dir}.\nMissing files:\n{missing}"
        )

    pipe = DiffusionPipeline.from_pretrained(
        report.local_dir,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        torch_dtype=_torch_dtype(torch_dtype),
    )
    pipe._diffusion_ot_local_dir = Path(report.local_dir)
    if device:
        pipe = pipe.to(device)
    return pipe


def load_sit_components(
    pretrained_config_path: str | Path,
    project_root: str | Path | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
) -> SiTComponents:
    pipe = load_sit_pipeline(
        pretrained_config_path,
        project_root=project_root,
        device=device,
        torch_dtype=torch_dtype,
    )
    return SiTComponents(
        pipeline=pipe,
        transformer=pipe.transformer,
        vae=pipe.vae,
        scheduler=pipe.scheduler,
        local_dir=Path(getattr(pipe, "_diffusion_ot_local_dir", ".")),
    )


def validate_transformer_config(transformer: Any, expected: dict[str, Any]) -> list[str]:
    config = getattr(transformer, "config", None)
    if config is None:
        return ["transformer has no config attribute"]

    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = getattr(config, key, None)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    return mismatches
