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


def _from_pretrained_kwargs(dtype_name: str | None) -> dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    return {"dtype": dtype} if dtype is not None else {}


def _load_with_dtype_fallback(loader: Any, path: str | Path, **kwargs: Any):
    try:
        return loader(path, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        return loader(path, **legacy_kwargs)


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

    kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        **_from_pretrained_kwargs(torch_dtype),
    }
    try:
        pipe = _load_with_dtype_fallback(
            DiffusionPipeline.from_pretrained,
            report.local_dir,
            **kwargs,
        )
    except Exception as exc:
        message = str(exc)
        if "scheduling_flow_match_sit" in message or "SiTFlowMatchScheduler" in message:
            raise RuntimeError(
                "The local SiT snapshot appears to mix scheduler metadata from an older "
                "BiliSakura/SiT-diffusers conversion with files from the current snapshot. "
                "Current HF main uses Diffusers FlowMatchEulerDiscreteScheduler via "
                "scheduler/scheduler_config.json and does not require "
                "scheduler/scheduling_flow_match_sit.py. Re-sync the local "
                "artifacts/pretrained/SiT-B-2-256 snapshot, or pin/download a commit whose "
                "model_index.json and scheduler files match."
            ) from exc
        raise
    pipe._diffusion_ot_local_dir = Path(report.local_dir)
    if device:
        pipe = pipe.to(device)
    return pipe


def load_sit_vae(
    pretrained_config_path: str | Path,
    project_root: str | Path | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    from diffusers import AutoencoderKL

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

    vae_path = Path(report.local_dir) / "vae"
    vae = _load_with_dtype_fallback(
        AutoencoderKL.from_pretrained,
        vae_path,
        **_from_pretrained_kwargs(torch_dtype),
    )
    if device:
        vae = vae.to(device)
    return vae


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
