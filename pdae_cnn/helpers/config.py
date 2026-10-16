"""Load existing PDAE training recipes for the compact CNN trainer."""

from pathlib import Path

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


def load_config(config_path, *, project_root=None, device=None, output_dir=None):
    """Load a YAML recipe, apply overrides, and resolve its project-local paths."""
    path = Path(config_path).expanduser().resolve()
    config = load_yaml_config(path)
    if project_root is not None:
        config["project_root"] = str(project_root)
    root = effective_project_root(config, fallback=find_project_root(path.parent))
    config["project_root"] = str(root)

    if device is not None:
        config["device"] = device
    if output_dir is not None:
        config["output_dir"] = str(output_dir)

    path_keys = ("data_config", "model_config", "output_dir")
    if config.get("pretrained_config"):
        path_keys += ("pretrained_config",)
    for key in path_keys:
        config[key] = str(resolve_project_local_path(config[key], root, field_name=key))
    return config
