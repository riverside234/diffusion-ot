from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class SnapshotReport:
    config_path: str
    repo_id: str
    subfolder: str
    revision: str | None
    local_dir: str
    exists: bool
    expected_files: list[str]
    missing_files: list[str]
    downloaded: bool = False
    metadata_file: str | None = None

    @property
    def ok(self) -> bool:
        return self.exists and not self.missing_files

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return current


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = _parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return data


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this repo's Stage 0 configs."""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value == "":
            return None
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"null", "none", "~"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(item.strip()) for item in inner.split(",")]
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            values = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                item_text = lines[index][1][2:].strip()
                index += 1
                if item_text:
                    values.append(parse_scalar(item_text))
                elif index < len(lines) and lines[index][0] > indent:
                    item_value, index = parse_block(index, lines[index][0])
                    values.append(item_value)
                else:
                    values.append(None)
            return values, index

        mapping: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
            key, separator, value = lines[index][1].partition(":")
            if separator != ":":
                raise ValueError(f"Unsupported YAML line: {lines[index][1]}")
            index += 1
            value = value.strip()
            if value:
                mapping[key.strip()] = parse_scalar(value)
            elif index < len(lines) and lines[index][0] > indent:
                child, index = parse_block(index, lines[index][0])
                mapping[key.strip()] = child
            else:
                mapping[key.strip()] = None
        return mapping, index

    parsed, end_index = parse_block(0, lines[0][0] if lines else 0)
    if end_index != len(lines):
        raise ValueError("Unsupported YAML structure in config file.")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML value must be a mapping.")
    return parsed


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    root = Path(project_root) if project_root is not None else find_project_root()
    return (root / raw_path).resolve()


def effective_project_root(
    config: dict[str, Any] | None = None,
    fallback: str | Path | None = None,
) -> Path:
    configured = config.get("project_root") if config else None
    if configured:
        return Path(configured).expanduser().resolve()
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    return find_project_root()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_project_local_path(
    path: str | Path,
    project_root: str | Path | None = None,
    field_name: str = "path",
) -> Path:
    root = effective_project_root(fallback=project_root)
    resolved = resolve_project_path(path, root)
    if not is_relative_to(resolved, root):
        raise ValueError(
            f"{field_name} must stay inside the project root. "
            f"Got {resolved}; project root is {root}."
        )
    return resolved


def expected_local_dir(config: dict[str, Any], project_root: str | Path | None = None) -> Path:
    local_dir = config.get("local_dir")
    if not local_dir:
        raise ValueError("Pretrained config is missing local_dir.")
    return resolve_project_local_path(local_dir, project_root, field_name="local_dir")


def missing_expected_files(local_dir: Path, expected_files: list[str]) -> list[str]:
    return [relative for relative in expected_files if not (local_dir / relative).exists()]


def _download_snapshot_to_parent(config: dict[str, Any], local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when download_if_missing is enabled."
        ) from exc

    repo_id = config["repo_id"]
    revision = config.get("revision")
    subfolder = config["subfolder"]
    allow_patterns = config.get("allow_patterns") or [f"{subfolder}/**"]
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir.parent),
    )


def verify_sit_snapshot(
    config_path: str | Path,
    project_root: str | Path | None = None,
    allow_download: bool | None = None,
    write_report: bool = True,
) -> SnapshotReport:
    config = load_yaml_config(config_path)
    root = effective_project_root(config, fallback=project_root or find_project_root(Path(config_path)))
    local_dir = expected_local_dir(config, root)
    expected_files = list(config.get("expected_files") or [])
    download_allowed = config.get("download_if_missing", False) if allow_download is None else allow_download

    downloaded = False
    missing = missing_expected_files(local_dir, expected_files) if local_dir.exists() else expected_files
    if missing and download_allowed:
        _download_snapshot_to_parent(config, local_dir)
        downloaded = True
        missing = missing_expected_files(local_dir, expected_files) if local_dir.exists() else expected_files

    metadata_file = config.get("metadata_file")
    metadata_path = (
        resolve_project_local_path(metadata_file, root, field_name="metadata_file")
        if metadata_file
        else local_dir / "snapshot_report.json"
    )
    report = SnapshotReport(
        config_path=str(Path(config_path)),
        repo_id=str(config.get("repo_id", "")),
        subfolder=str(config.get("subfolder", "")),
        revision=config.get("revision"),
        local_dir=str(local_dir),
        exists=local_dir.exists(),
        expected_files=expected_files,
        missing_files=missing,
        downloaded=downloaded,
        metadata_file=str(metadata_path),
    )

    if write_report and report.ok:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    return report


def format_snapshot_report(report: SnapshotReport) -> str:
    lines = [
        f"repo_id: {report.repo_id}",
        f"subfolder: {report.subfolder}",
        f"revision: {report.revision}",
        f"local_dir: {report.local_dir}",
        f"exists: {report.exists}",
        f"downloaded: {report.downloaded}",
        f"ok: {report.ok}",
    ]
    if report.missing_files:
        lines.append("missing_files:")
        lines.extend(f"  - {item}" for item in report.missing_files)
    return "\n".join(lines)
