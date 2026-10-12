"""Small analysis results, with resumable experiment state kept outside results/."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .feature_bank import file_hash


def create_work_directory(root, output, requested=None):
    """Keep caches and per-pair logs in the existing Git-ignored outputs tree."""
    root, output = Path(root).resolve(), Path(output).resolve()
    work = (Path(requested).resolve() if requested is not None else
            root / "outputs/infoot_vit" / f"{output.name}_{uuid.uuid4().hex[:8]}")
    if (work == output or work.is_relative_to(output) or output.is_relative_to(work)
            or work.is_relative_to(root / "results")):
        raise ValueError("Working files must be outside results/ and separate from the analysis output.")
    work.mkdir(parents=True, exist_ok=False)
    return work


def fit_diagnostics(directory):
    """Embed reports and every failed-pair history; skip thousands of success logs.

    These are diagnostic copies, not a loadable or certified mapping artifact.
    Original files stay in the work directory for resume and strict hash checks.
    Missing files are normal when fitting fails before producing a manifest.
    """
    directory = Path(directory)
    files, errors = {}, {}
    paths = [directory / name for name in ("manifest.json", "fit_report.json", "image_report.json",
             "pair_batch_report.json", "pair_selection_report.json", "pair_failures.jsonl")]
    attempts = sorted((directory / "logs").glob("*"))
    if attempts:
        paths += [attempts[-1] / name for name in ("run.json", "error.json", "inspection.json")]
    for path in paths:
        if not path.is_file():
            continue
        key = path.relative_to(directory).as_posix()
        try:
            with path.open(encoding="utf-8") as handle:
                data = ([json.loads(line) for line in handle if line.strip()]
                        if path.suffix == ".jsonl" else json.load(handle))
            files[key] = dict(sha256=file_hash(path), data=data)
        except (OSError, ValueError) as error:
            # Never hide the original experiment failure with an export failure.
            errors[key] = f"{type(error).__name__}: {error}"
    return dict(directory=str(directory.resolve()), files=files, export_errors=errors)


def mapping_diagnostics(directory, manifest):
    directory = Path(directory)
    reports = sorted((directory / "logs").glob("*/mapping_report.json"))
    return dict(manifest=manifest, summary=(json.loads(reports[-1].read_text(encoding="utf-8"))
                                          if reports else None))
