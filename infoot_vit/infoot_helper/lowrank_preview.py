"""Read-only preview of stopped global low-rank fits, without claiming convergence.

Kept outside lowrank/ and the fit implementation hash list so adding this test
entry point does not invalidate existing solver checkpoints for resume.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch

from .feature_bank import FeatureBank, checked_file, compatible_banks, digest, file_hash, write_json
from .sampling import sample_ids, sampling_record
from ..lowrank.config import MODE, SCHEMA
from ..lowrank.experiment import fingerprint, validate_kernels
from ..lowrank.kernels import features, projection_state
from ..lowrank.mapping import LowRankMapper
from ..lowrank.storage import checkpoint_digest, validate_factors
from .device import move, resolve_device


def check_preview_output(directory, output):
    if Path(output).resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("Checkpoint preview output must be outside the original fit directory.")


def _unchanged(directory, snapshot):
    preview = snapshot["checkpoint_preview"]
    if (file_hash(directory / "manifest.json") != preview["original_manifest_sha256"]
            or file_hash(directory / preview["checkpoint"]["file"]) != preview["checkpoint"]["sha256"]):
        raise ValueError("Fit changed while loading its checkpoint; stop fitting before previewing.")


def load_preview_snapshot(directory):
    """Verify the stored fit identity and factors; return an in-memory identity only."""
    directory = Path(directory).resolve()
    original_hash = file_hash(directory / "manifest.json")
    m = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (m.get("schema") != SCHEMA or m.get("config", {}).get("mode") != MODE
            or m.get("status") not in {"failed", "interrupted"}):
        raise ValueError("--preview-checkpoint requires a stopped failed/interrupted grouped_patch_lowrank fit.")
    if set(m["files"]) != {"image", "kernels", "samples"}:
        raise ValueError("Preview requires registered image/kernels/samples and unregistered global factors.")
    report = dict(config=m["config"], resources=None, sampling=m["sampling"],
        bank_ids=[m[f"{name}_bank"]["artifact_id"] for name in ("source", "target")],
        representation=m["representation"], runtime=m["runtime"], implementation=m["implementation"])
    # Use SAVED implementation/runtime identities, just as recorded at fitting.
    if m.get("artifact_id") is not None or fingerprint(report) != m.get("fit_fingerprint"):
        raise ValueError("Stopped low-rank fit fingerprint mismatch.")
    for name in ("source", "target"):
        bank = json.loads((directory / m[f"{name}_bank"]["path"] / "manifest.json").read_text(encoding="utf-8"))
        ids = sample_ids(bank["ids"], m["config"]["sampling"]["images_per_domain"], m["config"]["sampling"]["seed"])
        if (digest({k: v for k, v in bank.items() if k != "artifact_id"}) != bank.get("artifact_id")
                or bank["artifact_id"] != m[f"{name}_bank"]["artifact_id"]
                or bank["split"] != "train" or bank["domain"] != m[f"{name}_domain"]
                or bank["representation"] != m["representation"] or ids != m[f"{name}_ids"]
                or sampling_record(bank["ids"], ids, m["config"]["sampling"]["seed"]) != m["sampling"][name]):
            raise ValueError("Preview training-bank identity or ordered sample IDs changed.")
    for entry in m["files"].values():
        checked_file(directory, entry)
    path = (directory / "factors/latest.pt").resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Missing factors/latest.pt inside the stopped fit directory.")
    entry = dict(file="factors/latest.pt", sha256=file_hash(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("fit_fingerprint") != m["fit_fingerprint"]
            or state.get("checkpoint_sha256") != checkpoint_digest(state)):
        raise ValueError("Preview checkpoint factor/metadata checksum or fit identity mismatch.")
    patches = m["representation"]["grid"][0] * m["representation"]["grid"][1]
    shape = (len(m["source_ids"]) * patches, len(m["target_ids"]) * patches, m["config"]["transport_rank"])
    constraints = validate_factors(state, m["config"]["optimizer"], shape)
    step, record = state.get("step"), state.get("record")
    if (type(step) is not int or not 0 < step <= m["config"]["optimizer"]["max_steps"]
            or not isinstance(record, dict) or record.get("step") != step
            or record.get("status") not in {"max_steps", "line_search_failed"}):
        raise ValueError("Preview needs a recorded, nonconverged factor iteration; use normal testing for final artifacts.")
    snapshot = deepcopy(m)
    snapshot["files"]["factors"] = entry
    snapshot["checkpoint_preview"] = dict(policy="lowrank_checkpoint_preview_v1", converged=False,
        label=f"UNCONVERGED CHECKPOINT PREVIEW | step {step} | {record['status']}",
        step=step, original_status=m["status"], solver_status=record["status"],
        original_manifest_sha256=original_hash, checkpoint=entry,
        fit_fingerprint=m["fit_fingerprint"], factor_validation=constraints, last_record=record,
        interpretation="Read-only float32 checkpoint preview, not a converged mapping or proof of image quality. No refit or factor repair.")
    snapshot["artifact_id"] = digest(snapshot)
    _unchanged(directory, snapshot)
    return snapshot, state


class LowRankCheckpointPreview(LowRankMapper):
    """Reuse grouped projection/PDAE interfaces; only factor loading is different."""
    @classmethod
    def load(cls, directory, *, device=None, projection=None, run_log=None):
        directory = Path(directory).resolve()
        m, state = load_preview_snapshot(directory)
        resolve_device(device if device is not None else m["config"].get("device", "cpu"))
        banks = []
        for name in ("source", "target"):
            bank = FeatureBank.load(directory / m[f"{name}_bank"]["path"])
            if bank.artifact_id != m[f"{name}_bank"]["artifact_id"]:
                raise ValueError("Preview training bank changed while loading.")
            banks.append(bank.subset(m[f"{name}_ids"]))
        compatible_banks(*banks)
        mapper = cls(directory, m, state, *banks, device=device, projection=projection)
        _unchanged(directory, m)
        return mapper

    def __init__(self, directory, manifest, state, source, target, *, device=None, projection=None):
        files = self._initialize(directory, manifest, source, target, device, projection)
        self.factor_validation = manifest["checkpoint_preview"]["factor_validation"]
        q, r, g = (state[k].to(device=self.device, dtype=torch.float64) for k in ("q", "r", "g"))
        kernel = torch.load(files["kernels"], map_location="cpu", weights_only=True)
        n, m = q.shape[0], r.shape[0]
        validate_kernels(kernel, (n, m, self.config["kernel_rank"]), self.x.shape[-1], self.config["kernel"])
        self.kernel_source = move(kernel["source"], self.device)
        target_state = move(kernel["target"], self.device)
        multiplier = self.config["projection"]["bandwidth_multiplier"]
        # Same saved-factor algebra as LowRankMapper, kept here to preserve OLD
        # solver fingerprints. Dense-reference and final-loader parity are tested.
        if multiplier == 1:
            fx, fy = kernel["fx"].to(self.x), kernel["fy"].to(self.y)
        else:
            self.kernel_source = projection_state(self.kernel_source, multiplier)
            target_state = projection_state(target_state, multiplier)
            chunk = self.config["optimizer"]["chunk_size"]
            fx = features(self.x.flatten(0, 1), self.kernel_source, chunk)
            fy = features(self.y.flatten(0, 1), target_state, chunk)
        self.patch_projection_h = self.kernel_source["h"]
        self.cross = ((fx.T @ q) / g) @ (fy.T @ r).T
        self.density_y = (fy @ fy.mean(0)).reshape(len(self.y), self.y.shape[1])
        self.target_kernel_features = fy.reshape(len(self.y), self.y.shape[1], -1)
        if (self.density_y <= 0).any() or not torch.isfinite(self.density_y).all() or not torch.isfinite(self.cross).all():
            raise ValueError("Invalid checkpoint-preview KDE density.")

    def map_features(self, *args, **kwargs):
        result = super().map_features(*args, **kwargs)
        if kwargs.get("return_metadata", False):
            result.diagnostics["checkpoint_preview"] = self.manifest["checkpoint_preview"]
        return result

    def project_bank(self, bank, output, **kwargs):
        check_preview_output(self.directory, output)
        return super().project_bank(bank, output, **kwargs)

    def _save_mapping(self, bank, output, ids, result):
        m = super()._save_mapping(bank, output, ids, result)
        m["checkpoint_preview"] = self.manifest["checkpoint_preview"]
        m["checkpoint_preview_implementation_sha256"] = file_hash(Path(__file__))
        m["artifact_id"] = digest({k: v for k, v in m.items() if k != "artifact_id"})
        write_json(output / "manifest.json", m)
        return m


def label_preview_grid(path, preview):
    """A header preserves all three image rows and identifies unfinished fits."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()
    text = preview["label"]
    bounds = font.getbbox(text)
    banner = Image.new("RGB", (bounds[2] - bounds[0] + 16, bounds[3] - bounds[1] + 12), "#fff0c2")
    ImageDraw.Draw(banner).text((8 - bounds[0], 6 - bounds[1]), text, font=font, fill="#5f3900")
    banner = banner.resize((banner.width * 2, banner.height * 2), Image.Resampling.NEAREST)
    with Image.open(path) as grid:
        labeled = Image.new("RGB", (max(grid.width, banner.width), grid.height + banner.height), "#fff0c2")
        labeled.paste(banner, (0, 0))
        labeled.paste(grid, (0, banner.height))
    labeled.save(path)
