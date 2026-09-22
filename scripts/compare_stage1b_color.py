"""Compare HistoGAN metrics only on matching fixed Stage 1B validation panels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def read_records(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            # A resumed run can repeat its initial validation step.
            records[int(record["step"])] = record
    return records


def compare_validation_logs(baseline_path, candidate_path, *, step=None):
    baseline, candidate = read_records(baseline_path), read_records(candidate_path)
    common = baseline.keys() & candidate.keys()
    if not common or (step is not None and step not in common):
        raise ValueError("The requested/common validation step is missing.")
    step = max(common) if step is None else step
    left, right = baseline[step], candidate[step]
    if left.get("weights") != right.get("weights") or "weights" not in left:
        raise ValueError("Validation weight modes must match (raw versus EMA).")
    a, b = left["decoded_translation"], right["decoded_translation"]
    for key in ("seed", "query_ids", "reference_ids", "num_steps", "guidance_scale",
                "color_histogram_protocol", "color_histogram_parameters", "color_histogram_target"):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"Cannot compare different/missing validation {key}; re-evaluate both checkpoints with one protocol.")
    for key in ("fit_bandwidth", "projection_bandwidth"):
        before = left.get("projection_probe", {}).get(key)
        after = right.get("projection_probe", {}).get(key)
        if before is None or after is None or before != after:
            raise ValueError(f"Cannot compare different/missing {key}.")
    report = {"step": step, "weights": left["weights"], "protocol": a["color_histogram_protocol"],
              "baseline": str(baseline_path), "candidate": str(candidate_path), "directions": {}}
    for direction, source in (("cat_to_dog", "cat"), ("dog_to_cat", "dog")):
        # Schema 2 names weight-zero control measurements as diagnostics.
        x, y = [row[direction].get("per_image_color_histogram_loss",
                   row[direction].get("diagnostics", {}).get("per_image_color_histogram_distance"))
                for row in (a, b)]
        if x is None or y is None:
            raise ValueError(f"Missing per-image color metrics for {direction}.")
        if not x or len(x) != len(y) or len(x) != len(a["query_ids"][source]):
            raise ValueError(f"Per-image metrics/IDs do not match for {direction}.")
        if not all(math.isfinite(v) and 0 <= v <= 1.00001 for v in x + y):
            raise ValueError("Invalid Hellinger distances in validation metrics.")
        report["directions"][direction] = {
            "samples": len(x), "baseline_mean": mean(x), "candidate_mean": mean(y),
            "paired_mean_delta": mean(v - u for u, v in zip(x, y)),
            "fraction_improved": mean(v < u for u, v in zip(x, y)),
            "baseline_structure": a[direction].get("structure_loss", a[direction].get("diagnostics", {}).get("structure_cosine_distance")),
            "candidate_structure": b[direction].get("structure_loss", b[direction].get("diagnostics", {}).get("structure_cosine_distance")),
            "baseline_grid": a[direction].get("validation_grid"),
            "candidate_grid": b[direction].get("validation_grid"),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", required=True, help="JSON comparison report path.")
    args = parser.parse_args()
    report = compare_validation_logs(args.baseline, args.candidate, step=args.step)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
