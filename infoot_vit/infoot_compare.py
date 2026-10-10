"""Compare saved mappers on identical held-out queries, without OT refitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from infoot_vit.infoot_helper.feature_bank import FeatureBank, write_json
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.run_logging import RunLog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mappings", type=Path, nargs="+", required=True)
    p.add_argument("--query-bank", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    if min(a.count, a.threads) < 1:
        p.error("count and threads must be positive")
    torch.set_num_threads(a.threads)
    output = a.output_dir or ROOT / "results/infoot_vit" / f"compare_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    with RunLog(output, "comparison", vars(a)) as log:
        return _compare(a, output, log)


def _compare(a, output, log):
    bank = FeatureBank.load(a.query_bank)
    reports, population = [], None
    for i, path in enumerate(a.mappings):
        log.event("comparison_mapping_started", index=i, mapping=path)
        mapper = FeatureMapper.load(path)
        fitted_population = (mapper.source.artifact_id, mapper.target.artifact_id)
        if population is not None and population != fitted_population:
            raise ValueError("Comparison requires exactly the same source/target fit banks and representations.")
        population = fitted_population
        result, _ = mapper.project_bank(bank, output / f"{i:02d}_{mapper.mode}", count=a.count)
        reports.append(dict(mapping=str(path.resolve()), artifact_id=mapper.manifest["artifact_id"],
            mode=mapper.mode, partial=mapper.config["partial"], solver=mapper.config["solver"],
            projection=mapper.config["projection"], resources=mapper.manifest["resources"],
            fit_seconds=mapper.manifest["fit_seconds"], diagnostics=result.diagnostics))
        log.event("comparison_mapping_completed", index=i, mapper_id=mapper.manifest["artifact_id"])
    write_json(output / "comparison.json", dict(query_bank_id=bank.artifact_id, query_ids=bank.ids[:a.count],
        fit_banks=population, comparisons=reports,
        interpretation="Mapping-only numerical comparison. Include grouped_partial s=1 to isolate rejection; no image-quality claim.",
        diffusion_protocol="For fixed-checkpoint images, run infoot_test.py --generate with identical checkpoint, sampler, seed and guidance."))
    print(f"Saved mapping comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
