"""Project held-out SigLIP banks, optionally generate with a fixed PDAE checkpoint."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from infoot_vit.infoot_helper.feature_bank import FeatureBank, digest
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.run_logging import RunLog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--query-bank", type=Path, default=ROOT / "data/infoot_vit/cat_val")
    p.add_argument("--output-dir", type=Path, help="New directory for mapped tensors, metadata and optional images.")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--generate", action="store_true", help="Also run fixed-checkpoint diffusion inference; never train.")
    p.add_argument("--train-config", default="configs/stage1a_pdae_v2_l/dog.yaml")
    p.add_argument("--eval-config", default="configs/stage1a_eval/pdae_v2_l.yaml")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--weights", choices=["raw", "ema"], default="ema")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if min(a.count, a.chunk_size, a.threads, a.steps) < 1:
        p.error("Counts, chunk size, threads and steps must be positive")
    torch.set_num_threads(a.threads)
    if a.dry_run:
        # Validate small metadata before allocating supports/kernels or the DiT.
        m = json.loads((a.mapping / "manifest.json").read_text(encoding="utf-8"))
        q = json.loads((a.query_bank / "manifest.json").read_text(encoding="utf-8"))
        if (m["status"] != "complete" or m["representation"] != q["representation"] or q["split"] == "train"
                or m["source_domain"] != q["domain"] or set(q["ids"]) & (set(m["source_ids"]) | set(m["target_ids"]))
                or digest({k:v for k,v in m.items() if k != "artifact_id"}) != m["artifact_id"]
                or digest({k:v for k,v in q.items() if k != "artifact_id"}) != q["artifact_id"]):
            raise ValueError("Need completed mapper and compatible held-out query bank")
        print(json.dumps(dict(mapper_id=m["artifact_id"], mode=m["config"]["mode"], fit_resources=m["resources"],
            query_count=min(a.count, len(q["ids"])), mask="existing condition_padding_mask; True is padding",
            generate=a.generate, projection=m["config"]["projection"]), indent=2))
        return 0
    output = a.output_dir or ROOT / "results/infoot_vit" / f"{a.mapping.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    with RunLog(output, "mapping_test", vars(a)) as log:
        mapper, bank = FeatureMapper.load(a.mapping), FeatureBank.load(a.query_bank)
        log.event("artifacts_loaded", mapper_id=mapper.manifest["artifact_id"], query_bank_id=bank.artifact_id)
        result, manifest = mapper.project_bank(bank, output, count=a.count, chunk_size=a.chunk_size, run_log=log)
        if a.generate:
            from infoot_vit.infoot_helper.evaluate_mapping import generate
            log.event("generation_started")
            generate(mapper, bank, result, output, root=ROOT, train_config=a.train_config, eval_config=a.eval_config,
                     checkpoint=a.checkpoint, weights=a.weights, steps=a.steps, guidance=a.guidance, solver=a.solver,
                     seed=a.seed, batch_size=a.chunk_size, device=a.device)
            log.event("generation_completed", report="generation_report.json")
        print(f"Saved {len(manifest['ids'])} projections to {output}; mode={mapper.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
