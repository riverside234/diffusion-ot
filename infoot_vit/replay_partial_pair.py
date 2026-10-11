"""Replay a saved first_failed_pair.pt before launching another full pair fit."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.partial import solver_config
from infoot_vit.infoot_helper.partial_batch import solve_partial_batch, VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report; a sibling .pt stores the last accepted plan.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inner-acceleration", choices=("none", "newton"), default="newton")
    parser.add_argument("--max-inner-steps", type=int)
    parser.add_argument("--copies", type=int, default=1, help="Identical pair replicas for batch testing; not an AFHQ throughput benchmark.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.copies < 1 or args.threads < 1 or (args.max_inner_steps is not None and args.max_inner_steps < 1):
        parser.error("Copies, threads and iteration budgets must be positive.")
    plan_path = args.output.with_suffix(".pt")
    if args.output.suffix != ".json" or args.input.resolve() in {args.output.resolve(), plan_path.resolve()}:
        parser.error("Use a separate .json output path; never overwrite the reproducer.")
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    example = torch.load(args.input, map_location="cpu", weights_only=True)
    config = solver_config(dict(example["config"], inner_acceleration=args.inner_acceleration))
    if args.max_inner_steps is not None:
        config["max_inner_steps"] = args.max_inner_steps
    tensors = {name: example[name].to(device=device, dtype=torch.float64) for name in ("cost", "kx", "ky", "a", "b")}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    plans, reports = solve_partial_batch(*(tensors[name][None].expand(args.copies, -1, -1) for name in ("cost", "kx", "ky")),
        a=tensors["a"], b=tensors["b"], keep_mass=example["keep_mass"], config=config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    success = all(report["status"] == "converged" for report in reports)
    record = dict(status="converged" if success else "failed", input=str(args.input), input_sha256=file_hash(args.input),
        source_id=example.get("source_id"), target_id=example.get("target_id"), device=str(device), solver=VERSION,
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        config=config, keep_mass=example["keep_mass"], copies=args.copies, seconds=elapsed,
        copies_per_second=args.copies/elapsed,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        interpretation="Exact saved pair replay. Duplicate copies do not measure throughput across different AFHQ pairs.",
        reports=reports, replica_max_abs_difference=float((plans-plans[:1]).abs().max()))
    save_tensor(plan_path, dict(plan=plans[0].cpu(), report=reports[0], config=config, input_sha256=record["input_sha256"]))
    write_json(args.output, record)
    print(json.dumps({k:v for k,v in record.items() if k != "reports"}, indent=2))
    print(f"Saved replay report: {args.output}; plan: {plan_path}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
