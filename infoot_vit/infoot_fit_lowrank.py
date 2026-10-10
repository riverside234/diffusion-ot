"""Separate nonnegative low-rank grouped patch/partial InfoOT experiments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from infoot_vit.lowrank.experiment import fit,inspect_fit,compact_inspection
from infoot_vit.lowrank.kernels import METHODS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"infoot_vit/configs/grouped_patch_lowrank.yaml")
    p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--source-bank")
    p.add_argument("--target-bank")
    p.add_argument("--device",help="Numerical device, overriding the YAML (e.g. cuda:1 or cpu).")
    p.add_argument("--output-root",type=Path)
    p.add_argument("--resume",type=Path)
    p.add_argument("--max-steps",type=int,help="May be increased on resume; all mathematical settings remain fixed.")
    p.add_argument("--image-max-steps",type=int,help="Image-router outer budget, separate from --max-steps. Unregistered routers restart on resume.")
    p.add_argument("--h",type=float,help="image_solver.h: image-router fit bandwidth, not patch kernel.h.")
    p.add_argument("--lam",type=float,help="image_solver.lam: image-router MI weight, not patch optimizer.lam.")
    p.add_argument("--reg",type=float,help="image_solver.reg: image-router entropy weight, not patch optimizer.reg.")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--tune",action="store_true",help="Fit only the image router; terminal output only, no files/plans saved.")
    p.add_argument("--tune-log-every",type=int,default=25,help="Router tuning progress interval (default 25).")
    p.add_argument("--kernel-check-only",action="store_true",help="Audit grouped_patch_lowrank kernels before fitting either transport plan; saves diagnostics.")
    p.add_argument("--kernel-method",choices=METHODS,help="New fit only; override the kernel estimator for a controlled comparison.")
    p.add_argument("--kernel-rank",type=int,help="New fit only; override kernel rank, retaining resource checks.")
    a = p.parse_args(argv)
    if a.tune and (a.resume or a.output_root or a.dry_run or a.kernel_check_only):
        p.error("--tune cannot be combined with --resume, --output-root, --dry-run or --kernel-check-only")
    if a.tune_log_every < 1:
        p.error("--tune-log-every must be positive")
    c = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    for name in ("source_bank","target_bank","device"):
        if getattr(a,name) is not None:c[name] = getattr(a,name)
    if a.max_steps is not None:c.setdefault("optimizer",{})["max_steps"] = a.max_steps
    if a.image_max_steps is not None:c.setdefault("image_solver",{})["max_outer_steps"] = a.image_max_steps
    for name in ("h","lam","reg"):
        if getattr(a,name) is not None:c.setdefault("image_solver",{})[name] = getattr(a,name)
    if a.kernel_method is not None:
        c.setdefault("kernel",{})["method"] = a.kernel_method
        if a.kernel_method == METHODS[0]: c["kernel"]["orthogonal"] = False
    if a.kernel_rank is not None:c["kernel_rank"] = a.kernel_rank
    if a.tune:
        from infoot_vit.infoot_helper.tuning import tune_image_router
        report = tune_image_router(c,root=a.project_root,lowrank=True,log_every=a.tune_log_every)
        return 0 if report["status"] == "converged" else 2
    if a.dry_run:print(json.dumps(compact_inspection(inspect_fit(c,a.project_root)),indent=2))
    else:
        directory = fit(c,root=a.project_root,output_root=a.output_root,resume=a.resume,kernel_check_only=a.kernel_check_only)
        print(f"Saved {'kernel diagnostics' if a.kernel_check_only else 'low-rank mapping'}: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
