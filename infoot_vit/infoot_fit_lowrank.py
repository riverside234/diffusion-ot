"""Separate nonnegative low-rank grouped patch/partial InfoOT experiments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from infoot_vit.lowrank.experiment import fit,inspect_fit,compact_inspection


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
    p.add_argument("--dry-run",action="store_true")
    a = p.parse_args(argv)
    c = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    for name in ("source_bank","target_bank","device"):
        if getattr(a,name) is not None:c[name] = getattr(a,name)
    if a.max_steps is not None:c.setdefault("optimizer",{})["max_steps"] = a.max_steps
    if a.image_max_steps is not None:c.setdefault("image_solver",{})["max_outer_steps"] = a.image_max_steps
    if a.dry_run:print(json.dumps(compact_inspection(inspect_fit(c,a.project_root)),indent=2))
    else:print(f"Saved low-rank mapping: {fit(c,root=a.project_root,output_root=a.output_root,resume=a.resume)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
