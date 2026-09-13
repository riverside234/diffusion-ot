"""Run the preserved official-based InfoOT workflow in an isolated process."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "legacy" / "infoot_official_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "cache", "check", "test"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments
    manifest = json.loads((SNAPSHOT / "snapshot.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if hashlib.sha256((SNAPSHOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Legacy snapshot was modified: {record['path']}")
    # diffusion_ot and its subpackages are namespace packages. Their frozen
    # files win resolution; unchanged model/data/Stage 1A dependencies fall back
    # to the main source tree. No import redirection or source edits are needed.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(SNAPSHOT / "src"))
    protected = ["losses.infoot", "losses.semantic_prior", "losses.conditional_structure",
                 "training.train_joint_infoot", "evaluation.stage1b_eval"]
    paths = {}
    for suffix in protected:
        module = importlib.import_module(f"diffusion_ot.{suffix}")
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(SNAPSHOT):
            raise RuntimeError("Legacy workflows need a fresh Python process; an active module was already imported.")
        paths[suffix] = str(path)
    if args.command == "check":
        print(json.dumps({"snapshot": manifest["version"], "source_commit": manifest["source_commit"],
                          "verified_files": len(manifest["files"]), "modules": paths}, indent=2))
        return 0
    if args.command == "test":
        import pytest
        return pytest.main([str(SNAPSHOT / "tests"), *forwarded])
    scripts = {"train": "train_joint_infoot.py", "evaluate": "evaluate_infoot_alignment.py",
               "cache": "cache_infoot_structure.py"}
    script = SNAPSHOT / "scripts" / scripts[args.command]
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
