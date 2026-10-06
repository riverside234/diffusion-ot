from infoot_helper import infoot
from infoot_helper.transport.plan_io import bank_metadata
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.4)
parser.add_argument("--reg", type=float, default=0.02)
parser.add_argument("--lam", type=float, default=1)
parser.add_argument("--max-iter", type=int, default=50)
parser.add_argument("--sinkhorn-iter", type=int, default=5000)
parser.add_argument("--marginal-tol", type=float, default=1e-4)

args = parser.parse_args()

bank_dir = ROOT / "data/infoot_test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cat_bank = torch.load(
    bank_dir / "cat_bank.pt", map_location=device, weights_only=True
)
dog_bank = torch.load(
    bank_dir / "dog_bank.pt", map_location=device, weights_only=True
)

Xs = cat_bank["v_bank"].float()
Xt = dog_bank["v_bank"].float()
references = {
    name: bank_metadata(bank_dir / f"{name}_bank.pt", bank)
    for name, bank in (("cat", cat_bank), ("dog", dog_bank))
}

with torch.no_grad():
    solver = infoot.FusedInfoOT(
        Xs, Xt, h=args.h, reg=args.reg, lam=args.lam,
    )
    P = solver.solve(
        numIter=args.max_iter, sinkhorn_iter=args.sinkhorn_iter,
        marginal_tol=args.marginal_tol,
    )

infoot.save_plan(
    bank_dir / "cat_to_dog_plan.pt", P, args.h, args.reg, args.lam,
    optimization=solver.diagnostics_, banks=references,
)
print("Transport plan:", P.shape)
