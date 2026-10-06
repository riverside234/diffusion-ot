from infoot_helper import infoot
from infoot_helper.encoding import feature_stats, standardize
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.4)
parser.add_argument("--reg", type=float, default=0.02)
parser.add_argument("--lam", type=float, default=0.1)

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
stats = {"cat": feature_stats(Xs), "dog": feature_stats(Xt)}

with torch.no_grad():
    solver = infoot.FusedInfoOT(
        standardize(Xs, stats["cat"]),
        standardize(Xt, stats["dog"]),
        h=args.h, reg=args.reg, lam=args.lam,
    )
    P = solver.solve(numIter=30, verbose=True)

infoot.save_plan(bank_dir / "cat_to_dog_plan.pt", P, stats, args.h, args.reg)
print("Transport plan:", P.shape)
