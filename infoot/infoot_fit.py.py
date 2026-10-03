from infoot_helper import infoot
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

with torch.no_grad():
    solver = infoot.InfoOT(Xs, Xt, h=0.5, reg=0.05)
    P = solver.solve(numIter=100, verbose=True)

torch.save(P.cpu(), bank_dir / "cat_to_dog_plan.pt")
print("Transport plan:", P.shape)
