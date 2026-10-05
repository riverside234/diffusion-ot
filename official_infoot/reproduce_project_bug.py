import numpy as np
from infoot import InfoOT, FusedInfoOT


source = np.array([[0.0], [1.0]])
target = np.array([[2.0], [3.0]])
query = np.array([[0.5]])  # A new point, not in source.

for solver_type in (InfoOT, FusedInfoOT):
    solver = solver_type(source, target, h=0.5, reg=1.0)
    solver.solve(numIter=5, verbose=False)

    print(f"\n{solver_type.__name__}")
    print("Fitted points:", solver.project(source, method="conditional").ravel())
    try:
        print("New point:", solver.project(query, method="conditional"))
    except NameError as error:
        print(f"New point: {type(error).__name__}: {error}")

print("\nWhy: the new-point branch uses undefined Xs and Xt instead of self.Xs and self.Xt.")
print("That branch also reads local P before assigning it, instead of using self.P.")
