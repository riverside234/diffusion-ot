# Plan for finding a better InfoOT transport plan

Find a feasible transport plan `P` with a lower value of the fitting objective. Start with several feasible initial plans, optimize each, and retain the best feasible result. The implementation now provides these steps, including optional continuation. It does not guarantee a global minimum or better generated images.

Run offline fitting with `python infoot/infoot_fit.py --restarts 6 --seed 0 --max-iter 50`. Add `--continuation` for entropy continuation. Existing `--h`, `--reg`, and `--lam` arguments select the target objective. `--sinkhorn-iter` defaults to 5000 iterations; retries may use larger budgets. This argument and `--marginal-tol` control inner-solve accuracy. Each call also evaluates a full-step baseline, so six starts mean seven runs before optional continuation. Co-training defaults to two starts plus this baseline and records the selected run in its logs and checkpoints.

The helpers are split into `objective.py`, `transport_utils.py`, `optimization.py`, `multistart.py`, and `plan_io.py` under `infoot/infoot_helper/transport`. Existing `InfoOT.solve()` and `FusedInfoOT.solve()` return tensors and delegate one run to `optimization.py`; import `solve_multistart()` from `infoot_helper.transport.multistart` to compare starts. New functions live in these helper files.

The MI gradient now follows the same clamp as the scalar loss. Entropy uses the dtype's minimum normal positive value as its logarithm floor, replacing the previous `1e-8` floor that distorted tiny transport masses. Both fitting and co-training use the same revised objective. Saved metadata records bank file hashes and checkpoint contents at fitting time; this does not retroactively establish which checkpoint created an older bank.

1. **Define one fixed comparison problem.**

   For FusedInfoOT, minimize

   $$
   F(P)=\langle P,C\rangle-\lambda I(P;K_s,K_t)-\varepsilon H(P),
   \qquad P\geq0,\quad P\mathbf{1}=p,\quad P^\top\mathbf{1}=q.
   $$

   Here, `epsilon = reg`, `lambda = lam`, and `p`, `q` are the current uniform marginals. For plain InfoOT, omit the cross-domain cost and use its existing MI weight of one. Compare starts within the same solver and objective. The objective and Sinkhorn proposal follow Section 4.3 of the [InfoOT paper](https://proceedings.mlr.press/v202/chuang23a/chuang23a.pdf).

   Compute `C`, `Ks`, and `Kt` once for a fixed set and ordering of reference features. Preserve raw `v`, Euclidean distances, and the current kernel bandwidth rule. Freeze `h`, `reg`, `lam`, feature values, and numerical settings while comparing starts. If a fixed bank variance is introduced later, evaluate that change separately and reuse the same variance for every candidate.

2. **Make the objective and feasibility checks reliable.**

   Reuse `fitting_loss()` for candidate ranking and line search, and `migrad()` for the negative MI gradient. Check the manual MI gradient against autograd on small examples. Account for the current loss clamps: a clipped scalar loss and an unclipped gradient can disagree when a clamp is active. Resolve such disagreements before interpreting a failed line search as convergence.

   Require finite, nonnegative plans and measure

   $$
   r(P)=\max\left(\|P\mathbf{1}-p\|_1,\|P^\top\mathbf{1}-q\|_1\right).
   $$

   Start with `marginal_tol = 1e-4`. Check the actual residual, not only Sinkhorn's internal stopping flag. Retry an inaccurate inner solve with a larger budget or reject that proposal. Do not repair a plan with row normalization alone, which can break its column marginals.

   Record the full-step update method as the baseline, using the same feasibility checks and numerical settings as the new candidates. Its trajectory may differ numerically from an older implementation because of improved inner solves and the corrected objective clamps.

3. **Generate different feasible starting plans.**

   Begin with six starts for offline FusedInfoOT fitting:

   | Starts | Construction | Purpose |
   | --- | --- | --- |
   | 1 | `outer(p, q)` | Preserve the current initialization |
   | 2 | Balance a sharper random positive matrix to `p, q` | Ensure even the two-start training budget explores a different match |
   | 3 | Entropic OT using only `C` | Start from a geometric match |
   | 4-5 | Entropic OT using `C + tau * random_noise`, with distinct seeds and perturbation scales | Explore nearby geometric matches |
   | 6 | Balance a more diffuse random positive matrix to `p, q` | Explore another match |

   Use log-Sinkhorn for initialization and verify the marginals. Random positive matrices can be represented by random log weights and balanced without explicitly exponentiating large values. These constructions support unequal source and target counts. For plain InfoOT, replace cost-based starts with additional random feasible starts.

   Perturbed costs are used only to construct initial plans. Every optimization run then uses the original objective. Record seeds and perturbation scales. Changing a seed while still initializing with `outer(p, q)` does not create a new starting point.

4. **Improve each plan with controlled Sinkhorn updates.**

   Add an optional `P0` argument to `solve()`. Validate and clone it; keep the current uniform initialization when `P0` is omitted.

   At each iteration, form the existing gradient cost:

   $$
   G=C+\lambda\,\operatorname{migrad}(P,K_s,K_t).
   $$

   For plain InfoOT, use its existing `migrad` cost. Solve the entropic subproblem with log-Sinkhorn to obtain a feasible proposal `Q`. Entropy is already handled by that subproblem; do not add its gradient to `G` as well.

   Backtrack over `alpha = 1, 1/2, 1/4, ...` and evaluate

   $$
   P_{\mathrm{candidate}}=(1-\alpha)P+\alpha Q.
   $$

   Accept a finite candidate that decreases the full objective beyond numerical noise and satisfies the marginal tolerance. Convex combinations preserve feasibility when both endpoints are feasible. Keep the best feasible iterate encountered during the run, including the initial plan.

   Check the undamped proposal residual `||Q - P||_1` before updating `P`; a tiny accepted step alone does not establish stationarity. Use a starting residual tolerance of `1e-5` and require three consecutive successful residual checks. If line search fails while the proposal residual remains large, tighten the inner solve, then report a stall if it still fails. Report iteration limits separately from convergence.

   Reuse Sinkhorn dual warm starts within a run when supported. Reset them between independent restarts. The proposed damping and restart strategy extends the existing solver; it is not a guarantee of global optimality. POT describes the related [generalized conditional-gradient approach](https://pythonot.github.io/quickstart.html#generic-solvers).

5. **Select the best feasible result and compare compute fairly.**

   Run starts sequentially and reuse the distance and kernel matrices. Keep the current and best plans instead of retaining every dense plan on the GPU. Select the lowest full objective among all eligible results, including the baseline.

   Log the initialization, seed, total objective, transport cost, MI, entropy, marginal residual, proposal residual, accepted step sizes, iterations, stopping reason, and elapsed time. When final losses are close, refine the strongest candidates with tighter inner tolerances and, where practical, float64 before claiming improvement.

   Begin with a fixed subset of roughly 512-1024 features per domain to measure runtime and memory, then repeat on the intended bank. Subset and full-bank objectives are different problems and must not be ranked together. Compare the restart strategy with one longer baseline run under approximately equal total compute.

   For co-training, begin with two starts per reference batch. Hold the current features fixed during fitting and keep the selected plan detached. Recompute the differentiable alignment loss from live features afterward. Do not reuse a previous batch's plan as an initialization unless its rows and columns refer to the same samples. Compare candidates within each batch, not across batches.

6. **Add entropy continuation as a second experiment.**

   If different initializations repeatedly reach similar outcomes, add a candidate that follows `reg = 5 * target_reg -> 2 * target_reg -> target_reg`. For a target of `0.02`, this is `0.10 -> 0.04 -> 0.02`.

   Initialize each stage with the preceding stage's feasible plan. Keep `C`, kernels, and the MI weight fixed. Reset dual warm starts when changing `reg` unless their rescaling is explicitly implemented. Finish with optimization at the target `reg` and rank only the final target-objective value against the other candidates. This is an additional search heuristic; intermediate losses from different regularization strengths are not comparable.

7. **Keep implementation small and preserve the existing pipeline.**

   | Location | Planned responsibility |
   | --- | --- |
   | [infoot.py](<C:/Users/BobXu/Desktop/diffusion research code/infoot/infoot_helper/infoot.py>) | Extend `solve()` with `P0`, feasibility checks, controlled updates, and diagnostics; retain its tensor return value |
   | `objective.py` | Shared loss, density ratio, and consistent MI gradient |
   | `transport_utils.py` | Feasibility checks, log-Sinkhorn with retries, and backtracking |
   | `optimization.py` | One fitting run and its diagnostics |
   | `multistart.py` | Seeded initial plans, best-run selection, and optional continuation |
   | `plan_io.py` | Compatible plan saving and reference file identities |
   | [infoot_fit.py](<C:/Users/BobXu/Desktop/diffusion research code/infoot/infoot_fit.py>) | Add restart count, seed, and iteration-budget arguments; call the wrapper and save its result |
   | [infoot_cotraining_helper.py](<C:/Users/BobXu/Desktop/diffusion research code/infoot/infoot_helper/infoot_cotraining_helper.py>) | Let `fit_transport()` use the same wrapper with a smaller restart budget |
   | [test_local_infoot_pipeline.py](<C:/Users/BobXu/Desktop/diffusion research code/tests/test_local_infoot_pipeline.py>) | Add focused checks for initialization, feasibility, objective selection, gradients, and save/load consistency |

   Keep the current saved-plan keys so existing readers remain compatible. Add solver type, seed, selected initialization, actual objective parameters, and diagnostics. Record the reference-bank identities and ordering plus the model checkpoint identity; a mutable `latest.pt` pathname alone cannot establish compatibility. Save effective kernel scales if bank-derived fixed scales are adopted.

8. **Define acceptance criteria before running the experiment.**

   - Initial and returned plans satisfy the same marginal tolerance, including tests with unequal domain sizes.
   - Accepted local updates decrease the same scalar objective within the chosen numerical tolerance.
   - The selected result is no worse than the eligible baseline under the same objective and feasibility checks.
   - A claimed improvement exceeds measured numerical variation and reproduces with recorded seeds.
   - Small examples verify that objective and gradient stabilization agree; entropy is counted once.
   - Saving and loading preserve the plan, its metadata, and conditional mapping.
   - Stalls, invalid inner solves, and iteration limits are visible in diagnostics.

   Evaluate generated images separately using fixed validation cats, model checkpoints, sampling settings, and noise. Also record mapped-feature spread and conditional-weight concentration. A lower fitting objective does not by itself establish better translation or resolve repeated-image collapse.

   Continuation is implemented but disabled by default. Establish a reproducible comparison with ordinary restarts before enabling it on the lab banks.
