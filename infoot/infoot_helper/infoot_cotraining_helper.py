import torch
from diffusion_ot.training.native_flow import native_flow_objective
from . import infoot


def save_domain_checkpoints(domains, output_dir, step):
    for name, domain in domains.items():
        torch.save(
            {
                "step": step,
                "domain": name,
                "model": domain.branch.pdae_state_dict(),
                "config": domain.training_config,
                "native_flow_objective": native_flow_objective(domain.training_config),
            },
            output_dir / f"{name}_step_{step:06d}.pt",
        )


@torch.no_grad()
def fit_transport(v_cat, v_dog, h=0.4, mi_weight=0.10,
                  reg=0.02, iterations=100, diagnostics=None):
    solver = infoot.FusedInfoOT(
        v_cat,
        v_dog,
        h=h,
        lam=mi_weight,
        reg=reg,
    )
    P = solver.solve(numIter=iterations, verbose=False)
    if diagnostics is not None:
        diagnostics.update(solver.diagnostics_)
    return P


def alignment_loss(
    v_cat, v_dog, P, h=0.5, mi_weight=0.10, eps=1e-8, reg=0.02
):
    solver = infoot.FusedInfoOT(
        v_cat, v_dog, h=h, lam=mi_weight, reg=reg
    )
    return infoot.fitting_loss(
        P.detach(),
        solver.Ks,
        solver.Kt,
        reg=reg,
        C=solver.C,
        mi_weight=mi_weight,
        eps=eps,
    )


def conditional_mapping(v_query, v_source, v_target, P, h=0.4):
    solver = infoot.InfoOT(v_source, v_target, h=h)
    solver.P = P.detach()

    scores = solver.conditional_score(v_query)
    return infoot.projection(scores, v_target)
