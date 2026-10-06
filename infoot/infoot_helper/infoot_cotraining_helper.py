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
def fit_transport(m_cat, m_dog, h=0.4, mi_weight=0.10,
                  reg=0.02, iterations=100):
    solver = infoot.FusedInfoOT(
        m_cat,
        m_dog,
        h=h,
        lam=mi_weight,
        reg=reg,
    )
    return solver.solve(numIter=iterations, verbose=False)


def alignment_loss(
    m_cat, m_dog, P, h=0.5, mi_weight=0.10, eps=1e-5, reg=0.02
):
    solver = infoot.FusedInfoOT(
        m_cat, m_dog, h=h, lam=mi_weight, reg=reg
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


def conditional_mapping(m_query, m_source, m_target, P, v_target, h=0.4):
    solver = infoot.InfoOT(m_source, m_target, h=h)
    solver.P = P.detach()

    scores = solver.conditional_score(m_query)
    return infoot.projection(scores, v_target)
