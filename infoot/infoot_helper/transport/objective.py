import torch


def _densities(P, Kx, Ky):
    joint = Kx @ P @ Ky.T
    marginal = torch.outer(Kx.mean(1), Ky.mean(1))
    return joint, marginal.clamp_min(torch.finfo(P.dtype).tiny)


def ratio(P, Kx, Ky):
    joint, marginal = _densities(P, Kx, Ky)
    return joint / marginal


def loss_terms(P, Kx, Ky, C=None, eps=1e-8):
    mi = (P * ratio(P, Kx, Ky).clamp_min(eps).log()).sum()
    entropy = -(P * P.clamp_min(torch.finfo(P.dtype).tiny).log()).sum()
    cost = P.new_zeros(()) if C is None else (P * C).sum()
    return cost, mi, entropy


def fitting_loss(P, Kx, Ky, reg, C=None, mi_weight=1.0, eps=1e-8):
    cost, mi, entropy = loss_terms(P, Kx, Ky, C, eps)
    return cost - mi_weight * mi - reg * entropy


def migrad(P, Kx, Ky, eps=1e-8):
    joint, marginal = _densities(P, Kx, Ky)
    density = joint / marginal
    active = density >= eps
    weights = torch.where(active, P, 0) / torch.where(active, joint, 1)
    return -density.clamp_min(eps).log() - Kx.T @ weights @ Ky
