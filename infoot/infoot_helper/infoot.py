"""
InfoOT solver
"""
# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License

import torch
import scipy.io
import ot
from tqdm import tqdm


def dist(z1, z2, delta=5000):
    x1, x2 = z1[:-1], z2[:-1]
    y1, y2 = z1[-1], z2[-1]
    if y1 != y2:
        return torch.linalg.norm(x1 - x2) + delta
    else:
        return torch.linalg.norm(x1 - x2)

def ratio(P, Kx, Ky):
    '''
    compute the ratio berween joint and marginal densities
    Parameters
    ----------
    P : transportation plan
    Kx: source kernel matrix
    Ky: target kernel matrix

    Returns
    ----------
    ratio matrix for (x_i, y_j)
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = torch.outer(f_x, f_y)
    constC = Kx.new_zeros((len(Kx), len(Ky)))
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    return f_xy / f_x_f_y

def fitting_loss(P, Kx, Ky, reg, C=None, mi_weight=1.0, eps=1e-8):
    mi = (P * ratio(P, Kx, Ky).clamp_min(eps).log()).sum()
    entropy = -(P * P.clamp_min(eps).log()).sum()
    cost = 0.0 if C is None else (P * C).sum()
    return cost - mi_weight * mi - reg * entropy

def save_plan(path, P, stats, h, reg):
    torch.save({
        "P": P.detach().cpu(),
        "stats": {
            name: tuple(value.detach().cpu() for value in values)
            for name, values in stats.items()
        },
        "h": h,
        "reg": reg,
    }, path)


def compute_kernel(Cx, Cy, h, feature_dims):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth
    feature_dims: source and target feature dimensions
    Returns
    ----------
    Kx: source kernel
    Ky: targer kernel
    '''
    h1 = h * feature_dims[0] ** 0.5
    h2 = h * feature_dims[1] ** 0.5
    # Gaussian kernel (without normalization)
    Kx = torch.exp(-(Cx / h1)**2 / 2)
    Ky = torch.exp(-(Cy / h2)**2 / 2)
    return Kx, Ky

def migrad(P, Kx, Ky):
    '''
    compute the gradient w.r.t. KDE mutual information
    Parameters
    ----------
    P : transportation plan
    Ks: source kernel matrix
    Kt: target kernel matrix

    Returns
    ----------
    negative gradient w.r.t. MI
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = torch.outer(f_x, f_y)
    constC = Kx.new_zeros((len(Kx), len(Ky)))
    # there's a negative sign in ot.gromov.tensor_product
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    P_f_xy = P / f_xy
    P_grad = -ot.gromov.tensor_product(constC, Kx, Ky, P_f_xy)
    P_grad = torch.log(f_xy / f_x_f_y) + P_grad
    return -P_grad

def projection(P, X):
    '''
    compute the projection based on similarity matrix
    Parameters
    ----------
    P : transportation plan or similarity matrix
    X : target data

    Returns
    ----------
    projected source data
    '''
    weights = torch.sum(P, dim = 1)
    X_proj = torch.matmul(P, X) / weights[:, None]
    return X_proj

class FusedInfoOT():
    '''
    Solver for Fused InfoOT
    Parameters
    ----------
    Xs: source data
    Xt: target data 
    h : bandwidth
    Ys: source label
    lam: weight for mutual information
    reg: weight for entropic regularization
    '''
    def __init__(self, Xs, Xt, h, Ys=None, lam=100., reg=1.0):
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.h = h
        self.lam = lam
        self.reg = reg
        self.feature_dims = (Xs.shape[1], Xt.shape[1])

        # init kernel
        self.C = torch.cdist(Xs, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        if Ys is not None:
            Zs = torch.cat((Xs, Ys.reshape(-1, 1)), dim=1)
            self.Cs = torch.cdist(Zs[:, :-1], Zs[:, :-1], compute_mode='donot_use_mm_for_euclid_dist') + (Zs[:, -1, None] != Zs[None, :, -1]) * 5000
        else:
            self.Cs = torch.cdist(Xs, Xs, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h, self.feature_dims)
        self.P = None

    def solve(self, numIter=50, verbose='True'):
        '''
        solve projected gradient descent via sinkhorn iteration
        '''
        p = self.Xs.new_zeros(len(self.Xs)) + 1. / len(self.Xs)
        q = self.Xt.new_zeros(len(self.Xt)) + 1. / len(self.Xt)
        P = torch.outer(p, q)
        if verbose:
            print('solve projected gradient descent...')
            for i in tqdm(range(numIter)):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, self.C + self.lam * grad_P,
                                       reg=self.reg, method='sinkhorn_log', stopThr=1e-4)
            loss = fitting_loss(
                    P, self.Ks, self.Kt, self.reg,
                    C=self.C, mi_weight=self.lam,
                )
            tqdm.write(f"Iteration {i + 1}: loss={loss.item():.6f}")

        else:
            for i in range(numIter):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, self.C + self.lam * grad_P,
                                       reg=self.reg, method='sinkhorn_log', stopThr=1e-4)
        self.P = P
        return P
    """
    def project(self, X, method='barycentric', h=None):
        if method not in ['conditional', 'barycentric']:
            raise Exception('only suppot conditional or barycebtric projection')
        if self.P is None:
            raise Exception('please run FusedInfoOT.solve() to obtain transportation plan')

        if h is None:
            h = self.h

        if torch.equal(X, self.Xs):
            if method == 'conditional':
                if h == self.h:
                    P = ratio(self.P, self.Ks, self.Kt)
                else:
                    _Ks, _Kt = compute_kernel(self.Cs, self.Ct, h)
                    P = ratio(self.P, _Ks, _Kt)
            else:
                P = self.P
            return projection(P, self.Xt)
        else:
            if method == 'conditional':
                _Cs = torch.cdist(X, Xs, compute_mode='donot_use_mm_for_euclid_dist')
                _Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
                _Ks, _Kt = compute_kernel(_Cs, _Ct, h)

                P = ratio(P, _Ks, _Kt)
                return projection(P, self.Xt)
            else:
                raise Exception('barycentric cannot generalize to new samples')
"""
    def conditional_score(self, X, h=None):
        if h is None:
            h = self.h
        _Cs = torch.cdist(X, self.Xs, compute_mode='donot_use_mm_for_euclid_dist')
        _Ks, _Kt = compute_kernel(_Cs, self.Ct, h, self.feature_dims)
        return ratio(self.P, _Ks, _Kt)


class InfoOT():
    '''
    Solver for InfoOT. Source and target can have different dimension.
    Parameters
    ----------
    Xs: source data
    Xt: target data
    h : bandwidth
    reg: weight for entropic regularization
    '''
    def __init__(self, Xs, Xt, h, reg=0.05):
        self.Xs = Xs
        self.Xt = Xt
        self.h = h
        self.reg = reg
        self.feature_dims = (Xs.shape[1], Xt.shape[1])

        # init kernel
        self.Cs = torch.cdist(Xs, Xs, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h, self.feature_dims)
        self.P = None

    def solve(self, numIter=100, verbose='True'):
        '''
        solve projected gradient descent via sinkhorn iteration
        '''
        p = self.Xs.new_zeros(len(self.Xs)) + 1. / len(self.Xs)
        q = self.Xt.new_zeros(len(self.Xt)) + 1. / len(self.Xt)
        P = torch.outer(p, q)
        if verbose:
            print('solve projected gradient descent...')
            for i in tqdm(range(numIter)):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, grad_P, reg=self.reg,
                                       method='sinkhorn_log', numItermax=3000, stopThr=1e-4)
        else:
            for i in range(numIter):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, grad_P, reg=self.reg,
                                       method='sinkhorn_log', numItermax=3000, stopThr=1e-4)
        self.P = P
        return P
    """
    def project(self, X, method='barycentric', h=None):
        if method not in ['conditional', 'barycentric']:
            raise Exception('only suppot conditional or barycebtric projection')
        if self.P is None:
            raise Exception('please run InfoOT.solve() to obtain transportation plan')

        if h is None:
            h = self.h

        if torch.equal(X, self.Xs):
            if method == 'conditional':
                if h == self.h:
                    P = ratio(self.P, self.Ks, self.Kt)
                else:
                    _Ks, _Kt = compute_kernel(self.Cs, self.Ct, h)
                    P = ratio(self.P, _Ks, _Kt)
            else:
                P = self.P
            return projection(P, self.Xt)
        else:
            if method == 'conditional':
                _Cs = torch.cdist(X, Xs, compute_mode='donot_use_mm_for_euclid_dist')
                _Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
                _Ks, _Kt = compute_kernel(_Cs, _Ct, h)

                P = ratio(P, _Ks, _Kt)
                return projection(P, self.Xt)
            else:
                raise Exception('barycentric cannot generalize to new samples')
"""

    def conditional_score(self, X, h=None):
        if h is None:
            h = self.h
        _Cs = torch.cdist(X, self.Xs, compute_mode='donot_use_mm_for_euclid_dist')
        _Ks, _Kt = compute_kernel(_Cs, self.Ct, h, self.feature_dims)
        return ratio(self.P, _Ks, _Kt)
