"""
InfoOT solver
"""
# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License

import torch

from .transport.objective import fitting_loss, migrad, ratio
from .transport.optimization import solve as optimize_plan
from .transport.plan_io import save_plan

def dist(z1, z2, delta=5000):
    x1, x2 = z1[:-1], z2[:-1]
    y1, y2 = z1[-1], z2[-1]
    if y1 != y2:
        return torch.linalg.norm(x1 - x2) + delta
    else:
        return torch.linalg.norm(x1 - x2)

def compute_kernel(Cx, Cy, h, Cx_reference=None):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth
    Cx_reference: source reference distances for out-of-sample queries
    Returns
    ----------
    Kx: source kernel
    Ky: targer kernel
    '''
    reference = Cx if Cx_reference is None else Cx_reference
    h1 = h * torch.sqrt((reference**2).mean() / 2)
    h2 = h * torch.sqrt((Cy**2).mean() / 2)
    # Gaussian kernel (without normalization)
    Kx = torch.exp(-(Cx / h1)**2 / 2)
    Ky = torch.exp(-(Cy / h2)**2 / 2)
    return Kx, Ky

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

        # init kernel
        self.C = torch.cdist(Xs, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        if Ys is not None:
            Zs = torch.cat((Xs, Ys.reshape(-1, 1)), dim=1)
            self.Cs = torch.cdist(Zs[:, :-1], Zs[:, :-1], compute_mode='donot_use_mm_for_euclid_dist') + (Zs[:, -1, None] != Zs[None, :, -1]) * 5000
        else:
            self.Cs = torch.cdist(Xs, Xs, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None
   
    def solve(self, numIter=50, verbose=True, P0=None, **options):
        return optimize_plan(self, numIter, verbose, P0, **options)
  

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
        _Ks, _Kt = compute_kernel(_Cs, self.Ct, h, Cx_reference=self.Cs)
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

        # init kernel
        self.Cs = torch.cdist(Xs, Xs, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ct = torch.cdist(Xt, Xt, compute_mode='donot_use_mm_for_euclid_dist')
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None

    def solve(self, numIter=100, verbose=True, P0=None, **options):
        return optimize_plan(self, numIter, verbose, P0, **options)
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
        _Ks, _Kt = compute_kernel(_Cs, self.Ct, h, Cx_reference=self.Cs)
        return ratio(self.P, _Ks, _Kt)
