"""Original 84-instance benchmark generator, unchanged numerical body.

Source: Choosing an Optimizer for Econometric Estimation (2026-11-03).
Public report: https://lalten.org/quomodoc/docs/econometric-optimizer-comparison
This module generates synthetic data only and does not import a prior fit.
"""

import numpy as np
from scipy.special import expit

FAMILIES = ["OLS", "Ridge", "Logit", "Poisson", "IV-GMM", "NLS", "SCM"]


def make_problem(family, size, level, seed):
    group = 0 if family in ["OLS", "Ridge"] else FAMILIES.index(family)
    rng = np.random.default_rng(20261021 + 100 * group + 10 * size + seed)
    n, p = [(2000, 8), (20000, 24)][size]
    if family == "SCM":
        n, p = [(100, 12), (500, 50)][size]
        factors, loadings = rng.normal(size=(n, 3)), rng.normal(size=(3, p))
        X = factors @ loadings / np.sqrt(3) + [1, 0.1, 0.01][level] * rng.normal(size=(n, p))
        beta = np.r_[0.2, 0.3, 0.5, np.zeros(p - 3)]
        y = X @ beta + 0.05 * rng.normal(size=n)
        return dict(
            family=family, X=X, y=y, beta=beta, x0=np.full(p, 1 / p), penalty=np.zeros(p), Z=None
        )
    features = rng.normal(size=(n, p - 1))
    if family in ["OLS", "Ridge"]:
        Q, _ = np.linalg.qr(features - features.mean(axis=0))
        V, _ = np.linalg.qr(rng.normal(size=(p - 1, p - 1)))
        singular = np.geomspace(1, 1 / [1, 100, 10000][level], p - 1)
        features = np.sqrt(n) * (Q * singular) @ V.T
    X = np.column_stack([np.ones(n), features])
    beta = rng.normal(size=p)
    beta[1:] *= 0.4 / np.linalg.norm(beta[1:])
    beta[0] = 0.2
    Z = None
    if family == "Logit":
        beta[1:] *= [1, 4, 12][level] / 0.4
        y = rng.binomial(1, expit(X @ beta)).astype(float)
    elif family == "Poisson":
        scales = np.r_[1, np.geomspace(1, [1, 100, 10000][level], p - 1)]
        X *= scales
        beta /= scales
        y = rng.poisson(np.exp(X @ beta)).astype(float)
    elif family == "IV-GMM":
        Z = np.column_stack([np.ones(n), rng.normal(size=(n, p + 3))])
        v = rng.normal(size=(n, p - 1))
        strength = np.ones(p - 1)
        strength[0] = [1, 0.1, 0.01][level]
        X[:, 1:] = Z[:, 1:p] * strength + v
        y = X @ beta + 0.7 * v[:, 0] + np.sqrt(1 - 0.7**2) * rng.normal(size=n)
    elif family == "NLS":
        y = np.exp(X @ beta) + 0.1 * rng.normal(size=n)
    else:
        y = X @ beta + 0.25 * rng.normal(size=n)
    penalty = np.r_[0.0, np.full(p - 1, 0.01 if family == "Ridge" else 0.0)]
    x0 = [0, 4, -6][level] * beta if family == "NLS" else np.zeros(p)
    return dict(
        family=family, X=np.ascontiguousarray(X), y=y, beta=beta, x0=x0, penalty=penalty, Z=Z
    )
