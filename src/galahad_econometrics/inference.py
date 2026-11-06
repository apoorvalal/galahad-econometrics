"""Explicit mean-criterion sandwich covariance; no silent generalized inverses."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .criteria import matrix, symmetric
from .results import InferenceError


def covariance(model, params, cov_type="HC1", clusters=None):
    """HC0, HC1 or one-way CR1, using original parameter units.

    H is the average criterion Hessian; rows of S are individual gradients.
    Cov = H^-1 (S'S) H^-1 / n^2. IVGMM uses D'Wz_i*u_i as each row,
    giving the fixed-weight GMM sandwich. Cluster covariance replaces S'S
    with the outer products of within-cluster gradient sums.
    """
    if not model.supports_inference:
        raise InferenceError(f"Covariance estimation is not implemented for {model.name}")
    if cov_type not in {"HC0", "HC1", "cluster"}:
        raise ValueError("cov_type must be 'HC0', 'HC1' or 'cluster'")
    if clusters is not None and cov_type != "cluster":
        raise ValueError("clusters requires cov_type='cluster'")
    n, p = model.nobs, model.n_params
    scores = matrix(model.scores(params), "individual gradients", (n, p))
    H = symmetric(model.hess(params), "inference Hessian", p)
    if np.linalg.matrix_rank(H) < p:
        raise InferenceError("Singular inference Hessian; no pseudoinverse is substituted")
    try:
        inverse = cho_solve(cho_factor(H), np.eye(p))
    except np.linalg.LinAlgError as exc:
        raise InferenceError("Inference requires positive definite sample curvature") from exc
    info = {
        "nobs": n,
        "n_params": p,
        "correction": 1.0,
        "bread": "exact mean-criterion Hessian",
        "reference": "normal",
    }
    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters is required for cov_type='cluster'")
        labels = np.asarray(clusters)
        if labels.ndim != 1 or labels.size != n:
            raise ValueError("clusters must have one label per observation")
        if any(
            v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
            for v in labels
        ):
            raise ValueError("Cluster labels must not be missing or nonfinite")
        try:
            _, codes = np.unique(labels, return_inverse=True)
        except TypeError as exc:
            raise ValueError("Cluster labels must be mutually comparable") from exc
        G = int(codes.max()) + 1
        if G < 2 or n <= p:
            raise InferenceError("CR1 requires at least two clusters and n > parameters")
        summed = np.zeros((G, p))
        np.add.at(summed, codes, scores)
        meat = summed.T @ summed
        info.update(n_clusters=G, correction=G / (G - 1) * (n - 1) / (n - p))
    else:
        meat = scores.T @ scores
        if cov_type == "HC1":
            if n <= p:
                raise InferenceError("HC1 requires n > parameters")
            info["correction"] = n / (n - p)
    cov = inverse @ meat @ inverse.T * (info["correction"] / n**2)
    return (cov + cov.T) / 2, info
