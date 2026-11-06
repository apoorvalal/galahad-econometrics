"""Fit all seven benchmark families; run from any directory after installation."""

import numpy as np
from scipy.special import expit

from galahad_econometrics import (
    IVGMM,
    OLS,
    Criterion,
    ExponentialNLS,
    Logit,
    Poisson,
    Ridge,
    SyntheticControl,
    arc,
)


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(1500, 3))
    D = np.column_stack([np.ones(len(X)), X])
    beta = np.array([0.2, 0.4, -0.3, 0.1])
    eta = D @ beta
    y_linear = eta + 0.3 * rng.normal(size=len(X))
    models = [
        OLS(X, y_linear),
        Ridge(X, y_linear),
        Logit(X, rng.binomial(1, expit(eta))),
        Poisson(X, rng.poisson(np.exp(eta))),
        IVGMM(X, y_linear, np.column_stack([X, rng.normal(size=len(X))])),
        ExponentialNLS(X, np.exp(eta) + 0.1 * rng.normal(size=len(X))),
    ]
    for model in models:
        for solver in ["tru", "arc"]:
            cov_type = "HC1" if model.supports_inference else None
            fit = model.fit(solver=solver, cov_type=cov_type)
            assert fit.success
            print(f"{model.name:37} {solver:4}  {fit.params.round(5)}")

    donors = rng.normal(size=(100, 6))
    treated = donors @ np.array([0.2, 0.3, 0.5, 0, 0, 0]) + 0.01 * rng.normal(size=100)
    scm = SyntheticControl(donors, treated).fit()
    print(f"Synthetic control                     qpb   {scm.params.round(5)}")
    print(f"Simplex gap: {scm.optimization.optimality:.2e}")

    criterion = Criterion(
        fun=lambda b: float(np.dot(b - 2, b - 2)),
        grad=lambda b: 2 * (b - 2),
        hess=lambda b: 2 * np.eye(len(b)),
    )
    custom = arc.solve(criterion, x0=np.zeros(3), trace=True)
    assert custom.success
    print(f"Custom criterion                      arc   {custom.x.round(5)}")


if __name__ == "__main__":
    main()
