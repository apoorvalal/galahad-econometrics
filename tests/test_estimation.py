import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares, minimize
from scipy.special import expit

from galahad_econometrics import (
    IVGMM,
    NLS,
    OLS,
    ExponentialNLS,
    Logit,
    Poisson,
    Ridge,
    SyntheticControl,
    arc,
    tru,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(201)
    X = rng.normal(size=(600, 3))
    D = np.column_stack([np.ones(len(X)), X])
    b = np.array([0.2, 0.4, -0.3, 0.1])
    return rng, X, D, b


@pytest.mark.parametrize("solver", [tru, arc])
@pytest.mark.parametrize("penalty", [0.0, 0.07])
def test_linear_closed_form(sample, solver, penalty):
    rng, X, D, b = sample
    y = D @ b + rng.normal(size=len(X))
    model = Ridge(X, y, penalty=penalty) if penalty else OLS(X, y)
    diagonal = np.r_[0.0, np.full(X.shape[1], penalty)]
    augmented = np.vstack([D, np.diag(np.sqrt(len(y) * diagonal))])
    target = np.r_[y, np.zeros(D.shape[1])]
    expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
    fit = model.fit(solver=solver)
    assert fit.success and fit.optimization.status == 0
    assert_allclose(fit.params, expected, atol=2e-7, rtol=1e-7)
    assert_allclose(fit.predict(X[:10]), D[:10] @ expected, atol=2e-7)


@pytest.mark.parametrize("solver", [tru, arc])
@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_likelihoods_against_independent_scipy(sample, solver, family):
    rng, X, D, b = sample
    if family == "logit":
        y = rng.binomial(1, expit(D @ b))
        model = Logit(X, y)

        def fun(t):
            return np.mean(np.logaddexp(0, D @ t) - y * (D @ t))

        def grad(t):
            return D.T @ (expit(D @ t) - y) / len(y)
    else:
        offset = rng.uniform(-0.4, 0.4, len(X))
        y = rng.poisson(np.exp(D @ b + offset))
        model = Poisson(X, y, offset=offset)

        def fun(t):
            return np.mean(np.exp(D @ t + offset) - y * (D @ t + offset))

        def grad(t):
            return D.T @ (np.exp(D @ t + offset) - y) / len(y)

    ref = minimize(fun, np.zeros(D.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-10})
    fit = model.fit(solver=solver, cov_type="HC1")
    assert_allclose(fit.params, ref.x, atol=3e-7, rtol=1e-6)
    assert_allclose(fit.optimization.fun, fun(ref.x), atol=1e-12)
    assert fit.bse.shape == b.shape
    assert (fit.bse > 0).all()
    if family == "poisson":
        with pytest.raises(ValueError, match="offset"):
            fit.predict(X[:5])
        assert_allclose(
            fit.predict(X[:5], offset=offset[:5]), np.exp(D[:5] @ fit.params + offset[:5])
        )


@pytest.mark.parametrize("solver", [tru, arc])
@pytest.mark.parametrize("custom_weight", [False, True])
def test_iv_gmm_closed_form(solver, custom_weight):
    rng = np.random.default_rng(401)
    Z = rng.normal(size=(1000, 5))
    v = rng.normal(size=1000)
    X = np.column_stack([Z[:, 0] + v, Z[:, 1] + 0.4 * rng.normal(size=1000)])
    y = 0.2 + X @ np.array([1.2, -0.4]) + 0.7 * v + rng.normal(size=1000)
    D, instruments = np.column_stack([np.ones(len(y)), X]), np.column_stack([np.ones(len(y)), Z])
    weight = np.diag(np.arange(1, 7)) if custom_weight else None
    model = IVGMM(X, y, Z, weight=weight)
    W = np.linalg.inv(instruments.T @ instruments / len(y)) if weight is None else weight
    M = instruments.T @ D / len(y)
    ref = np.linalg.solve(M.T @ W @ M, M.T @ W @ (instruments.T @ y / len(y)))
    fit = model.fit(solver=solver, cov_type="HC1")
    assert_allclose(fit.params, ref, atol=2e-7)
    omega = (
        (instruments * (y - D @ ref)[:, None]).T @ (instruments * (y - D @ ref)[:, None]) / len(y)
    )
    bread = np.linalg.inv(M.T @ W @ M)
    cov = bread @ M.T @ W @ omega @ W @ M @ bread / (len(y) - D.shape[1])
    assert_allclose(fit.covariance, cov, rtol=2e-6, atol=1e-11)


@pytest.mark.parametrize("solver", [tru, arc])
def test_nls_exact_curvature_and_fit(sample, solver):
    rng, X, D, b = sample
    y = np.exp(D @ b) + 0.2 * rng.normal(size=len(X))
    callbacks = NLS(
        y,
        mean=lambda t: np.exp(D @ t),
        jac=lambda t: np.exp(D @ t)[:, None] * D,
        weighted_hess=lambda t, r: D.T @ ((r * np.exp(D @ t))[:, None] * D),
        x0=np.zeros(D.shape[1]),
    )
    built_in = ExponentialNLS(X, y)
    # Away from the optimum, require the residual-curvature term, not just J'J.
    assert_allclose(callbacks.hess(-b), built_in.hess(-b), rtol=1e-12, atol=1e-12)
    ref = least_squares(
        lambda t: np.exp(D @ t) - y,
        np.zeros(D.shape[1]),
        jac=lambda t: np.exp(D @ t)[:, None] * D,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    for model in [callbacks, built_in]:
        fit = model.fit(solver=solver, cov_type="HC1")
        assert_allclose(fit.params, ref.x, atol=3e-7)
        assert_allclose(fit.fitted_values, np.exp(D @ fit.params))


def test_synthetic_control_against_slsqp():
    rng = np.random.default_rng(902)
    X = rng.normal(size=(100, 7))
    y = X @ np.array([0.2, 0.3, 0.5, 0, 0, 0, 0]) + 0.01 * rng.normal(size=len(X))
    model = SyntheticControl(X, y)
    ref = minimize(
        lambda w: 0.5 * np.mean((X @ w - y) ** 2),
        np.ones(7) / 7,
        jac=lambda w: X.T @ (X @ w - y) / len(y),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 7,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(7)},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    # Ask for the tighter gap needed to resolve small near-boundary weights;
    # the public tolerance controls an objective certificate, not coefficient error.
    fit = model.fit(gtol=1e-9)
    assert fit.success
    assert_allclose(fit.params, ref.x, atol=1e-6)
    assert fit.optimization.feasibility < 1e-8
    assert fit.optimization.optimality < 1e-9
    assert fit.optimization.history == ()


@pytest.mark.parametrize("solver", [tru, arc])
def test_scaled_feature_units(sample, solver):
    rng, X, D, b = sample
    y = rng.poisson(np.exp(D @ b))
    units = np.array([1e-3, 1e3, 1.0])
    original = Poisson(X, y).fit(solver=solver, cov_type="HC1")
    scaled = Poisson(X * units, y).fit(solver=solver, cov_type="HC1")
    transform = np.r_[1.0, units]
    assert_allclose(scaled.params * transform, original.params, atol=2e-7)
    assert_allclose(
        scaled.covariance * np.outer(transform, transform),
        original.covariance,
        rtol=1e-6,
        atol=1e-10,
    )
