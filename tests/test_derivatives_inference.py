import numpy as np
import pytest
from numpy.testing import assert_allclose

from galahad_econometrics import (
    IVGMM,
    OLS,
    ExponentialNLS,
    Logit,
    Poisson,
    Ridge,
    check_derivatives,
)
from galahad_econometrics.inference import covariance


@pytest.mark.parametrize("kind", ["ols", "ridge", "logit", "poisson", "iv", "nls"])
def test_exact_derivatives(kind):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)
    models = dict(
        ols=lambda: OLS(X, y),
        ridge=lambda: Ridge(X, y, penalty=[0, 0.1, 0.2, 0.3]),
        logit=lambda: Logit(X, (y > 0).astype(float)),
        poisson=lambda: Poisson(X, np.abs(y)),
        iv=lambda: IVGMM(X, y, np.column_stack([X, rng.normal(size=80)])),
        nls=lambda: ExponentialNLS(X, y),
    )
    errors = check_derivatives(models[kind](), np.array([0.2, -0.1, 0.1, -0.2]))
    assert errors["gradient_error"] < 1e-8
    assert errors["hessian_error"] < 1e-8


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "cluster"])
def test_ols_sandwich_against_direct_formula(cov_type):
    rng = np.random.default_rng(101)
    X = rng.normal(size=(240, 4))
    D = np.column_stack([np.ones(len(X)), X])
    labels = np.repeat(np.arange(30), 8)
    y = 0.3 + X @ rng.normal(size=4) + rng.normal(size=30)[labels] + rng.normal(size=240)
    fit = OLS(X, y).fit(cov_type=cov_type, clusters=labels if cov_type == "cluster" else None)
    ref = np.linalg.lstsq(D, y, rcond=None)[0]
    scores = D * (y - D @ ref)[:, None]
    if cov_type == "cluster":
        S = np.array([scores[labels == i].sum(axis=0) for i in range(30)])
        meat = S.T @ S * 30 / 29 * 239 / 235
        assert fit.cov_info["n_clusters"] == 30
    else:
        meat = scores.T @ scores * (240 / 235 if cov_type == "HC1" else 1)
    bread = np.linalg.inv(D.T @ D)
    assert_allclose(fit.covariance, bread @ meat @ bread, atol=1e-10, rtol=1e-6)
    assert fit.conf_int().shape == (5, 2)
    assert "std. error" in fit.summary()


def test_observation_replication_covariance_scaling():
    rng = np.random.default_rng(811)
    X, y = rng.normal(size=(200, 2)), rng.normal(size=200)
    a = OLS(X, y).fit(cov_type="HC0")
    b = OLS(np.tile(X, (3, 1)), np.tile(y, 3)).fit(cov_type="HC0")
    assert_allclose(a.params, b.params, atol=1e-8)
    assert_allclose(a.covariance / 3, b.covariance, rtol=1e-6, atol=1e-10)


def test_individual_gradient_normalization():
    rng = np.random.default_rng(49)
    X, y = rng.normal(size=(120, 2)), rng.normal(size=120)
    b = np.array([0.2, -0.1, 0.4])
    for model in [
        OLS(X, y),
        Logit(X, (y > 0).astype(float)),
        Poisson(X, abs(y)),
        ExponentialNLS(X, y),
        IVGMM(X, y, X),
    ]:
        assert_allclose(model.scores(b).mean(axis=0), model.grad(b), atol=1e-12)


def test_invalid_inference():
    model = OLS(np.ones((8, 2)), np.arange(8))
    from galahad_econometrics import InferenceError

    with pytest.raises(InferenceError, match="Singular"):
        covariance(model, np.zeros(3))
    identified = OLS(np.arange(8)[:, None], np.arange(8))
    for labels in [np.zeros(8), np.array([None] * 8), np.array([float("nan")] * 8)]:
        with pytest.raises((InferenceError, ValueError)):
            covariance(identified, np.zeros(2), "cluster", labels)
