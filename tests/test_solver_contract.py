import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose

from galahad_econometrics import (
    OLS,
    ConvergenceError,
    Criterion,
    InferenceError,
    Logit,
    Poisson,
    Ridge,
    SyntheticControl,
    arc,
    qpb,
    tru,
)


@pytest.fixture
def criterion():
    H = np.array([[4.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, -2.0])
    return Criterion(
        fun=lambda x: float(0.5 * x @ H @ x + g @ x), grad=lambda x: H @ x + g, hess=lambda x: H
    )


@pytest.mark.parametrize("solver", [tru, arc])
def test_criterion_api_and_input_preservation(criterion, solver):
    x0 = np.array([2.0, -4.0])
    saved = x0.copy()
    result = solver.solve(criterion, x0, trace=True)
    assert result.success
    assert_allclose(
        result.x, -np.linalg.solve(criterion.hess(x0), criterion.grad(np.zeros(2))), atol=1e-8
    )
    assert_allclose(x0, saved)
    assert result.nfev == result.njev == len(result.history)
    assert result.nhev > 0
    assert result.history[0].evaluation == 1
    assert result.info["status"] == 0


@pytest.mark.parametrize("solver", [tru, arc])
def test_iteration_limit_not_success(solver):
    c = Criterion(
        fun=lambda x: float((x[0] - 2) ** 4),
        grad=lambda x: np.array([4 * (x[0] - 2) ** 3]),
        hess=lambda x: np.array([[12 * (x[0] - 2) ** 2]]),
    )
    result = solver.solve(c, [8.0], maxiter=1)
    assert not result.success
    assert result.status == -18
    assert np.isfinite(result.fun)
    with pytest.raises(ConvergenceError) as caught:
        solver.solve(c, [8.0], maxiter=1, raise_on_failure=True)
    assert caught.value.result.status == -18


@pytest.mark.parametrize("solver", [tru, arc])
def test_callback_error_and_cleanup(criterion, solver):
    def objective(x):
        if np.linalg.norm(x) > 0:
            raise ArithmeticError("intentional trial callback failure")
        return criterion.fun(x)

    broken = Criterion(objective, criterion.grad, criterion.hess)
    with pytest.raises(ArithmeticError, match="intentional trial"):
        solver.solve(broken, [0.0, 0.0])
    # A second solve must be unaffected by the failed native session.
    assert solver.solve(criterion, [0.0, 0.0]).success


@pytest.mark.parametrize("solver", [tru, arc])
def test_callback_references_released(solver):
    c = Criterion(lambda x: float(x @ x), lambda x: 2 * x, lambda x: 2 * np.eye(len(x)))
    reference = weakref.ref(c)
    solver.solve(c, [1.0, 2.0])
    del c
    gc.collect()
    assert reference() is None


def test_threads_are_serialized(criterion):
    with ThreadPoolExecutor(max_workers=3) as pool:
        fits = list(
            pool.map(
                lambda start: tru.solve(criterion, start), [[0.0, 0.0], [2.0, 4.0], [-3.0, 9.0]]
            )
        )
    assert all(result.success for result in fits)
    assert_allclose(fits[0].x, fits[1].x, atol=1e-8)


def test_nested_native_solve_rejected(criterion):
    count = [0]

    def objective(x):
        count[0] += 1
        if count[0] > 1:
            tru.solve(criterion, [0.0, 0.0])
        return criterion.fun(x)

    with pytest.raises(RuntimeError, match="Nested"):
        tru.solve(Criterion(objective, criterion.grad, criterion.hess), [0.0, 0.0])
    assert tru.solve(criterion, [0.0, 0.0]).success


def test_validation_and_wrong_solver(criterion):
    for kwargs in [
        dict(gtol=0),
        dict(scale=[0, 1]),
        dict(maxiter=1.5),
        dict(maxiter=True),
        dict(scale="oops"),
        dict(options={"misspelled_option": 3}),
    ]:
        with pytest.raises(ValueError):
            tru.solve(criterion, [0.0, 0.0], **kwargs)
    with pytest.raises(ValueError, match="x0"):
        tru.solve(criterion)
    scm = SyntheticControl(np.ones((10, 2)), np.ones(10))
    with pytest.raises(ValueError, match="constrained"):
        tru.solve(scm)
    with pytest.raises(ValueError, match="SyntheticControl"):
        qpb.solve(criterion)
    assert tru.solve(criterion, [0.0, 0.0]).success


def test_nonfinite_and_wrong_derivatives(criterion):
    wrong = [
        Criterion(lambda x: float("inf"), criterion.grad, criterion.hess),
        Criterion(criterion.fun, lambda x: np.ones(3), criterion.hess),
        Criterion(criterion.fun, criterion.grad, lambda x: np.array([[1.0, 2.0], [0.0, 1.0]])),
    ]
    for c in wrong:
        with pytest.raises((FloatingPointError, ValueError)):
            arc.solve(c, [0.0, 0.0])
    assert arc.solve(criterion, [0.0, 0.0]).success


def test_data_and_inference_errors():
    X, y = np.arange(20).reshape(10, 2), np.ones(10)
    for ctor, response in [(Logit, -y), (Poisson, -y), (OLS, np.full(10, np.nan))]:
        with pytest.raises(ValueError):
            ctor(X, response)
    with pytest.raises(ValueError):
        Ridge(X, y, penalty=-1)
    with pytest.raises(InferenceError):
        Ridge(X, y).fit(cov_type="HC1")
    with pytest.raises(InferenceError):
        SyntheticControl(X, y).fit(cov_type="HC1")


def test_logit_stability():
    model = Logit(np.array([[-1000.0], [1000.0]]), np.array([0.0, 1.0]), intercept=False)
    assert np.isfinite(model.fun(np.ones(1)))
    assert np.isfinite(model.grad(np.ones(1))).all()
    assert np.isfinite(model.hess(np.ones(1))).all()


@pytest.mark.parametrize("solver", [tru, arc])
def test_out_of_domain_trial_is_rejected(solver):
    outside = []

    def objective(x):
        if not 0 < x[0] < 1.1:
            outside.append(x.copy())
            return float("inf")
        return float((x[0] ** 2 - 1) ** 2)

    criterion = Criterion(
        objective,
        lambda x: np.array([4 * x[0] * (x[0] ** 2 - 1)]),
        lambda x: np.array([[12 * x[0] ** 2 - 4]]),
    )
    result = solver.solve(criterion, [0.1], trace=True)
    assert result.success
    assert_allclose(result.x, [1.0], atol=1e-8)
    assert outside
    assert any(np.isposinf(p.objective) for p in result.history)
