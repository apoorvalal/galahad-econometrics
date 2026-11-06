# GALAHAD Econometrics

A small Python estimation library built on GALAHAD's trust-region (TRU), adaptive
cubic-regularization (ARC), and quadratic-programming (QPB) solvers. Model criteria
and optimizer selection are separate: the same likelihood can be passed to either
TRU or ARC, or replaced with your own differentiable criterion.

This is an experimental, dense-matrix prototype for small-to-moderate parameter
dimensions. It uses the actual GALAHAD bindings; there is no SciPy optimization
fallback. NumPy and SciPy supply array operations, stable special functions and
linear algebra.

## Install

From this repository:

```sh
uv sync --python 3.12 --extra dev
# These controls avoid initialization warnings from the upstream SLBLT backend.
export OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE
uv run python examples/quickstart.py
uv run pytest
```

For another environment, `python -m pip install /path/to/galahad-econometrics`.
Installation requires a compatible `galahad-optrove` build. Python 3.12 on Apple
Silicon is the locally validated platform; other platforms depend on upstream
wheel/build availability. The wrapper never changes process-wide thread or
OpenMP settings on import. Use your normal BLAS/OpenMP thread policy.

## Worked notebooks

The [executed notebook collection](notebooks/) covers custom criteria, Rosenbrock,
likelihood comparisons, all seven estimator families, ridge paths, IV/GMM and
bootstrap inference, exact-Hessian NLS, synthetic control, and JAX multinomial
logit. Code, checks and figures live together in interactive cells.

```sh
uv sync --extra notebooks --extra autodiff
uv run --extra notebooks --extra autodiff jupyter lab notebooks/
```

JAX is optional and required only for the multinomial-logit example. The
[notebook guide](notebooks/README.md) explains running the collection in fresh
kernels and exporting HTML without repeating any fits.

## Fit a model

```python
import numpy as np
from galahad_econometrics import Logit

rng = np.random.default_rng(42)
X = rng.normal(size=(3000, 3))
beta = np.array([0.2, 0.7, -0.4, 0.3])
eta = beta[0] + X @ beta[1:]
y = rng.binomial(1, 1 / (1 + np.exp(-eta)))

model = Logit(X, y)               # Adds an intercept; X contains features only.
fit = model.fit(solver="tru", cov_type="HC1")
print(fit.summary())
fit.params                        # Intercept first, then feature coefficients.
fit.bse                           # HC1 standard errors, in original units.
fit.predict(X[:5])                 # Probabilities.
fit.optimization.info             # Unmodified native GALAHAD diagnostics.

other = model.fit(solver="arc", cov_type="HC1")
```

`fit()` defaults to TRU, absolute gradient tolerance `1e-8`, direct exact-Hessian
subproblems and regressor-RMS parameter scaling. Pass `scale=None` to retain raw
optimization coordinates. Covariance is opt-in; `cov_type=None` returns point
estimates only. Use `intercept=False` when supplying an already-complete design.
There is no implicit centering, missing-value removal or formula interface.

## Supply a criterion directly

```python
from galahad_econometrics import Criterion, arc as module

A = np.array([[3.0, 0.5], [0.5, 2.0]])
b = np.array([1.0, -2.0])
criterion = Criterion(
    fun=lambda x: float(0.5 * x @ A @ x - b @ x),
    grad=lambda x: A @ x - b,
    hess=lambda x: A,
)
result = module.solve(criterion, x0=np.zeros(2), trace=True)
result.x, result.fun, result.success
```

Import `tru as module` to change solvers. Built-in models satisfy the same
`fun/grad/hess` contract: `tru.solve(Logit(X, y))` also works. Solver calls return
`OptimizeResult`; model `.fit()` adds predictions and optional inference. A user
criterion supplies its own objective scaling and exact derivatives. No numerical
Hessian or automatic-differentiation dependency is inserted silently.

## Included estimators

| Criterion | Constructor | Solver |
|---|---|---|
| OLS | `OLS(X, y)` | TRU / ARC |
| Ridge | `Ridge(X, y, penalty=0.01)` | TRU / ARC |
| Binary logit | `Logit(X, y)` | TRU / ARC |
| Poisson PML | `Poisson(X, y, offset=log_exposure)` | TRU / ARC |
| Fixed-weight linear IV/GMM (2SLS by default) | `IVGMM(X, y, Z)` | TRU / ARC |
| Nonlinear least squares | `ExponentialNLS(X, y)` or callback-based `NLS(...)` | TRU / ARC |
| Simplex synthetic control | `SyntheticControl(donors, treated)` | QPB |

All regression criteria use average loss. Ridge adds
`0.5 * sum(penalty * beta**2)` to that average; a scalar penalty leaves the added
intercept unpenalized. Convert a sum-loss ridge parameter by dividing it by `n`.
For IV, supply all regressors in `X` and all instruments in `Z`, including
exogenous regressors among the instruments. An intercept is added to both by
default. `weight=` is a fixed positive-definite GMM weight; no second-step
estimated-weight update is performed.

Generic NLS takes fitted means, their Jacobian and a weighted Hessian callback:

```python
from galahad_econometrics import NLS

# X here is the full design, including any desired constant.
nls = NLS(
    y,
    mean=lambda b: np.exp(X @ b),
    jac=lambda b: np.exp(X @ b)[:, None] * X,
    weighted_hess=lambda b, r: X.T @ ((r * np.exp(X @ b))[:, None] * X),
    x0=np.zeros(X.shape[1]),
)
fit = nls.fit(solver="arc", cov_type="HC1")
```

The callback gives `sum_i r_i * Hessian(mean_i)`, so exact NLS curvature can be
indefinite. Neither an `n × p × p` tensor nor a Gauss–Newton approximation is
required. `ExponentialNLS` provides this model directly, including new-data
prediction. Generic callback-based NLS predicts in sample; evaluate your own
mean function for new data.

## Inference and diagnostics

`HC0`, `HC1`, and one-way cluster-robust `cluster` covariance are available for
OLS, logit, Poisson, IV/GMM and NLS. Cluster fits require
`model.fit(cov_type="cluster", clusters=labels)`. CR1 uses
`G/(G-1) * (n-1)/(n-p)` and reports normal-approximation intervals. Ridge and
synthetic control return point estimates only; asking for their covariance
raises an explicit error. See [the methods](docs/methods.md) for normalization,
GMM covariance and exact NLS curvature.

Every result preserves native status and independently recomputes objective and
first-order accuracy. A successful native stop alone is insufficient. QPB also
checks simplex feasibility and the Frank–Wolfe/KKT gap. Model fitting raises
`ConvergenceError` by default on failure; its `.result` retains diagnostics. Use
`raise_on_failure=False` to inspect a failed fit. Raw `module.solve()` returns
failed results unless asked to raise.

With `trace=True`, `result.history` records **objective evaluation points**,
including rejected trials, in elapsed wall time. These are not accepted iterates.
QPB exposes endpoint diagnostics only. Native calls are serialized because the
upstream Python modules retain global state; nested solves inside callbacks are
rejected. Resources are released on success and exceptions.

## Scope

TRU/ARC are useful when exact curvature and reliable step control matter. OLS,
ridge and linear IV also have simpler direct solutions, which are independent
references in the tests; GALAHAD is not automatically the fastest choice for
those problems. These solvers find local stationary points. They do not establish
global optimality for NLS, identification in weak-IV samples, or existence of a
finite MLE under separation. Standard errors require statistical assumptions
beyond an optimizer's stopping test.

The prototype has no fixed-effect absorption, observation weights, sparse or
Hessian-vector interface, formula parser, general constrained-NLP interface,
multiway clustering or automatic inference for regularized/constrained fits.
CrabbyMetrics is neither a dependency nor modified by this project.

[API details](docs/api.md) · [Mathematical conventions](docs/methods.md)

## Validation

The initial version passes 45 tests and 156 additional solves on all 84 original
benchmark datasets: 72 TRU, 72 ARC and 12 QPB. All original data hashes match.
Each fit passes both native stopping checks and independently calculated
objective/first-order checks. This is numerical validation of this prototype on
that suite, rather than a new runtime comparison or a guarantee for other data.
See [validation details](validation/README.md) for references and reproduction.
