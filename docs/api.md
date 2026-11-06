# API reference

## Solvers and criteria

```python
from galahad_econometrics import Criterion, tru, arc, qpb

criterion = Criterion(fun=objective, grad=gradient, hess=hessian)
result = tru.solve(criterion, x0, gtol=1e-8, maxiter=1000,
                   scale=None, trace=False, options=None,
                   raise_on_failure=False)
```

`arc.solve` has the identical signature. A built-in model can replace `criterion`;
then `x0` defaults to the model's zero start. Synthetic control's start is uniform.
An arbitrary object providing `fun`, `grad` and `hess` works too. Callbacks must be
deterministic and correctly shaped throughout the evaluated domain. The start
must have finite objective and derivatives. A positive-infinite trial objective
marks an out-of-domain point for the native solver to reject; no derivative is
supplied at that point. NaN, negative infinity, invalid derivatives or a callback
exception abort the solve. Trial points are never silently clipped.

`scale` is a positive vector defining optimization coordinates `z = scale * x`.
Gradient and Hessian callbacks in these coordinates are `g / scale` and
`H / outer(scale, scale)`. Returned coefficients, gradients and covariances are
always in original units. `scale="auto"` uses a model's regressor-RMS scale;
generic NLS uses ones. `optimality` is the Euclidean norm of the final gradient
in optimization coordinates. Thus `gtol` has units determined by both the
objective normalization and this explicit scaling.

`options` accepts existing GALAHAD controls such as
`{"initial_radius": 0.5}` for TRU or `{"clock_time_limit": 10.0}`.
Nested dictionaries are merged and unknown keys raise `ValueError`.
The convenience arguments control `maxit` and the absolute gradient tolerance.
Exact Hessians (`model=2`), Euclidean optimization-space norm, zero relative
gradient tolerance, and direct subproblem solves are fixed by this prototype.
Native controls cannot override those invariants. `print_level=0` and `error=-1`
are defaults; explicitly supply printing controls if desired. Python native
warnings are retained in `result.warnings`.

`qpb.solve(model, x0=None, gtol=1e-8, maxiter=1000, options=None,
raise_on_failure=False)` currently accepts only `SyntheticControl`. There is no
`scale` or trace option: rescaling simplex weights changes the constraint itself,
and the binding has no iterate callback. QPB's endpoint is checked using the
unclipped weights, simplex feasibility and the Frank–Wolfe/KKT gap.

`check_derivatives(criterion, x, step=1e-5)` returns centered-difference
`gradient_error` and `hessian_error`. Use it to audit a user criterion. It neither
changes the supplied derivatives nor runs inside the optimization loop.

## Models

```python
OLS(X, y, intercept=True)
Ridge(X, y, penalty=0.01, intercept=True)
Logit(X, y, intercept=True, offset=None)
Poisson(X, y, intercept=True, offset=None)
IVGMM(X, y, Z, weight=None, intercept=True)
ExponentialNLS(X, y, intercept=True)
NLS(y, mean=mean, jac=jacobian, weighted_hess=weighted_hessian, x0=start)
SyntheticControl(donors, treated)
```

`X` and donor matrices have observations in rows. Responses and offsets are 1-D;
matrices and vectors must be nonempty, finite and dimensionally consistent. An
intercept is added by default to regression models, including both `X` and `Z`
for IV. Synthetic control adds no intercept. Prediction `X` always has the same
number of original feature columns passed at construction.

`Ridge.penalty` can be a nonnegative scalar or a length-parameter vector. A scalar
excludes the added intercept, whereas a vector is taken literally. Logit requires
binary outcomes. Poisson permits any nonnegative response, supporting PPML. When
a model was fitted with an offset, new-data prediction requires its new offset;
the training offset is reused only for in-sample prediction.

IV's default weight is the inverse instrument second-moment matrix, giving 2SLS.
The sample moment Jacobian must have full column rank; a custom weight must be
positive definite. Supply exogenous regressors among instruments explicitly. The
array interface cannot infer endogenous and exogenous roles for you.

For generic NLS, `mean(beta)` returns `n` fitted means, `jac(beta)` returns `n × p`,
and `weighted_hess(beta, residuals)` returns the `p × p` sum of individual mean
Hessians weighted by `mean(beta)-y`. This is the exact residual-curvature term.

## Estimation results

```python
fit = model.fit(solver="tru", cov_type=None, clusters=None, **solve_options)
```

`solver` accepts `"tru"`, `"arc"`, `"qpb"` or the corresponding module. Smooth
models default to TRU and `scale="auto"`; SCM defaults to QPB. All other solver
keywords, including `x0`, `gtol`, `trace` and `raise_on_failure`, pass through.
Model `.fit()` sets `raise_on_failure=True` unless explicitly changed. An
unsuccessful estimate can be inspected with `raise_on_failure=False`; inference
on such a fit is refused.

`FitResult` exposes `params`, `success`, `predict(X=None, offset=None)`,
`fitted_values`, `resid` (observed minus fitted), `covariance`, `bse`, `conf_int()`
and `summary()`. `optimization` contains the full `OptimizeResult`.

The optimizer result provides `x`, `fun`, `jac`, `nit`, `success`, native `status`,
`message`, `optimality`, `seconds`, `nfev`, `njev`, `nhev`, `info`, `warnings`,
`scale`, `feasibility` and `history`. `info` is the unmodified native information
dictionary; its counters may differ from adapter counts because the adapter
caches objective/gradient evaluations. Adapter counts include preflight checks
and exclude final independent rescoring. QPB has zero callback counters.

`history` entries have `evaluation`, `seconds`, `objective` and `gradient_norm`.
They describe candidate evaluation points, including rejected trials. They are
not a major-iteration trace. Trace logging is opt-in and adds overhead.

## Covariance

`cov_type` accepts `None`, `"HC0"`, `"HC1"`, or `"cluster"`. The last requires
one cluster label per observation and at least two clusters. HC1 and CR1 require
more observations than parameters. There is no multiway-cluster interpretation
of a matrix of labels. Normal intervals are reported, including for CR1; no
few-cluster adjustment or cluster-t reference distribution is implied.

Covariance is unsupported for ridge and SCM. Singular or non-positive-definite
inference curvature raises `InferenceError`; the library does not substitute
a pseudoinverse. The optimizer itself may solve a nonunique point-estimation
problem successfully. Optimization success and identified statistical inference
are distinct checks.

## Native lifecycle

The Python binding has module-global solver state even though GALAHAD's lower-level
interfaces support independent instances. A single wrapper lock covers all native
solves, and cleanup occurs in `finally` blocks. Nested solves inside a callback
raise rather than deadlock. Do not mix concurrent direct `galahad.*` calls with
this wrapper; those calls bypass its lock. Use separate processes for parallel
fits.

The tested upstream binding may continue evaluating callbacks after a Python
exception. The adapter records the original exception, requests a native abort
through finite sentinel values, then re-raises the original error after native
cleanup. That path never returns a fit or computes inference. Callback references
are detached after solves so the native module cannot retain the last design
matrix indefinitely.
