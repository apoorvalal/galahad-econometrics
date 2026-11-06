"""GALAHAD QPB adapter for simplex synthetic-control weights."""

import warnings
from time import perf_counter

import numpy as np

from ._native import MESSAGES, native_session, update_options
from .criteria import vector
from .models import SyntheticControl
from .results import ConvergenceError, OptimizeResult


def solve(criterion, x0=None, *, gtol=1e-8, maxiter=1000, options=None, raise_on_failure=False):
    """Solve a SyntheticControl criterion and independently check simplex KKT gap.

    ``optimality`` is max(0, grad(x)'x - min(grad(x))); ``feasibility`` is
    max(abs(sum(x)-1), max(-x,0)). Weights are returned without clipping.
    The native solver has no Python iterate callback; history is empty.
    """
    if not isinstance(criterion, SyntheticControl):
        raise ValueError("qpb.solve currently accepts SyntheticControl models only")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    p = criterion.n_params
    x = vector(criterion.x0 if x0 is None else x0, "x0", p)
    lower = np.tril_indices(p)
    ne = len(lower[0])
    started = perf_counter()
    with native_session("qpb") as (module, controls):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            controls.update(print_level=0, error=-1)
            update_options(controls, options)
            # Request stricter native accuracy than the exposed certificate,
            # without multiplying an already-tight tolerance by the donor count.
            tol = gtol / 10
            controls.update(maxit=int(maxiter), stop_p=tol, stop_d=tol, stop_c=tol)
            module.load(p, 1, "dense", ne, None, None, None, "dense", p, None, None, None, controls)
            x, *_ = module.solve_qp(
                p,
                1,
                criterion.constant,
                criterion.linear,
                ne,
                criterion.H[lower],
                p,
                np.ones(p),
                np.ones(1),
                np.ones(1),
                np.zeros(p),
                np.ones(p),
                x,
                np.zeros(1),
                np.zeros(p),
            )
            info = module.information()
    x = x.copy()
    f, g = criterion.value_gradient(x)
    feasibility = float(max(abs(x.sum() - 1), max(0.0, -x.min())))
    optimality = float(max(0.0, g @ x - g.min()))
    status = int(info["status"])
    success = bool(
        status == 0
        and np.isfinite(x).all()
        and np.isfinite(f)
        and feasibility <= gtol
        and optimality <= gtol
    )
    message = MESSAGES.get(status, f"GALAHAD returned status {status}")
    if status == 0 and not success:
        message = "Native stop reported success but independent simplex/KKT check failed"
    result = OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        success=success,
        status=status,
        message=message,
        nit=int(info["iter"]),
        optimality=optimality,
        solver="qpb",
        seconds=perf_counter() - started,
        feasibility=feasibility,
        info=info,
        warnings=tuple(str(w.message) for w in caught),
    )
    if raise_on_failure and not success:
        raise ConvergenceError(result)
    return result
