"""Shared exact-Hessian adapter for GALAHAD TRU and ARC."""

import warnings
from time import perf_counter

import numpy as np

from ._native import MESSAGES, native_session, update_options
from .criteria import symmetric, vector
from .results import ConvergenceError, Evaluation, OptimizeResult


class _Callbacks:
    def __init__(self, criterion, scale, trace):
        self.criterion, self.scale, self.trace = criterion, scale, trace
        self.lower = np.tril_indices(scale.size)
        self.last = self.f = self.g = self.h_at = self.h = None
        self.nf = self.ng = self.nh = 0
        self.error = None
        self.abort_value = -np.finfo(np.float64).max
        self.abort_gradient = np.zeros(scale.size)
        self.abort_hessian = np.eye(scale.size)[self.lower]
        self.history = []
        self.started = perf_counter()

    def fg(self, z):
        if self.last is None or not np.array_equal(z, self.last):
            x = z / self.scale
            combined = getattr(self.criterion, "value_gradient", None)
            if combined is None:
                f, g = self.criterion.fun(x), self.criterion.grad(x)
            else:
                f, g = combined(x)
            if np.ndim(f) != 0 or np.isnan(f) or np.isneginf(f):
                raise FloatingPointError("Objective is not a finite scalar; check the start/domain")
            # +inf is an out-of-domain trial, so the native method can reject
            # it and shrink its radius / increase regularization. No derivative
            # is manufactured there. The starting point must still be finite.
            g = None if np.isposinf(f) else vector(g, "gradient", z.size) / self.scale
            if g is not None and not np.isfinite(g).all():
                raise FloatingPointError("Parameter scaling overflowed the gradient")
            self.last, self.f, self.g = z.copy(), float(f), g
            self.nf += 1
            self.ng += 1
            if self.trace:
                self.history.append(
                    Evaluation(
                        self.nf,
                        perf_counter() - self.started,
                        float(f),
                        float(np.linalg.norm(g)) if g is not None else float("inf"),
                    )
                )
        return self.f, self.g

    def fun(self, z):
        if self.error is not None:
            return self.abort_value
        try:
            return self.fg(z)[0]
        except BaseException as exc:
            self.error = exc
            return self.abort_value

    def grad(self, z):
        if self.error is not None:
            return self.abort_gradient
        try:
            gradient = self.fg(z)[1]
            if gradient is None:
                raise FloatingPointError(
                    "Native solver requested a gradient at an infinite objective"
                )
            return gradient
        except BaseException as exc:
            self.error = exc
            return self.abort_gradient

    def hess(self, z):
        if self.error is not None:
            return self.abort_hessian
        try:
            if self.h_at is None or not np.array_equal(z, self.h_at):
                h = symmetric(self.criterion.hess(z / self.scale), "Hessian", z.size)
                self.h = np.ascontiguousarray((h / np.outer(self.scale, self.scale))[self.lower])
                if not np.isfinite(self.h).all():
                    raise FloatingPointError("Parameter scaling overflowed the Hessian")
                self.h_at = z.copy()
                self.nh += 1
            return self.h
        except BaseException as exc:
            self.error = exc
            return self.abort_hessian

    def release(self):
        # Native modules keep callback references even after terminate(). Do not
        # let the last solve retain large design matrices through these methods.
        self.criterion = self.last = self.g = self.h = self.h_at = None
        self.error = None
        self.history = []
        self.lower = self.scale = self.abort_gradient = self.abort_hessian = None


def solve(
    name,
    criterion,
    x0=None,
    *,
    gtol=1e-8,
    maxiter=1000,
    scale=None,
    trace=False,
    options=None,
    raise_on_failure=False,
):
    if getattr(criterion, "constrained", False):
        raise ValueError("This problem is constrained; use qpb.solve for synthetic control")
    if x0 is None:
        x0 = getattr(criterion, "x0", None)
    if x0 is None:
        raise ValueError("x0 is required for a user-supplied criterion")
    x0 = vector(x0, "x0")
    if getattr(criterion, "n_params", x0.size) != x0.size:
        raise ValueError("x0 does not match the model's parameter count")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if isinstance(scale, str):
        if scale != "auto" or not hasattr(criterion, "parameter_scale"):
            raise ValueError("scale='auto' requires a model with parameter_scale")
        scale = criterion.parameter_scale
    scale = np.ones_like(x0) if scale is None else vector(scale, "scale", x0.size)
    if np.any(scale <= 0):
        raise ValueError("scale entries must be strictly positive")
    z = np.ascontiguousarray(x0 * scale)
    cb = _Callbacks(criterion, scale, trace)
    started = perf_counter()
    try:
        # Validate the callback contract before entering the native solver.
        initial_value = cb.fun(z)
        if cb.error is not None:
            raise cb.error
        if not np.isfinite(initial_value):
            raise FloatingPointError("Starting objective must be finite; choose a valid x0")
        cb.hess(z)
        if cb.error is not None:
            raise cb.error
        with native_session(name) as (module, controls):
            # Python 3.12 warning filters are process-global: serialize their
            # capture/restoration together with the native calls.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                controls.update(print_level=0, error=-1)
                update_options(controls, options)
                # Exact derivatives, direct subproblems, Euclidean scaled-space
                # norm, and the requested absolute tolerance are invariant.
                controls.update(
                    maxit=int(maxiter),
                    model=2,
                    norm=-1,
                    subproblem_direct=True,
                    stop_g_absolute=float(gtol),
                    stop_g_relative=0.0,
                )
                ne = len(cb.lower[0])
                module.load(z.size, "dense", ne, None, None, None, options=controls)
                # Some versions of the C binding keep calling Python after a
                # callback raises, with PyErr still set. Never pass an exception
                # across that boundary. A recorded error supplies a finite abort
                # sentinel below obj_unbounded, then is re-raised in Python.
                # No result or inference can be returned from that abort path.
                z, _ = module.solve(z.size, ne, z, cb.fun, cb.grad, cb.hess)
                if cb.error is not None:
                    raise cb.error
                info = module.information()
        x = np.asarray(z / scale).copy()
        # Re-evaluate in original units, bypassing the adapter's last-point cache.
        f = float(criterion.fun(x))
        g = vector(criterion.grad(x), "final gradient", x.size)
        optimality = float(np.linalg.norm(g / scale))
        status = int(info["status"])
        success = status == 0 and np.isfinite(f) and optimality <= gtol * (1 + 1e-6)
        message = MESSAGES.get(status, f"GALAHAD returned status {status}")
        if status == 0 and not success:
            message = "Native stop reported success but independent gradient check failed"
        result = OptimizeResult(
            x=x,
            fun=f,
            jac=g,
            success=success,
            status=status,
            message=message,
            nit=int(info["iter"]),
            optimality=optimality,
            solver=name,
            seconds=perf_counter() - started,
            nfev=cb.nf,
            njev=cb.ng,
            nhev=cb.nh,
            scale=scale.copy(),
            history=tuple(cb.history),
            info=info,
            warnings=tuple(str(w.message) for w in caught),
        )
    finally:
        cb.release()
    if raise_on_failure and not result.success:
        raise ConvergenceError(result)
    return result
