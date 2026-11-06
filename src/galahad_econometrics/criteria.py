"""The small callback contract accepted by TRU and ARC."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class SmoothCriterion(Protocol):
    def fun(self, x: Array) -> float: ...
    def grad(self, x: Array) -> Array: ...
    def hess(self, x: Array) -> Array: ...


@dataclass(frozen=True)
class Criterion:
    """A scalar minimization objective with its exact gradient and dense Hessian.

    All callbacks receive a one-dimensional float64 parameter vector. Callbacks
    must treat that vector as read-only. ``hess`` returns the full symmetric
    matrix; the adapter handles GALAHAD's lower-triangle packing.
    """

    fun: Callable[[Array], float]
    grad: Callable[[Array], Array]
    hess: Callable[[Array], Array]


def vector(value, name, size=None):
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must have length {size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def matrix(value, name, shape=None):
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def symmetric(value, name, n):
    result = matrix(value, name, (n, n))
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return (result + result.T) * 0.5


def check_derivatives(criterion, x, step=1e-5):
    """Centered finite-difference checks; reports errors, never changes derivatives.

    Intended for small user criteria, not as a derivative provider to a solver.
    Errors are maximum absolute discrepancies, normalized by max(1, reference).
    """
    x = vector(x, "x")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    delta = step * np.maximum(1.0, np.abs(x))
    directions = np.diag(delta)
    fd_g = np.array(
        [(criterion.fun(x + d) - criterion.fun(x - d)) / (2 * h) for d, h in zip(directions, delta)]
    )
    fd_h = np.column_stack(
        [
            (criterion.grad(x + d) - criterion.grad(x - d)) / (2 * h)
            for d, h in zip(directions, delta)
        ]
    )
    g = vector(criterion.grad(x), "gradient", x.size)
    h = symmetric(criterion.hess(x), "Hessian", x.size)
    return dict(
        gradient_error=float(np.max(np.abs(g - fd_g)) / max(1.0, np.max(np.abs(g)))),
        hessian_error=float(np.max(np.abs(h - fd_h)) / max(1.0, np.max(np.abs(h)))),
    )
