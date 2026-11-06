"""Results keep native stopping status separate from independently checked accuracy."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtri

from .criteria import Array


@dataclass(frozen=True)
class Evaluation:
    evaluation: int
    seconds: float
    objective: float
    gradient_norm: float


@dataclass
class OptimizeResult:
    x: Array
    fun: float
    jac: Array
    success: bool
    status: int
    message: str
    nit: int
    optimality: float
    solver: str
    seconds: float
    nfev: int = 0
    njev: int = 0
    nhev: int = 0
    feasibility: float = 0.0
    scale: Array | None = None
    history: tuple[Evaluation, ...] = ()
    info: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        super().__init__(f"{result.solver}: {result.message} (status {result.status})")
        self.result = result


class InferenceError(ValueError):
    """Inference is unavailable or the sample curvature does not identify parameters."""


@dataclass
class FitResult:
    model: object
    optimization: OptimizeResult
    covariance: Array | None = None
    cov_type: str | None = None
    cov_info: dict = field(default_factory=dict)

    @property
    def params(self):
        return self.optimization.x

    @property
    def success(self):
        return self.optimization.success

    @property
    def bse(self):
        if self.covariance is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def fitted_values(self):
        return self.model.predict(self.params)

    @property
    def resid(self):
        return self.model.y - self.fitted_values

    def predict(self, X=None, *, offset=None):
        return self.model.predict(self.params, X, offset=offset)

    def conf_int(self, level=0.95):
        if self.bse is None:
            raise InferenceError("Request cov_type='HC1' or 'cluster' when fitting first")
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        width = ndtri((1.0 + level) / 2) * self.bse
        return np.column_stack([self.params - width, self.params + width])

    def summary(self):
        result = self.optimization
        lines = [
            f"{self.model.name} | GALAHAD {result.solver.upper()}",
            f"n={self.model.nobs}, parameters={self.params.size}, "
            f"success={result.success}, native status={result.status}",
            f"objective={result.fun:.8g}, optimality={result.optimality:.3g}, "
            f"iterations={result.nit}",
        ]
        if self.covariance is None:
            lines.append("Inference: not requested")
            lines.append(f"{'parameter':>14} {'estimate':>14}")
            lines.extend(
                f"{name:>14} {b:14.7g}" for name, b in zip(self.model.param_names, self.params)
            )
        else:
            lines.append(f"Inference: {self.cov_type}, normal-approximation 95% intervals")
            lines.append(
                f"{'parameter':>14} {'estimate':>14} {'std. error':>14} {'lower':>14} {'upper':>14}"
            )
            lines.extend(
                f"{name:>14} {b:14.7g} {se:14.7g} {lo:14.7g} {hi:14.7g}"
                for name, b, se, (lo, hi) in zip(
                    self.model.param_names, self.params, self.bse, self.conf_int()
                )
            )
        return "\n".join(lines)
