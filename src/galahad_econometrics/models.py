"""Dense, float64 model criteria. Designs are copied; no implicit data cleaning."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from .criteria import matrix, symmetric, vector
from .results import InferenceError


class Model:
    constrained = False
    supports_inference = True

    def fun(self, b):
        return self.value_gradient(b)[0]

    def grad(self, b):
        return self.value_gradient(b)[1]

    def fit(self, *, solver=None, cov_type=None, clusters=None, **options):
        """Estimate, optionally adding HC0/HC1 or one-way cluster-robust covariance.

        Smooth models default to TRU and regressor-RMS scaling; pass scale=None
        to use raw parameter units. Constrained synthetic control defaults to QPB.
        Failure raises ConvergenceError by default, with the result attached.
        """
        from . import arc, qpb, tru
        from .inference import covariance
        from .results import FitResult

        if cov_type is not None and not self.supports_inference:
            raise InferenceError(f"Covariance estimation is not implemented for {self.name}")
        if clusters is not None and cov_type != "cluster":
            raise ValueError("clusters requires cov_type='cluster'")
        if solver is None:
            solver = "qpb" if self.constrained else "tru"
        if isinstance(solver, str):
            if solver not in {"tru", "arc", "qpb"}:
                raise ValueError("solver must be 'tru', 'arc' or 'qpb'")
            solver = {"tru": tru, "arc": arc, "qpb": qpb}[solver]
        options.setdefault("raise_on_failure", True)
        if not self.constrained:
            options.setdefault("scale", "auto")
        optimization = solver.solve(self, **options)
        fit = FitResult(self, optimization)
        if cov_type is not None:
            if not optimization.success:
                raise InferenceError("Inference requires a fit that passes the accuracy check")
            fit.covariance, fit.cov_info = covariance(self, optimization.x, cov_type, clusters)
            fit.cov_type = cov_type
        return fit


class _Regression(Model):
    def __init__(self, X, y, *, intercept=True):
        X = matrix(X, "X")
        self.nobs, self.n_features = X.shape
        self.y = vector(y, "y", self.nobs)
        self.intercept = bool(intercept)
        self.X = np.column_stack([np.ones(self.nobs), X]) if intercept else X
        self.n_params = self.X.shape[1]
        self.x0 = np.zeros(self.n_params)
        self.parameter_scale = np.sqrt(np.mean(self.X**2, axis=0))
        self.parameter_scale[self.parameter_scale == 0] = 1.0
        self.param_names = (["intercept"] if intercept else []) + [
            f"x{i + 1}" for i in range(self.n_features)
        ]
        self.X.flags.writeable = self.y.flags.writeable = False

    def _design(self, X):
        if X is None:
            return self.X
        X = matrix(X, "prediction X")
        if X.shape[1] != self.n_features:
            raise ValueError(f"prediction X must have {self.n_features} columns")
        return np.column_stack([np.ones(len(X)), X]) if self.intercept else X

    def predict(self, b, X=None, *, offset=None):
        if offset is not None:
            raise ValueError(f"{self.name} does not use an offset")
        return self._design(X) @ b


class OLS(_Regression):
    """Minimize mean squared residuals divided by two."""

    name = "OLS"

    def __init__(self, X, y, *, intercept=True):
        super().__init__(X, y, intercept=intercept)
        self.H = self.X.T @ self.X / self.nobs

    def value_gradient(self, b):
        r = self.X @ b - self.y
        return float(r @ r / (2 * self.nobs)), self.X.T @ r / self.nobs

    def hess(self, b):
        return self.H

    def scores(self, b):
        return (self.X @ b - self.y)[:, None] * self.X


class Ridge(OLS):
    """Mean squared loss/2 + sum_j penalty_j * beta_j**2/2.

    A scalar penalty excludes the added intercept. A vector supplies explicit
    per-parameter penalties, including the intercept if present.
    """

    name = "Ridge"
    supports_inference = False

    def __init__(self, X, y, *, penalty=0.01, intercept=True):
        super().__init__(X, y, intercept=intercept)
        if np.ndim(penalty) == 0:
            penalty = np.full(self.n_params, float(penalty))
            if self.intercept:
                penalty[0] = 0.0
        self.penalty = vector(penalty, "penalty", self.n_params)
        if np.any(self.penalty < 0):
            raise ValueError("penalty must be nonnegative")
        self.H = self.H + np.diag(self.penalty)

    def value_gradient(self, b):
        f, g = super().value_gradient(b)
        return f + float(self.penalty @ (b * b) / 2), g + self.penalty * b


class _OffsetRegression(_Regression):
    def __init__(self, X, y, *, intercept=True, offset=None):
        super().__init__(X, y, intercept=intercept)
        self.has_offset = offset is not None
        self.offset = np.zeros(self.nobs) if offset is None else vector(offset, "offset", self.nobs)

    def _eta(self, b, X=None, offset=None):
        design = self._design(X)
        if offset is None:
            if X is not None and self.has_offset:
                raise ValueError("Supply prediction offsets for a model fitted with an offset")
            offset = self.offset if X is None else np.zeros(len(design))
        else:
            offset = vector(offset, "prediction offset", len(design))
        return design @ b + offset


class Logit(_OffsetRegression):
    """Binary-logit mean negative log-likelihood, with stable tails."""

    name = "Binary logit"

    def __init__(self, X, y, *, intercept=True, offset=None):
        super().__init__(X, y, intercept=intercept, offset=offset)
        if not np.isin(self.y, [0.0, 1.0]).all():
            raise ValueError("Logit y must contain only zeros and ones")

    def _residual(self, eta):
        return np.where(self.y == 1, -expit(-eta), expit(eta))

    def value_gradient(self, b):
        eta = self._eta(b)
        f = np.mean(np.logaddexp(0.0, np.where(self.y == 1, -eta, eta)))
        return float(f), self.X.T @ self._residual(eta) / self.nobs

    def hess(self, b):
        eta = self._eta(b)
        w = expit(eta) * expit(-eta)
        return self.X.T @ (w[:, None] * self.X) / self.nobs

    def scores(self, b):
        return self._residual(self._eta(b))[:, None] * self.X

    def predict(self, b, X=None, *, offset=None):
        return expit(self._eta(b, X, offset))


class Poisson(_OffsetRegression):
    """Poisson PML with log link, omitting the parameter-constant log(y!) term.

    Nonnegative fractional responses are allowed for PPML. Exposure enters as
    an explicitly supplied log exposure offset.
    """

    name = "Poisson PML"

    def __init__(self, X, y, *, intercept=True, offset=None):
        super().__init__(X, y, intercept=intercept, offset=offset)
        if np.any(self.y < 0):
            raise ValueError("Poisson y must be nonnegative")

    def value_gradient(self, b):
        eta = self._eta(b)
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(eta)
            return float(np.mean(mu - self.y * eta)), self.X.T @ (mu - self.y) / self.nobs

    def hess(self, b):
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(self._eta(b))
            return self.X.T @ (mu[:, None] * self.X) / self.nobs

    def scores(self, b):
        return (self.predict(b) - self.y)[:, None] * self.X

    def predict(self, b, X=None, *, offset=None):
        return np.exp(self._eta(b, X, offset))


class IVGMM(_Regression):
    """Fixed-weight linear GMM: 1/2 m(beta)' W m(beta), m=Z'(X beta-y)/n.

    X contains all regressors and Z all instruments. With intercept=True the
    constant is added to both. Default W=(Z'Z/n)^(-1) gives 2SLS. Exogenous
    regressors must also appear among the instruments supplied by the caller.
    This class does not perform a second-step estimated-weight update.
    """

    name = "Linear IV / fixed-weight GMM"

    def __init__(self, X, y, Z, *, weight=None, intercept=True):
        super().__init__(X, y, intercept=intercept)
        Z = matrix(Z, "Z")
        if Z.shape[0] != self.nobs:
            raise ValueError("Z and X must have the same number of rows")
        self.Z = np.column_stack([np.ones(self.nobs), Z]) if intercept else Z
        q = self.Z.shape[1]
        if q < self.n_params:
            raise ValueError("There must be at least as many instruments as parameters")
        self.D = self.Z.T @ self.X / self.nobs
        if np.linalg.matrix_rank(self.D) < self.n_params:
            raise ValueError("Sample moment Jacobian is rank deficient: parameters not identified")
        if weight is None:
            try:
                self.weight = cho_solve(cho_factor(self.Z.T @ self.Z / self.nobs), np.eye(q))
            except np.linalg.LinAlgError as exc:
                raise ValueError("Default IV weight requires full-rank instruments") from exc
        else:
            self.weight = symmetric(weight, "weight", q)
            try:
                np.linalg.cholesky(self.weight)
            except np.linalg.LinAlgError as exc:
                raise ValueError("GMM weight must be positive definite") from exc
        self.H = self.D.T @ self.weight @ self.D
        self.Z.flags.writeable = False

    def value_gradient(self, b):
        moment = self.Z.T @ (self.X @ b - self.y) / self.nobs
        return float(moment @ self.weight @ moment / 2), self.D.T @ self.weight @ moment

    def hess(self, b):
        return self.H

    def scores(self, b):
        return (self.X @ b - self.y)[:, None] * (self.Z @ self.weight @ self.D)


class NLS(Model):
    """Generic exact-Hessian nonlinear least squares.

    mean(b) returns n fitted means; jac(b) is their n-by-p Jacobian.
    weighted_hess(b, r) returns sum_i r_i * Hessian(mean_i(b)), a p-by-p
    matrix. This avoids allocating an n-by-p-by-p tensor. It is required;
    there is no silent Gauss-Newton replacement of exact curvature.
    """

    name = "Nonlinear least squares"

    def __init__(self, y, *, mean, jac, weighted_hess, x0):
        self.y = vector(y, "y")
        self.nobs = self.y.size
        self.x0 = vector(x0, "x0")
        self.n_params = self.x0.size
        self.mean, self.jac, self.weighted_hess = mean, jac, weighted_hess
        self.parameter_scale = np.ones(self.n_params)
        self.param_names = [f"beta{i + 1}" for i in range(self.n_params)]

    def value_gradient(self, b):
        r = vector(self.mean(b), "mean", self.nobs) - self.y
        J = matrix(self.jac(b), "mean Jacobian", (self.nobs, self.n_params))
        return float(r @ r / (2 * self.nobs)), J.T @ r / self.nobs

    def hess(self, b):
        r = vector(self.mean(b), "mean", self.nobs) - self.y
        J = matrix(self.jac(b), "mean Jacobian", (self.nobs, self.n_params))
        curvature = symmetric(self.weighted_hess(b, r), "weighted mean Hessian", self.n_params)
        return (J.T @ J + curvature) / self.nobs

    def scores(self, b):
        r = vector(self.mean(b), "mean", self.nobs) - self.y
        J = matrix(self.jac(b), "mean Jacobian", (self.nobs, self.n_params))
        return r[:, None] * J

    def predict(self, b, X=None, *, offset=None):
        if X is not None or offset is not None:
            raise ValueError(
                "Generic NLS predicts in sample; evaluate your mean function for new data"
            )
        return vector(self.mean(b), "mean", self.nobs)


class ExponentialNLS(_Regression):
    """Nonlinear least squares with conditional mean exp(X beta)."""

    name = "Exponential nonlinear least squares"

    def value_gradient(self, b):
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(self.X @ b)
            r = mu - self.y
            return float(r @ r / (2 * self.nobs)), self.X.T @ (mu * r) / self.nobs

    def hess(self, b):
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(self.X @ b)
            return self.X.T @ ((mu * (2 * mu - self.y))[:, None] * self.X) / self.nobs

    def scores(self, b):
        mu = self.predict(b)
        return (mu * (mu - self.y))[:, None] * self.X

    def predict(self, b, X=None, *, offset=None):
        return np.exp(super().predict(b, X, offset=offset))


class SyntheticControl(OLS):
    """Mean squared donor fit/2, weights >= 0 and sum(weights)=1; QPB only."""

    name = "Simplex synthetic control"
    constrained = True
    supports_inference = False

    def __init__(self, donors, treated):
        super().__init__(donors, treated, intercept=False)
        self.x0 = np.full(self.n_params, 1 / self.n_params)
        self.linear = -self.X.T @ self.y / self.nobs
        self.constant = float(self.y @ self.y / (2 * self.nobs))
        self.param_names = [f"donor{i + 1}" for i in range(self.n_params)]
