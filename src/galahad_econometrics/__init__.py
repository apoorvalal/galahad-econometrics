"""Small GALAHAD-backed econometric estimators and reusable solver modules."""

from . import arc, qpb, tru
from .criteria import Criterion, check_derivatives
from .models import IVGMM, NLS, OLS, ExponentialNLS, Logit, Poisson, Ridge, SyntheticControl
from .results import ConvergenceError, FitResult, InferenceError, OptimizeResult

__version__ = "0.1.0"
__all__ = [
    "Criterion",
    "check_derivatives",
    "tru",
    "arc",
    "qpb",
    "OLS",
    "Ridge",
    "Logit",
    "Poisson",
    "IVGMM",
    "NLS",
    "ExponentialNLS",
    "SyntheticControl",
    "FitResult",
    "OptimizeResult",
    "ConvergenceError",
    "InferenceError",
]
