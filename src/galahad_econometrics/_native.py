"""Serialize the module-global Python bindings, including cleanup on exceptions."""

import importlib
import threading
from contextlib import contextmanager

_lock = threading.Lock()
_state = threading.local()


@contextmanager
def native_session(name):
    # The Fortran library has instance APIs; the current Python extension does not.
    if getattr(_state, "active", False):
        raise RuntimeError("Nested GALAHAD solves inside a callback are unsupported")
    with _lock:
        _state.active = True
        module = None
        initialized = False
        try:
            try:
                module = importlib.import_module(f"galahad.{name}")
            except ImportError as exc:
                raise ImportError(
                    "Install galahad-optrove in this Python environment; "
                    "there is no fallback optimizer"
                ) from exc
            options = module.initialize()
            initialized = True
            yield module, options
        finally:
            try:
                if initialized:
                    module.terminate()
            finally:
                _state.active = False


def update_options(defaults, overrides):
    """Reject misspellings before passing controls to the C extension."""
    for key, value in (overrides or {}).items():
        if key not in defaults:
            raise ValueError(f"Unknown GALAHAD option: {key}")
        if isinstance(value, dict):
            if not isinstance(defaults[key], dict):
                raise ValueError(f"GALAHAD option {key} is not a nested option dictionary")
            update_options(defaults[key], value)
        else:
            defaults[key] = value


MESSAGES = {
    0: "Native solver converged and the independent accuracy check passed",
    -7: "Objective appears unbounded below",
    -9: "Matrix analysis failed",
    -10: "Matrix factorization failed",
    -11: "Linear system solve failed",
    -16: "Ill-conditioning prevented further progress",
    -17: "Step became too small",
    -18: "Iteration limit reached",
    -19: "Time limit reached",
    -82: "Native solver was stopped externally",
}
