"""Adaptive cubic regularization using GALAHAD ARC and direct RQS subproblems."""

from ._smooth import solve as _solve


def solve(criterion, x0=None, **kwargs):
    """Solve using exact Hessians. See ``docs/api.md`` for options and scaling."""
    return _solve("arc", criterion, x0, **kwargs)
