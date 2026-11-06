"""Trust-region minimization using GALAHAD TRU and direct TRS subproblems."""

from ._smooth import solve as _solve


def solve(criterion, x0=None, **kwargs):
    """Solve using exact Hessians. See ``docs/api.md`` for options and scaling."""
    return _solve("tru", criterion, x0, **kwargs)
