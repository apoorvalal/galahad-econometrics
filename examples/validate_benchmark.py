"""Check both solvers against all original benchmark datasets and saved references.

Usage: python examples/validate_benchmark.py /path/to/extracted/expanded
The directory must contain problems.csv and solutions.npz from the linked study.
No old solver runs are executed; references are used only for independent scoring.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

# This executable controls its own process; importing the library does not.
os.environ.setdefault("OMP_CANCELLATION", "TRUE")
os.environ.setdefault("OMP_PROC_BIND", "TRUE")

import numpy as np
from benchmark_cases import make_problem
from scipy.special import expit
from threadpoolctl import threadpool_limits

from galahad_econometrics import (
    IVGMM,
    OLS,
    ExponentialNLS,
    Logit,
    Poisson,
    Ridge,
    SyntheticControl,
)


def model_for(d):
    X, y = d["X"], d["y"]
    family = d["family"]
    if family == "SCM":
        return SyntheticControl(X, y)
    if family == "IV-GMM":
        return IVGMM(X, y, d["Z"], intercept=False)
    if family == "Ridge":
        return Ridge(X, y, penalty=d["penalty"], intercept=False)
    constructor = {"OLS": OLS, "Logit": Logit, "Poisson": Poisson, "NLS": ExponentialNLS}
    return constructor[family](X, y, intercept=False)


def score(d, x):
    """Direct residual/moment scoring, separate from library model callbacks."""
    X, y = d["X"], d["y"]
    n, family = len(y), d["family"]
    eta = X @ x
    if family in {"OLS", "Ridge", "SCM"}:
        residual = eta - y
        f = residual @ residual / (2 * n) + d["penalty"] @ (x * x) / 2
        g = X.T @ residual / n + d["penalty"] * x
    elif family == "IV-GMM":
        Q, _ = np.linalg.qr(d["Z"], mode="reduced")
        A, target = Q.T @ X / np.sqrt(n), Q.T @ y / np.sqrt(n)
        residual = A @ x - target
        f, g = residual @ residual / 2, A.T @ residual
    elif family == "Logit":
        f = np.mean(np.logaddexp(0.0, eta) - y * eta)
        g = X.T @ (expit(eta) - y) / n
    elif family == "Poisson":
        mu = np.exp(eta)
        f, g = np.mean(mu - y * eta), X.T @ (mu - y) / n
    else:
        mu = np.exp(eta)
        f, g = np.mean((mu - y) ** 2) / 2, X.T @ (mu * (mu - y)) / n
    scale = np.sqrt(np.mean(X**2, axis=0))
    return float(f), np.asarray(g), float(np.max(np.abs(g / scale)))


def digest_data(d):
    digest = hashlib.sha256()
    for key in ["X", "y", "beta", "x0", "penalty", "Z"]:
        if d[key] is not None:
            digest.update(np.ascontiguousarray(d[key]).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/benchmark-compatibility.json")
    )
    args = parser.parse_args()
    cases = list(csv.DictReader((args.benchmark / "problems.csv").open()))
    refs = np.load(args.benchmark / "solutions.npz", allow_pickle=False)
    rows, data_matches = [], 0
    with threadpool_limits(limits=1):
        for case in cases:
            family = case["family"]
            d = make_problem(family, int(case["size"]), int(case["level"]), int(case["seed"]))
            assert digest_data(d) == case["data_sha256"], f"Data changed for {case['case']}"
            data_matches += 1
            reference = refs[case["case"] + "_reference"]
            fref, _, _ = score(d, reference)
            model = model_for(d)
            for solver in ["qpb"] if family == "SCM" else ["tru", "arc"]:
                row = dict(
                    case=case["case"],
                    family=family,
                    solver=solver,
                    accepted=False,
                    native_success=False,
                )
                try:
                    fit = model.fit(
                        solver=solver,
                        x0=d["x0"],
                        raise_on_failure=False,
                        options={"clock_time_limit": 30.0},
                    )
                    result = fit.optimization
                    f, g, scaled_gradient = score(d, fit.params)
                    gap = (f - fref) / max(1.0, abs(fref))
                    feasibility = max(0.0, -float(fit.params.min()), abs(fit.params.sum() - 1))
                    fw_gap = float(g @ fit.params - g.min())
                    accepted = (
                        (feasibility <= 1e-8 and fw_gap <= 1e-7)
                        if family == "SCM"
                        else (gap <= 1e-8 and scaled_gradient <= 1e-6)
                    )
                    row.update(
                        accepted=bool(accepted and np.isfinite(f)),
                        native_success=bool(result.success),
                        status=result.status,
                        objective=f,
                        relative_objective_gap=float(gap),
                        independent_scaled_gradient=scaled_gradient,
                        solver_optimality=result.optimality,
                        iterations=result.nit,
                        callback_objective_difference=abs(result.fun - f),
                        feasibility=float(feasibility) if family == "SCM" else 0.0,
                        fw_gap=fw_gap if family == "SCM" else None,
                        coefficient_error=float(
                            np.linalg.norm(fit.params - reference)
                            / max(1.0, np.linalg.norm(reference))
                        ),
                    )
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
            if int(case["size"]) == 1 and int(case["level"]) == 2 and int(case["seed"]) == 1:
                group = [r for r in rows if r["family"] == family]
                print(
                    f"{family}: {sum(r['accepted'] for r in group)}/{len(group)} accepted; "
                    f"{sum(r['native_success'] for r in group)}/{len(group)} native + strict checks",
                    flush=True,
                )
    report = dict(
        benchmark="https://lalten.org/quomodoc/docs/econometric-optimizer-comparison",
        python=platform.python_version(),
        machine=platform.machine(),
        versions={
            p: importlib.metadata.version(p)
            for p in ["galahad-econometrics", "galahad-optrove", "numpy", "scipy"]
        },
        platform=platform.system(),
        data_hashes_matched=data_matches,
        instances=len(cases),
        fits=len(rows),
        independent_accepted=sum(r["accepted"] for r in rows),
        native_and_strict_accepted=sum(r["native_success"] for r in rows),
        settings="Default model.fit RMS scaling, gtol=1e-8, maxiter=1000, one thread",
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} fits; matched {data_matches} original data hashes")
    if not all(r["accepted"] and r["native_success"] for r in rows):
        raise SystemExit("Some fits failed; see the saved report")


if __name__ == "__main__":
    main()
