# Validation

Version 0.1.0 was validated on Apple Silicon with Python 3.12.9,
GALAHAD-OPTrove 5.5.3, NumPy 2.5.4 and SciPy 1.18.1.

## Unit and integration tests

Run `uv run pytest`. The 45 tests cover:

- OLS and ridge versus augmented-design least-squares solutions.
- Logit and offset Poisson versus independently specified SciPy objectives.
- Default 2SLS and custom-weight IV/GMM versus closed-form point estimates and
  independently constructed GMM covariance matrices.
- Exact generic and exponential NLS curvature, and fits versus SciPy least squares.
- Synthetic-control weights versus SLSQP, with independent simplex/KKT checks.
- Feature-unit invariance of Poisson coefficients and covariance.
- Finite-difference gradients and exact Hessians for each smooth family.
- HC0/HC1 and one-way CR1 versus explicit sandwich calculations, including a
  repeated-observation covariance normalization check.
- Trial-domain rejection, input validation, iteration limits, callback exceptions,
  reentrant-call rejection, thread serialization and callback-reference cleanup.

Ruff lint and formatting checks also pass. Tests deliberately distinguish
objective/gradient accuracy from raw coefficient precision near boundaries or
under weak curvature. Tighter coefficient resolution can require tighter
optimization certificates.

## Original 84-case suite

The public [optimizer study](https://lalten.org/quomodoc/docs/econometric-optimizer-comparison)
provides the benchmark's source/data generator, `problems.csv` and reference
vectors in `solutions.npz`. Its [source archive](https://lalten.org/pages/econometric-optimizer-comparison/galahadro-expanded-source.zip)
can be extracted locally for this optional check:

```sh
uv run python examples/validate_benchmark.py /path/to/extracted/expanded
```

Pass the directory containing `problems.csv` and `solutions.npz`. The copied
synthetic-data generator in `examples/benchmark_cases.py` is numerically unchanged.
Before any fit, each dataset is compared with its original SHA-256, including
design, response, true parameters, start, penalty and instruments. All 84 hashes
match in the validated environment. Hash differences on another numerical stack
are reported rather than silently relabelled as the original dataset.

| Solver | Applicable instances | Native and independent checks passed |
|---|---:|---:|
| TRU | 72 | 72 |
| ARC | 72 | 72 |
| QPB | 12 | 12 |

There is one fit per applicable solver/instance. These 156 runs check the new API;
they do not repeat the previous 3,024-fit timing experiment. Default model fitting
uses regressor-RMS coordinate scaling, gradient tolerance `1e-8`, 1,000 maximum
iterations, and one BLAS thread for this check. QPB uses original weight coordinates.

`benchmark-compatibility.json` retains every case, native status, independently
scored objective gap and gradient/certificate, and coefficient discrepancy from
the reference. The maximum relative objective gap is below `4.2e-9` across all
families. The largest objective discrepancy between the wrapper and independent
scoring is below `2.8e-17`. Ill-conditioning and nearly interchangeable donors can
make coefficient differences substantially larger than objective differences.

The audit uses direct residual losses, stable likelihood formulas, and a QR
projection for IV; it does not call the library's callbacks to score the returned
solution. For smooth models it requires relative objective gap at most `1e-8`
and regressor-scaled infinity gradient at most `1e-6`. For SCM it requires simplex
feasibility at most `1e-8` and Frank–Wolfe gap at most `1e-7`. Separately, every
library result passes its own stricter `gtol=1e-8` check and native status zero.

Validation of these finite samples does not certify identification, finite-MLE
existence, global optimization of a nonlinear criterion or inference coverage.

## Worked notebooks

The nine notebooks add 51 executed code cells and 13 embedded figures. Each runs
in its own fresh kernel, using the project's interpreter, from an empty temporary
working directory. No example reads saved estimates, downloads data or imports
notebook helper modules. All nine complete without error outputs; all displayed
tables have at most ten rows including headers.

Execution counts, code hashes, notebook hashes and package versions are saved in
[notebooks.json](notebooks.json); the compact structural checks are in
[notebook-audit.json](notebook-audit.json). The 45 original library tests still
pass. These examples add documentation and optional dependencies without
changing the numerical library implementation or repeating the earlier timing
benchmark. The notebook likelihood comparison is its own small three-repetition
demonstration, with separate instrumented convergence runs.
