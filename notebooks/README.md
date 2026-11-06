# Worked estimation notebooks

These notebooks contain the data generation, criteria, derivative calculations,
fitting, numerical checks, tables and figures in ordinary top-to-bottom cells.
The checked-in copies include their executed outputs for GitHub preview. Every
example uses synthetic data and needs no downloads, credentials, saved fits or
notebook helper modules.

## Contents

| Notebook | Topic |
|---|---|
| [example.ipynb](example.ipynb) | Criterion → TRU/ARC → OLS, predictions and robust inference |
| [banana.ipynb](banana.ipynb) | Rosenbrock curvature, trial points and BFGS comparison |
| [likelihood_comparison.ipynb](likelihood_comparison.ipynb) | Matched Gaussian/logit/Poisson criteria, clean timings and convergence |
| [likelihood_comparison_estimators.ipynb](likelihood_comparison_estimators.ipynb) | All seven estimator families, recovery and clustered inference |
| [regularization_comparison.ipynb](regularization_comparison.ipynb) | Ridge paths, matched scikit-learn penalties and held-out evaluation |
| [gmm.ipynb](gmm.ipynb) | Linear IV, explicit two-step GMM, score/full bootstrap and nonlinear moments |
| [nonlinear_least_squares.ipynb](nonlinear_least_squares.ipynb) | Exact versus Gauss–Newton curvature, multiple starts and NLS inference |
| [synth.ipynb](synth.ipynb) | QPB/SLSQP donor weights, counterfactual paths and a null-effect simulation |
| [autodiff_mnl.ipynb](autodiff_mnl.ipynb) | Float64 JAX derivatives for a custom multinomial-logit criterion |

The ordering starts with the small criterion interface and then develops the
specialized estimation examples. The standalone notebooks can be opened in any
order. The low-level likelihood comparison supplies callbacks explicitly; the
estimator tour shows the convenience API.

## Run interactively

From the repository root:

```sh
uv sync --extra notebooks --extra autodiff
uv run --extra notebooks --extra autodiff jupyter lab notebooks/
```

Select the project's Python environment as the notebook kernel and run all cells.
The `notebooks` extra adds Jupyter, plotting, tables and scikit-learn. The optional
`autodiff` extra adds CPU JAX and is needed only for `autodiff_mnl.ipynb`.
All other notebooks run with `--extra notebooks` alone.

Each notebook sets its own random seed and process-local OpenMP controls before
using GALAHAD. BLAS work is limited to one thread for reproducible comparisons.
The library itself does not change these process-wide controls on import.

## Execute and retain outputs

```sh
uv run --extra notebooks --extra autodiff python scripts/run_notebooks.py
```

The runner starts a fresh kernel using its own Python interpreter for every
notebook, executes from an empty temporary working directory, and replaces the
checked-in file only after successful completion. It verifies that code cells
are unchanged and saves hashes/counts in `validation/notebooks.json`.
It does not install or overwrite a global kernelspec. A selected notebook can be
run by passing its path after the script name.

For HTML reading copies, install Quarto and run:

```sh
uv run --extra notebooks python scripts/render_notebooks.py /path/to/html-output
```

Rendering uses retained outputs and never refits the models. The generated
gallery links both HTML reading pages and the original executable notebooks.

## Interpreting the examples

- Timed likelihood fits use the same starts and criterion normalization. A warm-up
  precedes three clean repetitions; convergence traces come from separate runs.
- Solver histories contain candidate evaluations, including rejected trials.
  They are not accepted-iteration paths. JAX compilation is reported separately.
- Statistical recovery from the DGP is distinct from numerical agreement with a
  sample reference solution. Averaging units are stated beside each recovery metric.
- Ridge's mean-loss penalty corresponds to scikit-learn `alpha = n * penalty`.
  Validation selects the penalty; the test set is used only for final evaluation.
- The IV bootstrap holds the original GMM weight fixed in both the linearized
  and full-resampling procedures. A second-step weight update is explicit.
- SCM uses only pre-treatment data to fit weights and retains native convergence
  flags in the simulation. No normal SCM or ridge covariance is manufactured.
- Notebook computations demonstrate the current API and its assumptions. They do
  not add new library estimators or establish a universal optimizer speed ranking.
