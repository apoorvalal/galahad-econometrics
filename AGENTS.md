# Development notes

- This is the canonical prototype repository. The earlier benchmark is a reference
  and must not be rewritten by library development. CrabbyMetrics is out of scope.
- Keep model criteria independent of TRU/ARC. Both supply exact dense Hessians;
  synthetic-control constraints belong to QPB.
- Average-loss normalization, intercept handling, original-unit outputs and
  `z = scale * beta` transformations are public API contracts.
- Keep native statuses and independent checks separate. Never loosen an independent
  accuracy threshold to hide a failed fit. Callback histories are evaluation points,
  including rejected trials, rather than accepted iterates.
- The native Python bindings have global state and retain callbacks. Preserve
  serialization, nested-call rejection, exception bridging and reference cleanup.
- Covariance is opt-in. The sandwich uses mean Hessians and a `1/n**2` factor with
  summed gradient outer products. Ridge/SCM inference is explicitly unsupported.
- Run `uv run pytest` and `uv run ruff check src tests examples scripts` after
  code changes. The optional original-suite audit is described in validation/README.md.
- Numerical test results and benchmarks are retained in validation/. Build outputs,
  venvs and disposable caches must stay untracked. Do not create a public remote
  or publish a package release without a request to do so.
- Notebooks are interactive, top-to-bottom sources with retained outputs, rather
  than scripts hidden inside main(). Keep all numerical examples self-contained.
- Before committing notebook revisions, run the complete scripts/run_notebooks.py
  collection with the notebooks/autodiff extras. The final execution manifest must
  cover every notebook. Use scripts/render_notebooks.py for reading copies; it
  exports retained outputs with --no-execute and must not change source notebooks.
