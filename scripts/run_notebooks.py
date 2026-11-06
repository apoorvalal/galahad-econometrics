"""Execute notebooks in fresh kernels and empty working directories.

Run with the notebook extra, and the autodiff extra for autodiff_mnl.ipynb:
uv run --extra notebooks --extra autodiff python scripts/run_notebooks.py
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def code_hash(nb):
    source = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=300, help="Per-cell limit in seconds")
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.paths or sorted((root / "notebooks").glob("*.ipynb"))
    rows = []
    for path in paths:
        path = path.resolve()
        nb = nbformat.read(path, as_version=4)
        original_hash = code_hash(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        # Use this interpreter rather than whichever global 'python3' kernel the
        # operator happens to have installed. This mutates no kernelspec on disk.
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(nb, km=manager, timeout=args.timeout, allow_errors=False)
        started = perf_counter()
        with TemporaryDirectory(prefix="galahad-notebook-", dir=args.scratch_dir) as work:
            client.execute(cwd=work, cleanup_kc=True)
        assert code_hash(nb) == original_hash, "Execution changed a source cell"
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        # Atomic replacement leaves the previous successful notebook untouched if
        # a new run fails. Rich results stay embedded beside the code that made them.
        staging = path.with_suffix(".ipynb.tmp")
        nbformat.write(nb, staging)
        staging.replace(path)
        row = {
            "notebook": path.name,
            "code_cells": len(code_cells),
            "png_outputs": sum(
                "image/png" in o.get("data", {}) for c in code_cells for o in c.outputs
            ),
            "seconds": perf_counter() - started,
            "code_sha256": original_hash,
            "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fresh_kernel": True,
            "empty_working_directory": True,
            "errors": 0,
        }
        rows.append(row)
        print(
            f"Executed {path.name}: {row['code_cells']} code cells, {row['png_outputs']} figures",
            flush=True,
        )
    report = {
        "python": sys.version.split()[0],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["galahad-econometrics", "galahad-optrove", "numpy", "scipy", "nbclient"]
        },
        "notebooks": rows,
        "all_executed_without_errors": True,
    }
    report_path = root / "validation" / "notebooks.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
