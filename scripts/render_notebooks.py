"""Export retained notebook outputs as a portable HTML reading gallery.

Requires Quarto and the notebook extra. This script never executes notebook code.
Canonical notebooks remain unchanged; HTML and downloadable copies go to output.
"""

import argparse
import hashlib
import html
import re
import shutil
import subprocess
from pathlib import Path

import nbformat
from build_docs import DOC_STYLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        cells = [c for c in nb.cells if c.cell_type == "code"]
        if any(c.execution_count is None for c in cells):
            raise ValueError(f"Execute every cell in {path.name} before exporting")
        if any(o.output_type == "error" for c in cells for o in c.outputs):
            raise ValueError(f"{path.name} retains an execution error")
        title = next(
            c.source.splitlines()[0].removeprefix("# ")
            for c in nb.cells
            if c.cell_type == "markdown"
        )
        original = path.read_bytes()
        shutil.copy2(path, output / path.name)
        subprocess.run(
            [
                "quarto",
                "render",
                path.name,
                "--to",
                "html",
                "--no-execute",
                "--output",
                path.stem + ".html",
                "--quiet",
                "-M",
                "theme:cosmo",
                "-M",
                "toc:true",
                "-M",
                "code-fold:false",
                "-M",
                "html-math-method:katex",
                "-M",
                "self-contained-math:true",
                "-M",
                "embed-resources:true",
                "-M",
                "pagetitle:" + title,
            ],
            cwd=output,
            check=True,
        )
        assert path.read_bytes() == original, "Rendering changed the canonical notebook"
        # The downloadable notebook is always the exact checked-in execution.
        (output / path.name).write_bytes(original)
        rendered = output / (path.stem + ".html")
        content = rendered.read_text().replace("</head>", DOC_STYLE + "\n</head>")
        navigation = (
            '<p><a href="index.html">All notebooks</a> · '
            f'<a href="{html.escape(path.name)}" download>Download executable notebook</a></p>'
        )
        content = re.sub(r"(<main\b[^>]*>)", lambda m: m.group(1) + navigation, content, count=1)
        rendered.write_text(content)
        entries.append((path.stem, title))
        print(f"Rendered {path.name} from retained outputs", flush=True)
    shutil.copy2(root / "notebooks" / "README.md", output / "README.md")
    rows = "".join(
        f'<tr><td><a href="{stem}.html">{html.escape(title)}</a></td>'
        f'<td><a href="{stem}.ipynb" download>.ipynb</a></td></tr>'
        for stem, title in entries
    )
    index = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GALAHAD Econometrics notebooks</title>
<style>body{font:17px/1.65 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#263238}
a{color:#126095}table{border-collapse:collapse;width:100%;font-size:.94rem}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}
pre{background:#f3f5f7;padding:14px;overflow:auto;border-radius:6px}code{font-size:.9em}h1{line-height:1.25}</style>
</head><body><main><p><a href="../">Library documentation</a></p>
<h1>GALAHAD Econometrics notebooks</h1>
<p>Nine worked notebooks with visible estimation code, simulated data, independent checks and embedded figures.
Each notebook has been executed top-to-bottom in its own fresh kernel from an empty working directory.
The reading pages use those retained outputs and are exported without rerunning any fits.</p>
<table><thead><tr><th scope="col">Read the notebook</th><th scope="col">Download</th></tr></thead><tbody>"""
        + rows
        + """</tbody></table>
<h2>Run interactively</h2>
<pre><code>uv sync --extra notebooks --extra autodiff
uv run --extra notebooks --extra autodiff jupyter lab notebooks/</code></pre>
<p>JAX is required only for the multinomial-logit example. All other notebooks need only the notebooks extra.
The <a href="https://github.com/apoorvalal/galahad-econometrics/tree/main/notebooks">private GitHub collection</a>
includes the executed notebooks and their full guide.</p>
<h2>Reproduce the executions</h2>
<pre><code>uv run --extra notebooks --extra autodiff python scripts/run_notebooks.py</code></pre>
<p>Solver histories are candidate evaluations, including rejected trials. Clean timing comparisons use separate
uninstrumented repetitions. Recovery metrics identify their estimation run and averaging unit.</p>
</main></body></html>"""
    )
    (output / "index.html").write_text(index)
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    import json

    (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
