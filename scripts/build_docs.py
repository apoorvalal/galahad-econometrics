"""Render portable documentation copies with Quarto; not needed to use the library.

Usage: python scripts/build_docs.py /path/to/output
The output directory is generated; canonical Markdown files are never edited.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

DOC_STYLE = """<style id="prototype-document-layout">
main {min-width:0} main a {overflow-wrap:anywhere}
main pre, main div.sourceCode {max-width:100%;overflow-x:auto}
main table {display:block;max-width:100%;overflow-x:auto;font-size:.9rem}
main .katex-display {max-width:100%;overflow-x:auto}
</style>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pages = {
        "README.md": "index",
        "docs/api.md": "api",
        "docs/methods.md": "methods",
        "validation/README.md": "validation",
    }
    for source, stem in pages.items():
        content = (root / source).read_text()
        title, body = content.split("\n", 1)
        for original, dest in pages.items():
            body = body.replace(f"]({original})", f"]({dest}.html)")
        if stem == "index":
            body += (
                "\n## Package downloads\n\n"
                "[Source repository snapshot](galahad-econometrics-source.zip) · "
                "[Installable wheel](galahad_econometrics-0.1.0-py3-none-any.whl) · "
                "[Source distribution](galahad_econometrics-0.1.0.tar.gz)\n"
            )
        front = (
            f'---\ntitle: "{title.removeprefix("# ")}"\n'
            "format:\n  html:\n    theme: cosmo\n    toc: true\n"
            "    html-math-method: katex\n    self-contained-math: true\n"
            "    embed-resources: true\nexecute:\n  enabled: false\n---\n\n"
        )
        source_path = output / f"{stem}.qmd"
        source_path.write_text(front + body)
        subprocess.run(["quarto", "render", source_path.name, "--quiet"], cwd=output, check=True)
        html = output / f"{stem}.html"
        html.write_text(html.read_text().replace("</head>", DOC_STYLE + "\n</head>"))
        source_path.unlink()
    for artifact in ["benchmark-compatibility.json", "install-check.json"]:
        src = root / "validation" / artifact
        if src.exists():
            shutil.copy2(src, output / artifact)


if __name__ == "__main__":
    main()
