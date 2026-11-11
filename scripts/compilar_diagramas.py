# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Pre-compile monografia/figuras/tikz/*.tex into vector PDFs in monografia/figuras/diagramas/.

Why: the free Overleaf plan stops a compile after 10 s, and loading TikZ costs about 0.3-0.6 s on every LaTeX pass
(4-5 passes per compile). The monograph therefore includes these PDFs with \\includegraphics and does not load TikZ.
The .tex sources stay in figuras/tikz/ (edit them, then run this script again).

    uv run scripts/compilar_diagramas.py

Needs a local LaTeX with tikz, standalone, pslatex, mathptmx. The script checks the files first and stops with a
message if one is missing, instead of letting MiKTeX try (and fail) to install packages through pop-up windows.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "monografia" / "figuras" / "tikz"
OUT = ROOT / "monografia" / "figuras" / "diagramas"
NEEDED = ["standalone.cls", "standalone.sty", "tikz.sty", "pslatex.sty", "mathptmx.sty", "amssymb.sty", "xkeyval.sty"]
WRAPPER = r"""\documentclass[12pt,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{pslatex}
\usepackage{mathptmx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,calc}
\begin{document}
\input{%s}
\end{document}
"""


def kpse(name):
    r = subprocess.run(["kpsewhich", name], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    missing = [n for n in NEEDED if not kpse(n)]
    if missing:
        sys.exit(f"faltam no LaTeX local: {missing}. Instale-os (MiKTeX Console / tlmgr) e rode de novo.")
    OUT.mkdir(parents=True, exist_ok=True)
    for tex in sorted(SRC.glob("*.tex")):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(tex, tmp / tex.name)
            (tmp / "w.tex").write_text(WRAPPER % tex.stem, encoding="utf-8")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "w.tex"], cwd=tmp,
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0 or not (tmp / "w.pdf").exists():
                sys.exit(f"{tex.name}: pdflatex falhou\n{(tmp / 'w.log').read_text(errors='replace')[-1500:]}")
            shutil.copy(tmp / "w.pdf", OUT / f"{tex.stem}.pdf")
        print(f"  {OUT / (tex.stem + '.pdf')}")


if __name__ == "__main__":
    main()
