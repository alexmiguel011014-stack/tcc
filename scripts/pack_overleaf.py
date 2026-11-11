# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""GOALS 5 item 5.8.1: zip monografia/ for Overleaf (New Project -> Upload Project).

Includes the gitignored figuras/dados/ (the PDF needs them); excludes LaTeX build outputs, working files and the
TikZ sources (figuras/tikz/: the pre-compiled figuras/diagramas/*.pdf go instead, so Overleaf does not load TikZ).

    uv run scripts/pack_overleaf.py            -> dist/monografia_overleaf.zip
"""

import fnmatch
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "monografia"
OUT = ROOT / "dist" / "monografia_overleaf.zip"
EXCLUDE = ["*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.loq", "*.bbl", "*.blg", "*.fdb_latexmk",
           "*.fls", "*.synctex.gz", "*.glg", "*.glo", "*.gls", "*.acn", "*.acr", "*.ist", "main.pdf",
           "CHECKLIST_UFLA.md"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            rel = p.relative_to(SRC).as_posix()
            if rel.startswith("figuras/tikz/"):      # fonte dos diagramas: o PDF pré-compilado é que vai para o Overleaf
                continue
            if p.is_file() and not any(fnmatch.fnmatch(p.name, pat) for pat in EXCLUDE):
                z.write(p, p.relative_to(SRC).as_posix())
                n += 1
    print(f"{OUT} — {n} arquivos, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
