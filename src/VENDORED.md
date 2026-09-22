# Vendored `mggp` library

- Source: https://github.com/alexmiguel011014-stack/TreinamentoMGGP_2_0 — `src/` at commit `7514bfb`
  (copied 2026-09-21 from the local clone `D:\ProjetosPessoais\IC\TreinamentoMGGP_2_0`).
- Files: `base.py`, `mggp.py`, `predictors.py`, `crossings.py`, `mutations.py` (`main.py` not vendored — demo script).
- Imports inside the lib are absolute (`from src.predictors import …`); callers must have the repo root on `sys.path`.
- `ruff` excludes this directory; the lib is not reformatted.

## Local patches

(none)
