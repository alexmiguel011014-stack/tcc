# TCC — repo rules

Slip-angle (β) estimation as a third MIMO output of the IC's MGGP virtual sensor. Plan: `GOALS.md`.
IC context: `CONTEXTO_IC.md`. Dataset legend: `docs/column_inventory.md`.

## Data
- `Database/*.xlsx` (Wang/Jilin) is partnership data: **never commit it**, never paste rows into docs or
  chat, never add derived time-series plots to the public repo without the advisor's OK
  (`docs/fig/*.png` stay untracked until then; aggregated tables are fine).
- `pd.read_excel(path, header=None)`, 24 float columns, no header. Inputs `u = cols [14, 2, 22, 23, 11]`
  (ax_nobias, acc_y, wheel front, wheel rear, YawRate), outputs `vx = col 18`, `vy = col 17`.
- Slip angle: col 3 (`Sideslip_angle`, deg, raw) and col 19 (`SA_smooth`, rad). Both are **computed**
  from the velocity pair (GOALS 1). **β for training/evaluation := `atan2(col 17, col 18)`, not col 19**
  — `vy_smooth` carries +0.30…0.37 m/s offsets on wang31/81/32 that col 19 was never recomputed for.
- Wheel-speed columns are rad/s (r_eff ≈ 0.30 m), not m/s. Col 15 is an exact copy of col 17.
- Tracks: wang21 train · wang22 val · wang31 testA · wang81 testB · wang32 testC.

## Library
- `src/` is the IC's `mggp` lib vendored at `7514bfb` (`src/VENDORED.md`). Do not edit it without adding a
  line there; `ruff` excludes it. Imports are absolute (`from src.mggp import MGGP`) → scripts insert the
  repo root into `sys.path`.
- NARX MIMO: every output's lagged values are terminals for every tree; Free-Run feeds all predicted
  outputs back. Operators available: `add`, `subtraction`, `mul`, `sign` — **no division**.
- Fitness = plain mean of per-output RMSE, no normalisation → β must be trained in degrees.
- `froe_mode=True` in MIMO is unverified (IC report says it did not work).

## IC anchors (hyper-parameters reused verbatim)
- **T48** (champion): `nDelays [1,2,5,10,25,50]`, `nTerms 7`, `maxHeight 6` — best seed `modelo_rmse_9.pkl`,
  4-track mean RMSE 0.2121. **T19** (elected, cheapest): `nDelays [1,2,5,10]`, `nTerms 3`, `maxHeight 5`,
  `modelo_rmse_6.pkl`, 0.2239. Fixed: 300 gen × 300 pop, k=300, MShooting→FreeRun, mut 0.3, cross 0.8,
  elite 10 %, `['add','subtraction','mul']`.

## Tooling
- Python `>=3.12,<3.13` pinned for parity with the IC (`numpy 2.2.6`, `numba 0.64.0`).
- Scripts carry PEP 723 inline deps: `uv run scripts/<x>.py`. `pyproject.toml` + `uv.lock` back
  `uv sync` for tests/lint; `requirements.txt` is `uv export --no-dev` for pip-only machines.
- Windows consoles are cp1252: scripts call `sys.stdout.reconfigure(encoding="utf-8")` at import.
- Long training runs go through `scripts/run_e2_mimo3.py` (resumable via `checklist.csv`/`attempts.csv`).
