# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "deap==1.4.3", "numpy==2.2.6", "pandas==2.3.3", "scipy==1.15.3", "scikit-learn==1.7.2",
#   "matplotlib==3.10.8", "numba==0.64.0", "tqdm==4.67.3", "joblib==1.5.3", "ipython==8.38.0",
#   "openpyxl", "dill",
# ]
# ///
"""GOALS 3 — E2: train the IC champion configuration (T48) as a 5×3 MIMO (vx, vy, β) N times.

Port of `TrainingMGGP_LOOP.ipynb` (IC repo) to a resumable CLI:
  * one fixed configuration (--config t48 | t19) instead of the control spreadsheet;
  * third output β = atan2(vy_smooth, vx_smooth) (cols 17, 18) — GOALS 1 verdict; NOT col 19;
  * --arm 5x2 keeps the IC's two outputs (arm A) so the same script produces the baseline;
  * every model is validated Free-Run on the 5 tracks (Wang 3.2 included) with RMSE per output,
    plus β̂ = atan2(v̂y, v̂x) computed from the predicted velocities (so a 5×3 model reports both
    its evolved β output and the post-hoc β);
  * checkpoint files let the loop resume after a power failure exactly like the notebook did.

Run from the repo root:
  uv run scripts/run_e2_mimo3.py --config t48 --arm 5x3 --runs 30
  uv run scripts/run_e2_mimo3.py --config t48 --arm 5x2 --runs 30          # arm A baseline
  uv run scripts/run_e2_mimo3.py --runs 1 --generations 3 --population 20 --max-samples 1500 --out results/smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import dill
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the vendored lib uses absolute `from src.… import …`
from src.mggp import MGGP  # noqa: E402

DATA_DIR = ROOT / "Database"
TRACKS = {  # name -> (file, role)
    "Wang 2.1 (Treino)": ("wang21dv_bic_MGGP.xlsx", "train"),
    "Wang 2.2 (Validação)": ("wang22dv_bic_MGGP.xlsx", "val"),
    "Wang 3.1 (Teste A)": ("wang31dv_bic_MGGP.xlsx", "testA"),
    "Wang 8.1 (Teste B)": ("wang81dv_bic_MGGP.xlsx", "testB"),
    "Wang 3.2 (Teste C)": ("wang32dv_bic_MGGP.xlsx", "testC"),
}
INPUT_COLS = [14, 2, 22, 23, 11]  # ax_nobias, acc_y, wheel front, wheel rear, YawRate — as the IC
VX_COL, VY_COL = 18, 17

# IC anchors (parametros_utilizados.csv of each training). Only nDelays/nTerms/maxHeight differ.
CONFIGS = {
    "t48": {"nDelays": [1, 2, 5, 10, 25, 50], "nTerms": 7, "maxHeight": 6},
    "t19": {"nDelays": [1, 2, 5, 10], "nTerms": 3, "maxHeight": 5},
}
FIXED = {
    "generations": 300, "populationSize": 300, "evaluationMode": "RMSE", "k": 300,
    "evaluationType": "MShooting", "evaluationTypeTest": "FreeRun",
    "mutationRate": 0.3, "crossoverRate": 0.8, "elitePercentage": 10,
    "mode": "MIMO", "froe_mode": False, "operators": ["add", "subtraction", "mul"],
}
APPROVAL_RMSE_MAX = 100.0  # notebook rule: 0 < RMSE(wang22, FreeRun) < 100


# ----------------------------------------------------------------------------- data
def load_track(file: str, arm: str, beta_unit: str, max_samples: int | None):
    """Same loader as the IC notebook (`carregar_dados_bicicleta`) + optional β third output."""
    df = pd.read_excel(DATA_DIR / file, header=None)
    if max_samples:
        df = df.iloc[:max_samples]
    u = df.iloc[:, INPUT_COLS].to_numpy(dtype=float)
    vx, vy = df[VX_COL].to_numpy(dtype=float), df[VY_COL].to_numpy(dtype=float)
    beta = np.arctan2(vy, vx)
    if beta_unit == "deg":
        beta = np.degrees(beta)
    y = np.column_stack([vx, vy, beta]) if arm == "5x3" else np.column_stack([vx, vy])
    return u, y


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def to_rad(x: np.ndarray, beta_unit: str) -> np.ndarray:
    return np.radians(x) if beta_unit == "deg" else x


# ----------------------------------------------------------------------------- checkpoints
def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def free_run(model, y: np.ndarray, u: np.ndarray, params: dict):
    args = (y, u) if params["evaluationTypeTest"] != "MShooting" else (params["k"], y, u)
    yp, yd = model.predict(params["evaluationTypeTest"], *args)
    n = min(len(yp), len(yd))
    return yp[:n], yd[:n]


# ----------------------------------------------------------------------------- validation
def validate_model(model, model_name: str, arm: str, beta_unit: str, params: dict, out: Path,
                   max_samples: int | None) -> list[dict]:
    rows = []
    n_out = 3 if arm == "5x3" else 2
    fig, axes = plt.subplots(len(TRACKS), 3, figsize=(20, 4 * len(TRACKS)))
    fig.suptitle(f"Free-Run validation — {model_name} ({arm}, β in {beta_unit})", fontsize=15, y=1.0)

    for i, (track, (file, role)) in enumerate(TRACKS.items()):
        u, y = load_track(file, arm, beta_unit, max_samples)
        try:
            yp, yd = free_run(model, y, u, params)
            ok = np.all(np.isfinite(yp)) and np.abs(yp).max() < 1e4
        except Exception as e:  # divergence / LinAlg — record and move on, like the notebook
            print(f"    ! {track}: {type(e).__name__}: {e}")
            yp, yd, ok = np.zeros_like(y), y, False

        beta_true = to_rad(yd[:, 2], beta_unit) if n_out == 3 else np.arctan2(yd[:, 1], yd[:, 0])
        beta_post = np.arctan2(yp[:, 1], yp[:, 0])  # β from the predicted velocities (arm A logic)
        row = {
            "Modelo": model_name, "Pista": track, "Papel": role, "ok": bool(ok),
            "RMSE_Vx": rmse(yd[:, 0], yp[:, 0]) if ok else np.nan,
            "RMSE_Vy": rmse(yd[:, 1], yp[:, 1]) if ok else np.nan,
            "RMSE_beta_atan2_deg": np.degrees(rmse(beta_true, beta_post)) if ok else np.nan,
        }
        if n_out == 3:
            beta_out = to_rad(yp[:, 2], beta_unit)
            row["RMSE_beta_out_deg"] = np.degrees(rmse(beta_true, beta_out)) if ok else np.nan
        row["RMSE_Medio_VxVy"] = (row["RMSE_Vx"] + row["RMSE_Vy"]) / 2
        rows.append(row)

        # --- plots: vx | vy | β (evolved output if 5x3, post-hoc atan2 always)
        for j, (lbl, col) in enumerate([("Vx [m/s]", 0), ("Vy [m/s]", 1)]):
            ax = axes[i, j]
            ax.plot(yd[:, col], color="black", lw=0.8, label="Real")
            ax.plot(yp[:, col], color="tab:blue" if j == 0 else "tab:red", lw=0.8, ls="--", label="Previsto")
            key = "RMSE_Vx" if j == 0 else "RMSE_Vy"
            ax.set_title(f"{track} — {lbl} | RMSE {row[key]:.4f}")
            ax.grid(True, ls=":", alpha=0.6)
        ax = axes[i, 2]
        ax.plot(np.degrees(beta_true), color="black", lw=0.8, label="β real")
        ax.plot(np.degrees(beta_post), color="tab:green", lw=0.8, ls="--",
                label=f"atan2(v̂y, v̂x) | {row['RMSE_beta_atan2_deg']:.3f}°")
        if n_out == 3:
            ax.plot(np.degrees(beta_out), color="tab:purple", lw=0.8, ls=":",
                    label=f"saída 3 | {row['RMSE_beta_out_deg']:.3f}°")
        ax.set_title(f"{track} — β [deg]")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    (out / "fig").mkdir(exist_ok=True)
    fig.savefig(out / "fig" / f"{model_name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=CONFIGS, default="t48")
    ap.add_argument("--arm", choices=["5x3", "5x2"], default="5x3")
    ap.add_argument("--beta-unit", choices=["deg", "rad"], default="deg",
                    help="unit of the β output during training; fitness is the plain mean of per-output RMSE, "
                         "so rad (std≈0.04) would make β invisible next to vx (std≈3.6 m/s)")
    ap.add_argument("--runs", type=int, default=30, help="approved models to reach")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: 3 × runs")
    ap.add_argument("--seed", type=int, default=2026, help="base seed; attempt i uses seed+i")
    ap.add_argument("--out", type=Path, default=None, help="default: results/e2/<config>_<arm>_<unit>")
    ap.add_argument("--generations", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--population", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--max-samples", type=int, default=None, help="truncate every track (smoke tests only)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    params = {**FIXED, **CONFIGS[a.config]}
    if a.generations:
        params["generations"] = a.generations
    if a.population:
        params["populationSize"] = a.population
    out = a.out or ROOT / "results" / "e2" / f"{a.config}_{a.arm}_{a.beta_unit}"
    (out / "models").mkdir(parents=True, exist_ok=True)
    max_attempts = a.max_attempts or 3 * a.runs

    meta = {"config": a.config, "arm": a.arm, "beta_unit": a.beta_unit, "beta_def": "atan2(col17, col18)",
            "input_cols": INPUT_COLS, "vx_col": VX_COL, "vy_col": VY_COL, "runs": a.runs, "base_seed": a.seed,
            "max_samples": a.max_samples, "params": params, "lib": "src/ vendored @7514bfb"}
    (out / "params.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # same file the IC produced, for side-by-side reading
    pd.DataFrame(list(params.items()), columns=["Parametro", "Valor"]).to_csv(out / "parametros_utilizados.csv", index=False)

    ck_path, log_path, rep_path = out / "checklist.csv", out / "attempts.csv", out / "relatorio_validacao_rmse.csv"
    ck = read_csv(ck_path, ["modelo", "seed", "attempt", "rmse_val", "treino", "validacao", "started", "finished"])
    attempts_df = read_csv(log_path, ["attempt", "seed", "rmse_val", "approved", "seconds", "when"])
    approved = int((ck["treino"] == "Concluído").sum()) if len(ck) else 0
    attempt = int(attempts_df["attempt"].max()) if len(attempts_df) else 0
    print(f"[E2] {a.config} {a.arm} β={a.beta_unit} → {out}\n[E2] resuming: {approved}/{a.runs} approved, "
          f"{attempt} attempts done\n")

    u_train, y_train = load_track(TRACKS["Wang 2.1 (Treino)"][0], a.arm, a.beta_unit, a.max_samples)
    u_val, y_val = load_track(TRACKS["Wang 2.2 (Validação)"][0], a.arm, a.beta_unit, a.max_samples)
    print(f"[E2] train {u_train.shape} → {y_train.shape} | val {u_val.shape} → {y_val.shape}")

    # --- 1. factory: keep training until `runs` models pass the wang22 Free-Run gate
    while approved < a.runs and attempt < max_attempts:
        attempt += 1
        seed = a.seed + attempt
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        print(f"\n{'=' * 80}\n[E2] attempt {attempt}/{max_attempts} · seed {seed} · approved {approved}/{a.runs}\n{'=' * 80}")

        tmp = out / "temp_mggp.pkl"
        mggp = MGGP(inputs=u_train, outputs=y_train, validation=(u_val, y_val), filename=str(tmp), **params)
        mggp.run()
        best = mggp._hof[0]

        err_val = 999.0
        try:
            yp, yd = free_run(best, y_val, u_val, params)
            err_val = float(best.score(yd, yp, params["evaluationMode"]))
        except Exception as e:
            print(f"    ! validation failed: {type(e).__name__}: {e}")
        ok = 0.0 < err_val < APPROVAL_RMSE_MAX and np.isfinite(err_val)
        secs = round(time.time() - t0)

        if ok:
            approved += 1
            name = f"modelo_rmse_{approved}"
            with open(out / "models" / f"{name}.pkl", "wb") as f:
                dill.dump(best, f)
            ck.loc[len(ck)] = [name, seed, attempt, err_val, "Concluído", "Pendente",
                               datetime.fromtimestamp(t0).isoformat(timespec="seconds"), datetime.now().isoformat(timespec="seconds")]
            ck.to_csv(ck_path, index=False)
            print(f" -> modelo {approved} APROVADO | RMSE val {err_val:.4f} | {secs} s")
        else:
            print(f" -> REPROVADO | RMSE val {err_val:.4f} | {secs} s")
        attempts_df.loc[len(attempts_df)] = [attempt, seed, err_val, ok, secs, datetime.now().isoformat(timespec="seconds")]
        attempts_df.to_csv(log_path, index=False)

        tmp.unlink(missing_ok=True)
        del mggp, best
        gc.collect()

    if approved < a.runs:
        print(f"\n[E2] stopped at {approved}/{a.runs} approved after {attempt} attempts (max {max_attempts}).")

    # --- 2. Free-Run validation of every approved model on the 5 tracks (resumable per model)
    print(f"\n[E2] validating {int((ck['validacao'] != 'Concluído').sum())} pending model(s) on {len(TRACKS)} tracks…")
    for idx in ck.index[ck["validacao"] != "Concluído"]:
        name = ck.at[idx, "modelo"]
        path = out / "models" / f"{name}.pkl"
        if not path.exists():
            print(f"    ! {name}: file missing, skipped")
            continue
        with open(path, "rb") as f:
            model = dill.load(f)
        t0 = time.time()
        rows = validate_model(model, name, a.arm, a.beta_unit, params, out, a.max_samples)
        rep = read_csv(rep_path, list(rows[0].keys()))
        rep = pd.concat([rep[rep["Modelo"] != name], pd.DataFrame(rows)], ignore_index=True)
        rep.to_csv(rep_path, index=False)
        ck.at[idx, "validacao"] = "Concluído"
        ck.to_csv(ck_path, index=False)
        print(f" -> {name} validado ({round(time.time() - t0)} s)")

    # --- 3. aggregate: mean ± std over models, per track
    if rep_path.exists():
        rep = pd.read_csv(rep_path)
        metrics = [c for c in rep.columns if c.startswith("RMSE_")]
        agg = rep[rep["ok"]].groupby("Pista")[metrics].agg(["mean", "std", "count"]).round(4)
        agg.to_csv(out / "summary_agg.csv")
        print("\n[E2] summary (approved & finite models):\n")
        print(agg.xs("mean", axis=1, level=1).to_string())
    print(f"\n[E2] done → {out}")


if __name__ == "__main__":
    main()
