# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "deap==1.4.3", "numpy==2.2.6", "pandas==2.3.3", "scipy==1.15.3", "scikit-learn==1.7.2",
#   "matplotlib==3.10.8", "numba==0.64.0", "tqdm==4.67.3", "joblib==1.5.3", "ipython==8.38.0",
#   "openpyxl", "dill",
# ]
# ///
"""GOALS 3 — E2 campaign: N independent MGGP trainings per arm (5x2 baseline and 5x3), in parallel.

Port of the IC's `TrainingMGGP_LOOP.ipynb` to a resumable, parallel CLI.

    driver (default)  builds one job queue for every arm in --arms and keeps --workers training
                      subprocesses busy (default: every logical CPU). Each job = one independent
                      evolution with its own seed; the vendored lib is single-threaded, so this is
                      where the CPU goes to 100 %.
    worker            one attempt: train on Wang 2.1, gate on Wang 2.2 Free-Run
                      (0 < RMSE < --gate-max, the IC rule), and — if approved — validate Free-Run on
                      the 5 tracks, save the model (dill), the predictions (.npz) and a figure.

Rules per arm (campaign):
  * goal: --runs approved models;
  * an attempt "fails" when its Wang 2.2 Free-Run explodes (inf/NaN/≥ gate) or crashes;
  * --max-consecutive-fail (5) consecutive failures ⇒ the campaign is FRACASSO: nothing else is
    launched for it and its in-flight jobs are terminated (no compute spent on a config that keeps
    blowing up). A success resets the counter (not cumulative). "Consecutive" is in order of
    completion, which is the only well-defined order when jobs run in parallel;
  * never over-launch: in-flight ≤ runs − approved, so no approved-beyond-target compute is wasted;
  * Ctrl+C terminates the workers; re-running the same command resumes from attempts.csv.

Outputs: results/e2/<config>/<arm>/{models,fig,predictions}/modelo_rmse_N.*, attempts.csv,
relatorio_validacao_rmse.csv, summary_agg.csv, STATUS.txt, attempts/attempt_XXX/log.txt.

    uv run scripts/run_e2_mimo3.py --config t48 --arms 5x2,5x3 --runs 30            # the real thing
    uv run scripts/run_e2_mimo3.py --arms 5x3 --runs 2 --workers 4 --generations 2 \\
        --population 10 --max-samples 800 --out results/smoke                       # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
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

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 (β, →, ±)
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
GATE_TRACK = "Wang 2.2 (Validação)"
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
ATTEMPT_COLS = ["attempt", "seed", "status", "rmse_val", "seconds", "model", "started", "finished", "error"]
FINAL = {"approved", "rejected", "crashed", "cancelled"}
WORKER_ENV = {  # one single-threaded process per attempt; no BLAS/numba oversubscription
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    "TQDM_DISABLE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
}


# ----------------------------------------------------------------------------- data / metrics
def build_params(config: str, generations: int | None, population: int | None) -> dict:
    p = {**FIXED, **CONFIGS[config]}
    if generations:
        p["generations"] = generations
    if population:
        p["populationSize"] = population
    return p


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


def free_run(model, y: np.ndarray, u: np.ndarray, params: dict):
    args = (y, u) if params["evaluationTypeTest"] != "MShooting" else (params["k"], y, u)
    yp, yd = model.predict(params["evaluationTypeTest"], *args)
    n = min(len(yp), len(yd))
    return yp[:n], yd[:n]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- worker
def validate_model(model, label: str, arm: str, beta_unit: str, params: dict, out: Path,
                   max_samples: int | None) -> list[dict]:
    """Free-Run on the 5 tracks → rows (RMSE per output, evolved β and post-hoc atan2 β), fig.png, predictions.npz."""
    rows, preds = [], {}
    n_out = 3 if arm == "5x3" else 2
    fig, axes = plt.subplots(len(TRACKS), 3, figsize=(20, 4 * len(TRACKS)))
    fig.suptitle(f"Free-Run validation — {label} ({arm}, β in {beta_unit})", fontsize=15, y=1.0)

    for i, (track, (file, role)) in enumerate(TRACKS.items()):
        u, y = load_track(file, arm, beta_unit, max_samples)
        try:
            yp, yd = free_run(model, y, u, params)
            ok = bool(np.all(np.isfinite(yp)) and np.abs(yp).max() < 1e4)
        except Exception as e:  # divergence / LinAlg — record and move on, like the notebook
            print(f"    ! {track}: {type(e).__name__}: {e}")
            yp, yd, ok = np.zeros_like(y), y, False
        preds[f"{role}_yd"], preds[f"{role}_yp"] = yd, yp

        beta_true = to_rad(yd[:, 2], beta_unit) if n_out == 3 else np.arctan2(yd[:, 1], yd[:, 0])
        beta_post = np.arctan2(yp[:, 1], yp[:, 0])  # β from the predicted velocities (arm A logic)
        row = {
            "Modelo": label, "Pista": track, "Papel": role, "ok": ok,
            "RMSE_Vx": rmse(yd[:, 0], yp[:, 0]) if ok else np.nan,
            "RMSE_Vy": rmse(yd[:, 1], yp[:, 1]) if ok else np.nan,
            "RMSE_beta_atan2_deg": np.degrees(rmse(beta_true, beta_post)) if ok else np.nan,
        }
        if n_out == 3:
            beta_out = to_rad(yp[:, 2], beta_unit)
            row["RMSE_beta_out_deg"] = np.degrees(rmse(beta_true, beta_out)) if ok else np.nan
        row["RMSE_Medio_VxVy"] = (row["RMSE_Vx"] + row["RMSE_Vy"]) / 2
        rows.append(row)

        for j, (lbl, col) in enumerate([("Vx [m/s]", 0), ("Vy [m/s]", 1)]):
            ax = axes[i, j]
            ax.plot(yd[:, col], color="black", lw=0.8, label="Real")
            ax.plot(yp[:, col], color="tab:blue" if j == 0 else "tab:red", lw=0.8, ls="--", label="Previsto")
            ax.set_title(f"{track} — {lbl} | RMSE {row['RMSE_Vx' if j == 0 else 'RMSE_Vy']:.4f}")
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
    fig.savefig(out / "fig.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out / "predictions.npz", **preds)
    return rows


def worker(a: argparse.Namespace) -> None:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = build_params(a.config, a.generations, a.population)
    random.seed(a.seed)
    np.random.seed(a.seed)
    res = {"attempt": a.attempt, "seed": a.seed, "arm": a.arm, "approved": False, "rmse_val": None, "error": None}
    t0 = time.time()
    print(f"[worker] {a.arm} attempt {a.attempt} seed {a.seed} started {now()}")
    try:
        u_train, y_train = load_track(TRACKS["Wang 2.1 (Treino)"][0], a.arm, a.beta_unit, a.max_samples)
        u_val, y_val = load_track(TRACKS[GATE_TRACK][0], a.arm, a.beta_unit, a.max_samples)
        mggp = MGGP(inputs=u_train, outputs=y_train, validation=(u_val, y_val),
                    filename=str(out / "temp_mggp.pkl"), **params)
        mggp.run()
        best = mggp._hof[0]
        try:
            yp, yd = free_run(best, y_val, u_val, params)
            err = float(best.score(yd, yp, params["evaluationMode"]))
        except Exception as e:  # explosion inside the predictor counts as a failed attempt
            err, res["error"] = float("inf"), f"gate: {type(e).__name__}: {e}"
        res["rmse_val"] = err if np.isfinite(err) else None
        res["approved"] = bool(np.isfinite(err) and 0.0 < err < a.gate_max)
        print(f"[worker] gate {GATE_TRACK}: RMSE {err:.4f} → {'APROVADO' if res['approved'] else 'REPROVADO'}")
        if res["approved"]:
            with open(out / "model.pkl", "wb") as f:
                dill.dump(best, f)
            (out / "equation.txt").write_text(f"{best}\ntheta:\n{best._theta}\n", encoding="utf-8")
            rows = validate_model(best, f"attempt_{a.attempt:03d}", a.arm, a.beta_unit, params, out, a.max_samples)
            pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"[worker] CRASH {res['error']}")
    res["seconds"] = round(time.time() - t0)
    (out / "temp_mggp.pkl").unlink(missing_ok=True)
    (out / "result.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"[worker] finished in {res['seconds']} s")


# ----------------------------------------------------------------------------- driver
class Campaign:
    """One arm: its attempt ledger, its approved-model counter and its consecutive-failure counter."""

    def __init__(self, arm: str, root: Path, a: argparse.Namespace):
        self.arm, self.a = arm, a
        self.dir = root / arm
        for sub in ("attempts", "models", "fig", "predictions"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "attempts.csv"
        self.df = pd.read_csv(self.csv) if self.csv.exists() else pd.DataFrame(columns=ATTEMPT_COLS)
        stale = ~self.df["status"].isin(FINAL)  # launched by a previous driver that died → relaunch
        for k in self.df.loc[stale, "attempt"]:
            shutil.rmtree(self.dir / "attempts" / f"attempt_{int(k):03d}", ignore_errors=True)
        self.df = self.df[~stale].reset_index(drop=True)
        self.next_attempt = int(self.df["attempt"].max()) + 1 if len(self.df) else 1
        self.approved = int((self.df["status"] == "approved").sum())
        self.consec_fail = self._trailing_failures()
        self.inflight: dict[int, tuple[subprocess.Popen, object]] = {}
        self.state = "FRACASSO" if self.consec_fail >= a.max_consecutive_fail else (
            "DONE" if self.approved >= a.runs else "RUNNING")
        self._write_status()

    def _trailing_failures(self) -> int:
        n = 0
        for st in self.df.sort_values("finished")["status"][::-1]:
            if st in ("rejected", "crashed"):
                n += 1
            elif st == "approved":
                break
        return n

    def need(self) -> int:
        return max(0, self.a.runs - self.approved - len(self.inflight)) if self.state == "RUNNING" else 0

    def launch(self) -> None:
        k = self.next_attempt
        self.next_attempt += 1
        seed = self.a.seed + k  # same seed for the same attempt index in every arm (paired design)
        adir = self.dir / "attempts" / f"attempt_{k:03d}"
        adir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--arm", self.arm, "--config", self.a.config,
               "--attempt", str(k), "--seed", str(seed), "--out-dir", str(adir), "--beta-unit", self.a.beta_unit,
               "--gate-max", str(self.a.gate_max)]
        for flag in ("generations", "population", "max_samples"):
            if getattr(self.a, flag):
                cmd += [f"--{flag.replace('_', '-')}", str(getattr(self.a, flag))]
        log = open(adir / "log.txt", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, **WORKER_ENV})
        self.inflight[k] = (proc, log)
        self.df.loc[len(self.df)] = [k, seed, "running", np.nan, np.nan, "", now(), "", ""]
        self._save()
        print(f"[{self.arm}] ▶ attempt {k:03d} (seed {seed}) launched · {len(self.inflight)} in flight")

    def poll(self) -> bool:
        """Collect finished workers. Returns True if anything changed."""
        changed = False
        for k, (proc, log) in list(self.inflight.items()):
            if proc.poll() is None:
                continue
            log.close()
            del self.inflight[k]
            self._finish(k, proc.returncode)
            changed = True
        if self.state == "RUNNING" and self.approved >= self.a.runs and not self.inflight:
            self.state = "DONE"
            self._write_status()
        return changed

    def _finish(self, k: int, rc: int) -> None:
        adir = self.dir / "attempts" / f"attempt_{k:03d}"
        rj = adir / "result.json"
        res = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
        row = self.df.index[self.df["attempt"] == k][0]
        self.df.loc[row, ["finished", "seconds", "rmse_val"]] = [now(), res.get("seconds", np.nan), res.get("rmse_val", np.nan)]
        if not res:
            status, self.df.loc[row, "error"] = "crashed", f"worker exit code {rc} (no result.json)"
        elif res.get("approved"):
            status = "approved"
        elif res.get("error") and res.get("rmse_val") is None and "gate" not in str(res.get("error")):
            status, self.df.loc[row, "error"] = "crashed", res["error"]
        else:
            status, self.df.loc[row, "error"] = "rejected", res.get("error") or ""

        if status == "approved":
            self.approved += 1
            self.consec_fail = 0
            name = f"modelo_rmse_{self.approved}"
            self.df.loc[row, "model"] = name
            shutil.move(adir / "model.pkl", self.dir / "models" / f"{name}.pkl")
            shutil.move(adir / "fig.png", self.dir / "fig" / f"{name}.png")
            shutil.move(adir / "predictions.npz", self.dir / "predictions" / f"{name}.npz")
            val = pd.read_csv(adir / "validation.csv")
            val["Modelo"] = name
            rep_path = self.dir / "relatorio_validacao_rmse.csv"
            rep = pd.concat([pd.read_csv(rep_path), val], ignore_index=True) if rep_path.exists() else val
            rep.to_csv(rep_path, index=False)
            self._summary(rep)
            print(f"[{self.arm}] ✔ attempt {k:03d} APROVADO → {name} · RMSE val {res['rmse_val']:.4f} · "
                  f"{res['seconds']} s · {self.approved}/{self.a.runs}")
        else:
            self.consec_fail += 1
            print(f"[{self.arm}] ✘ attempt {k:03d} {status.upper()} · RMSE val {res.get('rmse_val')} · "
                  f"consecutive failures {self.consec_fail}/{self.a.max_consecutive_fail}")
            if self.consec_fail >= self.a.max_consecutive_fail and self.state == "RUNNING":
                self.state = "FRACASSO"
                print(f"[{self.arm}] ■ FRACASSO: {self.consec_fail} consecutive failures — stopping this arm, "
                      f"terminating {len(self.inflight)} in-flight job(s)")
                self.terminate("cancelled")
        self.df.loc[row, "status"] = status
        self._save()
        self._write_status()

    def terminate(self, status: str) -> None:
        for k, (proc, log) in list(self.inflight.items()):
            if proc.poll() is None:
                proc.terminate()
            log.close()
            row = self.df.index[self.df["attempt"] == k][0]
            self.df.loc[row, ["status", "finished"]] = [status, now()]
        self.inflight.clear()
        self._save()
        self._write_status()

    def _summary(self, rep: pd.DataFrame) -> None:
        metrics = [c for c in rep.columns if c.startswith("RMSE_")]
        ok = rep[rep["ok"].astype(bool)]
        if len(ok):
            ok.groupby("Pista")[metrics].agg(["mean", "std", "count"]).round(4).to_csv(self.dir / "summary_agg.csv")

    def _save(self) -> None:
        self.df.to_csv(self.csv, index=False)

    def _write_status(self) -> None:
        counts = self.df["status"].value_counts().to_dict()
        (self.dir / "STATUS.txt").write_text(
            f"{self.state}\narm={self.arm} approved={self.approved}/{self.a.runs} consecutive_failures={self.consec_fail} "
            f"in_flight={len(self.inflight)} attempts={counts}\nupdated={now()}\n", encoding="utf-8")

    def line(self) -> str:
        c = self.df["status"].value_counts().to_dict()
        return (f"{self.arm}: {self.state} · {self.approved}/{self.a.runs} approved · {len(self.inflight)} running · "
                f"{c.get('rejected', 0)} rejected · {c.get('crashed', 0)} crashed · consec-fail {self.consec_fail}")


def driver(a: argparse.Namespace) -> None:
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in ("5x2", "5x3"):
            sys.exit(f"unknown arm {arm!r} (use 5x2, 5x3)")
    workers = a.workers or os.cpu_count() or 1
    root = a.out or ROOT / "results" / "e2" / a.config
    root.mkdir(parents=True, exist_ok=True)
    params = build_params(a.config, a.generations, a.population)
    meta = {"config": a.config, "arms": arms, "runs": a.runs, "max_consecutive_fail": a.max_consecutive_fail,
            "workers": workers, "beta_unit": a.beta_unit, "beta_def": "atan2(col17, col18)", "gate": GATE_TRACK,
            "gate_max": a.gate_max, "input_cols": INPUT_COLS, "vx_col": VX_COL, "vy_col": VY_COL, "base_seed": a.seed,
            "max_samples": a.max_samples, "params": params, "lib": "src/ vendored @7514bfb", "started": now()}
    (root / "params.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(list(params.items()), columns=["Parametro", "Valor"]).to_csv(root / "parametros_utilizados.csv", index=False)

    camps = [Campaign(arm, root, a) for arm in arms]
    print(f"[E2] {a.config} · arms {arms} · {a.runs} runs each · {workers} workers · β in {a.beta_unit} → {root}")
    for c in camps:
        print("[E2] resume:", c.line())

    last_heartbeat = 0.0
    try:
        while any(c.state == "RUNNING" for c in camps):
            running = sum(len(c.inflight) for c in camps)
            for c in camps:  # fill free workers, first arm first
                while running < workers and c.need() > 0:
                    c.launch()
                    running += 1
            changed = any(c.poll() for c in camps)
            if changed or time.time() - last_heartbeat > 600:
                print(f"[E2 {datetime.now():%H:%M}] " + " | ".join(c.line() for c in camps))
                last_heartbeat = time.time()
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print("\n[E2] Ctrl+C — terminating workers; re-run the same command to resume")
        for c in camps:
            c.terminate("running")  # left non-final on purpose → relaunched on resume
        sys.exit(130)

    print("\n[E2] finished:")
    for c in camps:
        print("   ", c.line())
        sp = c.dir / "summary_agg.csv"
        if sp.exists():
            agg = pd.read_csv(sp, header=[0, 1], index_col=0)
            print(agg.xs("mean", axis=1, level=1).to_string(), "\n")


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", choices=CONFIGS, default="t48")
    common.add_argument("--beta-unit", choices=["deg", "rad"], default="deg",
                        help="β unit for the 5x3 target; the lib's fitness is the plain mean of per-output RMSE, "
                             "so rad (std≈0.04) would make β invisible next to vx (std≈3.6 m/s)")
    common.add_argument("--gate-max", type=float, default=100.0, help="approve if 0 < RMSE(Wang 2.2 Free-Run) < this")
    common.add_argument("--generations", type=int, default=None, help="override (smoke tests)")
    common.add_argument("--population", type=int, default=None, help="override (smoke tests)")
    common.add_argument("--max-samples", type=int, default=None, help="truncate every track (smoke tests only)")

    d = sub.add_parser("driver", parents=[common], help="run the campaign(s) (default)")
    d.add_argument("--arms", default="5x2,5x3", help="comma-separated: 5x2, 5x3")
    d.add_argument("--runs", type=int, default=30, help="approved models to reach, per arm")
    d.add_argument("--max-consecutive-fail", type=int, default=5)
    d.add_argument("--workers", type=int, default=None, help="parallel trainings (default: all logical CPUs)")
    d.add_argument("--seed", type=int, default=2026, help="attempt k uses seed+k in every arm")
    d.add_argument("--out", type=Path, default=None, help="default: results/e2/<config>")
    d.add_argument("--poll", type=float, default=10.0, help="seconds between checks on the workers")

    w = sub.add_parser("worker", parents=[common], help="one attempt (launched by the driver)")
    w.add_argument("--arm", choices=["5x2", "5x3"], required=True)
    w.add_argument("--attempt", type=int, required=True)
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--out-dir", required=True)

    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["driver", *argv]
    a = ap.parse_args(argv)
    worker(a) if a.cmd == "worker" else driver(a)


if __name__ == "__main__":
    main()
