# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["deap==1.4.3", "numpy==2.2.6", "pandas==2.3.3", "scikit-learn==1.7.2", "numba==0.64.0",
#                 "tqdm==4.67.3", "openpyxl>=3.1", "dill>=0.3.8"]
# ///
"""GOALS 2 (reproduce the IC) + GOALS 3 E0/E1: Free-Run of the IC anchor models on the 5 Wang tracks.

Prints RMSE vx/vy per track, checks the IC's 4-track means (T48 0.2121, T19 0.2239 ± 0.005), and writes
  <out>/e0/predictions_<model>_<track>.csv   (vx, vy measured and predicted)   — partnership-derived, gitignored
  <out>/e0/rmse_ic.csv                        (one row per model × track)
  <out>/e1/rmse_sa.csv                        (β̂ = atan2(v̂y, v̂x) vs β = atan2(col 17, col 18), deg and rad)

    uv run scripts/reproduce_ic.py --data D:/ProjetosPessoais/TCC/Database --out D:/ProjetosPessoais/TCC/results
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import dill
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mggp import MGGP  # noqa: E402  (registers the DEAP creator classes needed by dill)

IC = Path("D:/ProjetosPessoais/IC/TreinamentoMGGP_2_0")
MODELS = {  # IC anchors (CLAUDE.md): name -> (pkl, nDelays, nTerms, maxHeight, IC 4-track mean)
    "T48": (IC / "RESULTADOS-3/Treinamento_48/TRAINING/modelo_rmse_9.pkl", [1, 2, 5, 10, 25, 50], 7, 6, 0.2121),
    "T19": (IC / "RESULTADOS-2/Treinamento_19/TRAINING/modelo_rmse_6.pkl", [1, 2, 5, 10], 3, 5, 0.2239),
}
TRACKS = {"train": "wang21", "val": "wang22", "testA": "wang31", "testB": "wang81", "testC": "wang32"}
IC_TRACKS = ["train", "val", "testA", "testB"]          # the IC's mean used these four
TOL = 0.005


def load_track(data, name):
    df = pd.read_excel(Path(data) / f"{name}dv_bic_MGGP.xlsx", header=None)
    return df.iloc[:, [14, 2, 22, 23, 11]].to_numpy(float), df.iloc[:, [18, 17]].to_numpy(float)


def load_model(path, n_delays, n_terms, max_height, u, y):
    """The pickled _funcs are stale: rebuild them with a pset whose arguments are renamed like in training."""
    with tempfile.TemporaryDirectory() as tmp:
        g = MGGP(inputs=u[:400], outputs=y[:400], nDelays=n_delays, nTerms=n_terms, maxHeight=max_height,
                 mode="MIMO", operators=["add", "subtraction", "mul"], k=300, filename=str(Path(tmp) / "x.pkl"))
    with open(path, "rb") as f:
        model = dill.load(f)
    g.element.compileModel(model)
    return model


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "Database"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "e0").mkdir(parents=True, exist_ok=True)
    (out / "e1").mkdir(parents=True, exist_ok=True)

    data = {role: load_track(a.data, name) for role, name in TRACKS.items()}
    rows, sa_rows, ok = [], [], True
    for mname, (pkl, nd, nt, mh, ic_mean) in MODELS.items():
        model = load_model(pkl, nd, nt, mh, *data["train"])
        for role, (u, y) in data.items():
            yp, yd = model.predict("FreeRun", y, u)
            n = min(len(yp), len(yd))
            yp, yd = yp[:n], yd[:n]
            r = {"modelo": mname, "pista": TRACKS[role], "papel": role,
                 "rmse_vx": rmse(yd[:, 0] - yp[:, 0]), "rmse_vy": rmse(yd[:, 1] - yp[:, 1])}
            r["rmse_medio"] = (r["rmse_vx"] + r["rmse_vy"]) / 2
            rows.append(r)
            pd.DataFrame({"vx": yd[:, 0], "vy": yd[:, 1], "vx_hat": yp[:, 0], "vy_hat": yp[:, 1]}).to_csv(
                out / "e0" / f"predictions_{mname}_{TRACKS[role]}.csv", index=False)
            b, bh = np.arctan2(yd[:, 1], yd[:, 0]), np.arctan2(yp[:, 1], yp[:, 0])
            e = (bh - b + np.pi) % (2 * np.pi) - np.pi
            sa_rows.append({"modelo": mname, "pista": TRACKS[role], "papel": role, "rmse_beta_rad": rmse(e),
                            "rmse_beta_deg": float(np.degrees(rmse(e))), "bias_beta_deg": float(np.degrees(e.mean()))})
        m4 = np.mean([r["rmse_medio"] for r in rows if r["modelo"] == mname and r["papel"] in IC_TRACKS])
        hit = abs(m4 - ic_mean) <= TOL
        ok &= hit
        print(f"{mname}: média 4 pistas da IC = {m4:.4f} (IC {ic_mean:.4f}) -> {'OK' if hit else 'DIVERGE'}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "e0" / "rmse_ic.csv", index=False)
    pd.DataFrame(sa_rows).to_csv(out / "e1" / "rmse_sa.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(pd.DataFrame(sa_rows).round(4).to_string(index=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
