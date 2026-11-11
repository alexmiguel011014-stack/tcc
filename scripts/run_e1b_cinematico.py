# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "pandas==2.3.3", "openpyxl>=3.1"]
# ///
"""GOALS 3 E1b: kinematic sideslip estimator, a physics-based reference with no training.

    β̂[k] = β̂[k-1] + T · (a_y[k-1] / v̂x[k-1] − s · ψ̇[k-1]),   β̂[0] = 0,   T = 1/200 s
    v̂x = c · u4 (rear wheel speed, rad/s), c fitted by least squares on Wang 2.1 only
    where v̂x < 1 m/s the estimate is held (β̂[k] = β̂[k-1])

The sign s of the yaw-rate term is chosen on Wang 2.1 only (the IC found the dataset's sign convention opposite to
the textbook); both signs are reported. Reference β = atan2(col 17, col 18).

    uv run scripts/run_e1b_cinematico.py --data D:/ProjetosPessoais/TCC/Database --out D:/ProjetosPessoais/TCC/results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TRACKS = {"train": "wang21", "val": "wang22", "testA": "wang31", "testB": "wang81", "testC": "wang32"}
T = 1 / 200
VX_MIN = 1.0


def load(data, name):
    df = pd.read_excel(Path(data) / f"{name}dv_bic_MGGP.xlsx", header=None)
    return {"ay": df[2].to_numpy(float), "r": df[11].to_numpy(float), "u4": df[23].to_numpy(float),
            "vx": df[18].to_numpy(float), "vy": df[17].to_numpy(float)}


def integrate(ay, r, vx_hat, s):
    beta = np.zeros(len(ay))
    for k in range(1, len(ay)):
        v = vx_hat[k - 1]
        beta[k] = beta[k - 1] if v < VX_MIN else beta[k - 1] + T * (ay[k - 1] / v - s * r[k - 1])
    return beta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "Database"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out) / "e1b"
    out.mkdir(parents=True, exist_ok=True)

    tracks = {role: load(a.data, name) for role, name in TRACKS.items()}
    tr = tracks["train"]
    c = float(np.dot(tr["vx"], tr["u4"]) / np.dot(tr["u4"], tr["u4"]))
    print(f"v̂x = {c:.4f} · u4 (ajustado na Wang 2.1)")

    rows = []
    for role, d in tracks.items():
        beta = np.arctan2(d["vy"], d["vx"])
        for s in (+1, -1):
            e = np.degrees(integrate(d["ay"], d["r"], c * d["u4"], s) - beta)
            rows.append({"pista": TRACKS[role], "papel": role, "sinal": s, "c_vx_u4": c,
                         "rmse_beta_deg": float(np.sqrt(np.mean(e**2))), "bias_deg": float(e.mean()),
                         "drift_deg_final": float(e[-1])})
    df = pd.DataFrame(rows)
    train = df[df["papel"] == "train"].set_index("sinal")["rmse_beta_deg"]
    s_best = int(train.idxmin())
    df["escolhido"] = df["sinal"] == s_best
    df.to_csv(out / "rmse_beta_cinematico.csv", index=False)
    print(f"sinal escolhido na Wang 2.1: s = {s_best:+d}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
