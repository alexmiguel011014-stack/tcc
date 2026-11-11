# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "pandas==2.3.3", "scipy>=1.13", "openpyxl>=3.1"]
# ///
"""Pós-análise da correção do TCC (itens 1 e 2 do bloco 0).

(1) Reavalia, SEM retreinar, os modelos 5×2 e 5×3 já treinados contra a referência com o deslocamento de v_y removido.
(2) Ajusta um modelo linear (ARX por mínimos quadrados, mesmas entradas e atrasos) como linha de base.

Escreve só tabelas agregadas em <results>/analise_extra/ (nada de séries da parceria).

    uv run scripts/e3_referencia_corrigida_e_linear.py --results D:/ProjetosPessoais/TCC/results/e2/t48 --data D:/ProjetosPessoais/TCC/Database
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
ROLES = ["train", "val", "testA", "testB", "testC"]
TEST = ["testA", "testB", "testC"]
FILES = {"train": "wang21", "val": "wang22", "testA": "wang31", "testB": "wang81", "testC": "wang32"}
U_COLS, LAGS = [14, 2, 22, 23, 11], [1, 2, 5, 10, 25, 50]
LIM = 1e4
RNG = np.random.default_rng(0)


def beta_deg(vy, vx):
    return np.degrees(np.arctan2(vy, vx))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    aj, cur = np.empty(len(p)), 0.0
    for i, j in enumerate(o):
        cur = max(cur, (len(p) - i) * p[j])
        aj[j] = min(1.0, cur)
    return aj


def metr(ref, est):
    e = est - ref
    r = float(np.sqrt(np.mean(e**2)))
    b = float(np.mean(e))
    return r, b, float(np.sqrt(max(r * r - b * b, 0.0)))


def load(data):
    d = {}
    for r, f in FILES.items():
        d[r] = pd.read_excel(data / f"{f}dv_bic_MGGP.xlsx", header=None).to_numpy(float)
    return d


def referencias(d, n_by_role):
    """Referência original (col. 17/18) e corrigida (v_y menos o deslocamento estimado contra a coluna β bruta).

    A janela avaliada de cada modelo descarta o maior atraso que ele usa; por isso o trecho é o final da pista (n amostras).
    """
    out = {}
    for r in ROLES:
        x = d[r]
        off = float(np.median(x[:, 17] - np.tan(np.radians(x[:, 3])) * x[:, 9])) if r in TEST else 0.0
        n = n_by_role[r]
        vx, vy = x[-n:, 18], x[-n:, 17]
        out[r] = dict(off=off, vx=vx, vy=vy, vy_c=vy - off, b=beta_deg(vy, vx), b_c=beta_deg(vy - off, vx))
    return out


def linha(arm, modelo, role, ref, yp, n_out):
    ok = bool(np.all(np.isfinite(yp)) and np.max(np.abs(yp)) < LIM)
    lin = dict(arm=arm, modelo=modelo, role=role, ok=ok)
    if not ok:
        return [dict(lin, variante=v) for v in ("A", "B_out", "B_atan2")] if n_out == 3 else [dict(lin, variante="A")]
    vars_ = {"A": beta_deg(yp[:, 1], yp[:, 0])} if n_out == 2 else {
        "B_out": yp[:, 2], "B_atan2": beta_deg(yp[:, 1], yp[:, 0])}
    rows = []
    for v, bh in vars_.items():
        ro, bo, so = metr(ref["b"], bh)
        rc, bc, sc = metr(ref["b_c"], bh)
        rows.append(dict(lin, variante=v, rmse_orig=ro, vies_orig=bo, rmse_semvies_orig=so,
                         rmse_corr=rc, vies_corr=bc, rmse_semvies_corr=sc,
                         rmse_vy_orig=metr(ref["vy"], yp[:, 1])[0], rmse_vy_corr=metr(ref["vy_c"], yp[:, 1])[0],
                         rmse_vx=metr(ref["vx"], yp[:, 0])[0]))
    return rows


def arx_design(Y, U, t0, fb=True):
    """Regressores no instante t: saídas e entradas nos atrasos LAGS (sem atraso 0) + constante."""
    cols = [np.ones(len(Y) - t0)]
    for L in LAGS:
        if fb:
            cols += [Y[t0 - L:len(Y) - L, k] for k in range(Y.shape[1])]
        cols += [U[t0 - L:len(U) - L, k] for k in range(U.shape[1])]
    return np.column_stack(cols)


def simula_arx(theta, Y0, U, n_out, fb=True):
    """Simulação livre a partir de Y0 (últimas 50 amostras reais antes do início), realimentando as saídas estimadas."""
    n = len(U) - 50
    Y = np.vstack([Y0, np.zeros((n, n_out))])
    for t in range(50, 50 + n):
        reg = [1.0]
        for L in LAGS:
            reg += (list(Y[t - L]) if fb else []) + list(U[t - L])
        Y[t] = np.asarray(reg) @ theta
        if not np.all(np.isfinite(Y[t])) or np.max(np.abs(Y[t])) > LIM:
            Y[t:] = np.nan
            break
    return Y[50:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/e2/t48")
    ap.add_argument("--data", default="Database")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.results) / "analise_extra"
    out.mkdir(exist_ok=True)
    d = load(Path(a.data))

    # ---- (1) reavaliação dos modelos já treinados -------------------------------------------------
    rows, refs = [], None
    for arm, n_out in (("5x2", 2), ("5x3", 3)):
        for f in sorted((res / arm / "predictions").glob("*.npz")):
            z = np.load(f)
            refs_m = referencias(d, {r: len(z[f"{r}_yd"]) for r in ROLES})
            if refs is None:
                refs = refs_m
            for r in ROLES:
                rows += linha(arm, f.stem, r, refs_m[r], z[f"{r}_yp"], n_out)
    por = pd.DataFrame(rows)
    por.to_csv(out / "reavaliacao_por_modelo.csv", index=False)
    pd.DataFrame([{"role": r, "deslocamento_vy_m_s": refs[r]["off"]} for r in ROLES]).to_csv(out / "deslocamentos.csv", index=False)
    print("deslocamento de v_y removido (m/s):", {r: round(refs[r]["off"], 3) for r in ROLES})

    ok = por[por["ok"]]
    cols = ["rmse_orig", "rmse_semvies_orig", "rmse_corr", "rmse_semvies_corr", "vies_orig", "vies_corr", "rmse_vy_orig", "rmse_vy_corr"]
    resumo = ok.groupby(["role", "variante"])[cols].median().reset_index()
    resumo["n"] = ok.groupby(["role", "variante"])["modelo"].count().to_numpy()
    resumo.to_csv(out / "reavaliacao_resumo.csv", index=False)

    # Q1 (Mann–Whitney contra A, Holm sobre as 6 comparações das pistas de teste), para cada referência
    testes = []
    for ref in ("orig", "corr"):
        for v in ("B_out", "B_atan2"):
            for r in TEST:
                x = ok[(ok.role == r) & (ok.variante == v)][f"rmse_{ref}"].to_numpy()
                y = ok[(ok.role == r) & (ok.variante == "A")][f"rmse_{ref}"].to_numpy()
                u = stats.mannwhitneyu(x, y, alternative="two-sided")
                boot = [np.median(RNG.choice(x, len(x))) - np.median(RNG.choice(y, len(y))) for _ in range(5000)]
                testes.append(dict(ref=ref, variante=v, role=r, n1=len(x), n2=len(y), dif_mediana=np.median(x) - np.median(y),
                                   ic_inf=np.percentile(boot, 2.5), ic_sup=np.percentile(boot, 97.5),
                                   cliff=2 * u.statistic / (len(x) * len(y)) - 1, p=u.pvalue))
    t = pd.DataFrame(testes)
    t["p_holm"] = np.nan
    for ref in ("orig", "corr"):
        idx = t.index[t.ref == ref]
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    t.to_csv(out / "reavaliacao_testes_q1.csv", index=False)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    print(resumo[resumo.role.isin(TEST)].to_string(index=False))
    print(t.to_string(index=False))

    # ---- (2) linha de base linear (ARX por mínimos quadrados) -------------------------------------
    lin = []
    for n_out, fb in ((2, True), (3, True), (2, False), (3, False)):
        def ys(r, n_out=n_out):
            x = d[r]
            Y = np.column_stack([x[:, 18], x[:, 17]] + ([beta_deg(x[:, 17], x[:, 18])] if n_out == 3 else []))
            return Y, x[:, U_COLS]
        Y, U = ys("train")
        t0 = 50
        theta, *_ = np.linalg.lstsq(arx_design(Y, U, t0, fb), Y[t0:], rcond=None)
        for r in ROLES:
            Yr, Ur = ys(r)
            n = len(refs[r]["vx"])
            ini = len(Yr) - n                       # mesmo trecho avaliado nos modelos MGGP
            Ys = simula_arx(theta, Yr[ini - 50:ini], Ur[ini - 50:], n_out, fb)
            yp = Ys[: n]
            ref = refs[r]
            ok_ = bool(np.all(np.isfinite(yp)) and len(yp) == n)
            base = dict(modelo=f"{'ARX' if fb else 'FIR'} linear 5x{n_out}", role=r, ok=ok_)
            if not ok_:
                lin.append(dict(base, variante="divergiu"))
                continue
            for v, bh in ({"A": beta_deg(yp[:, 1], yp[:, 0])} if n_out == 2 else
                          {"B_out": yp[:, 2], "B_atan2": beta_deg(yp[:, 1], yp[:, 0])}).items():
                ro, bo, so = metr(ref["b"], bh)
                rc, bc, sc = metr(ref["b_c"], bh)
                lin.append(dict(base, variante=v, rmse_beta_orig=ro, vies_orig=bo, rmse_beta_corr=rc, rmse_vx=metr(ref["vx"], yp[:, 0])[0],
                                rmse_vy=metr(ref["vy"], yp[:, 1])[0]))
    L = pd.DataFrame(lin)
    L.to_csv(out / "linear_arx.csv", index=False)
    print(L.to_string(index=False))


if __name__ == "__main__":
    main()
