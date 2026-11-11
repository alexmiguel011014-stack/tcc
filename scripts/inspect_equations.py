# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "pandas==2.3.3", "sympy>=1.13"]
# ///
"""GOALS 3 (decision 5.0.4 c): physical reading of the β output (output 3) of the 5x3 models.

Each gene of `attempts/attempt_XXX/equation.txt` is expanded symbolically (add, subtraction, mul; qd = delay of d
samples; a bare terminal is the previous sample, so lag = 1 + sum of the delays — the IC report's convention).
Variables: y1 = v_x, y2 = v_y, y3 = β (deg), u1 = a_x, u2 = a_y, u3/u4 = front/rear wheel speed, u5 = ψ̇.

For every model the script sums the coefficients of the linear monomials of each variable (any lag) and flags the
structures the kinematics predicts: persistence in β (Σθ of linear y3 ≈ 1), integration of a_y, the yaw-rate term
(T·180/π = 0.2865 deg per rad/s per sample), and a v_y-based reading of β (β ≈ v_y / v_x in rad).

    uv run scripts/inspect_equations.py --results D:/ProjetosPessoais/TCC/results/e2/t48
writes <results>/analise/estrutura_beta.csv (one row per model) and estrutura_beta_resumo.csv.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"y1": "vx", "y2": "vy", "y3": "beta", "u1": "ax", "u2": "ay", "u3": "w_f", "u4": "w_r", "u5": "r"}
T_DEG = 180 / np.pi / 200          # one sample of a rad/s rate, in degrees
_TOKEN = re.compile(r"\s*(add|subtraction|mul|q\d+|[uy]\d+|[(),])")


def _tokens(s):
    pos, out = 0, []
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"token inesperado em {s[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def expand(gene):
    """Gene text -> sympy expression over symbols like vy_1 (= v_y[k-1])."""
    toks = _tokens(gene)

    def parse(i, delay):
        t = toks[i]
        if re.fullmatch(r"[uy]\d+", t):
            return sp.Symbol(f"{NAMES[t]}_{delay + 1}"), i + 1
        if t.startswith("q"):
            arg, j = parse(i + 2, delay + int(t[1:]))
            return arg, j + 1                      # skip ')'
        a, j = parse(i + 2, delay)
        b, j = parse(j + 1, delay)                 # skip ','
        op = {"add": sp.Add, "subtraction": lambda x, y: x - y, "mul": sp.Mul}[t]
        return op(a, b), j + 1

    expr, _ = parse(0, 0)
    return sp.expand(expr)


def read_equation(path):
    head, _, tail = Path(path).read_text(encoding="utf-8").partition("theta:")
    parts = re.split(r"Output (\d+):", head)
    genes = {int(k): [t.strip() for t in b.strip().splitlines() if t.strip()] for k, b in zip(parts[1::2], parts[2::2], strict=True)}
    num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    thetas = [[float(x) for x in num.findall(b)] for b in re.findall(r"array\(\[(.*?)\]\)", tail, re.S)]
    return genes, thetas


def linear_sums(genes, theta):
    """Σθ of the linear monomials per variable, and the total |θ| of nonlinear monomials."""
    total = sp.Float(theta[0])
    for g, th in zip(genes[1:], theta[1:], strict=True):
        total += th * expand(g)
    lin, nonlin, by_lag = defaultdict(float), 0.0, defaultdict(dict)
    for term, coef in sp.expand(total).as_coefficients_dict().items():
        syms = term.free_symbols
        if not syms:
            continue
        if len(syms) == 1 and sp.degree(term, list(syms)[0]) == 1:
            var, lag = str(list(syms)[0]).rsplit("_", 1)
            lin[var] += float(coef)
            by_lag[var][int(lag)] = by_lag[var].get(int(lag), 0.0) + float(coef)
        else:
            nonlin += abs(float(coef))
    return lin, nonlin, by_lag


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results/e2/t48"))
    a = ap.parse_args()
    res = Path(a.results)
    led = pd.read_csv(res / "5x3" / "attempts.csv")
    led = led[led["status"] == "approved"]

    rows = []
    for _, r in led.iterrows():
        genes, thetas = read_equation(res / "5x3" / "attempts" / f"attempt_{int(r.attempt):03d}" / "equation.txt")
        lin, nonlin, by_lag = linear_sums(genes[3], thetas[2])
        vy_lags = by_lag.get("vy", {})
        rows.append({
            "modelo": r.model, "attempt": int(r.attempt),
            "persistencia_beta": lin.get("beta", 0.0),            # Σθ de β[k-i]
            "soma_ay": lin.get("ay", 0.0),                         # integração de a_y (deg por m/s²)
            "soma_r": lin.get("r", 0.0),                           # termo de guinada (deg por rad/s)
            "soma_vy": lin.get("vy", 0.0),                         # leitura via v_y (deg por m/s)
            "max_abs_coef_vy": max((abs(v) for v in vy_lags.values()), default=0.0),
            "soma_vx": lin.get("vx", 0.0),
            "nao_linear_abs": nonlin,
            "usa_ay": "ay" in lin, "usa_r": "r" in lin, "usa_vy": "vy" in lin,
        })
    df = pd.DataFrame(rows)
    out = res / "analise"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "estrutura_beta.csv", index=False)

    q = lambda c: f"{df[c].median():+.4g} [{df[c].quantile(.25):+.4g}; {df[c].quantile(.75):+.4g}]"  # noqa: E731
    resumo = pd.DataFrame([
        {"grandeza": "Σθ persistência β (esperado ≈ 1)", "mediana [IQR]": q("persistencia_beta"), "modelos com o termo": int((df.persistencia_beta != 0).sum())},
        {"grandeza": f"Σθ a_y (deg/(m/s²); cinemática ≈ {T_DEG:.4f}/v_x)", "mediana [IQR]": q("soma_ay"), "modelos com o termo": int(df.usa_ay.sum())},
        {"grandeza": f"Σθ ψ̇ (deg/(rad/s); cinemática ±{T_DEG:.4f})", "mediana [IQR]": q("soma_r"), "modelos com o termo": int(df.usa_r.sum())},
        {"grandeza": "Σθ v_y (deg/(m/s))", "mediana [IQR]": q("soma_vy"), "modelos com o termo": int(df.usa_vy.sum())},
        {"grandeza": "maior |θ| de um atraso de v_y", "mediana [IQR]": q("max_abs_coef_vy"), "modelos com o termo": int(df.usa_vy.sum())},
        {"grandeza": "Σ|θ| não lineares", "mediana [IQR]": q("nao_linear_abs"), "modelos com o termo": int((df.nao_linear_abs > 0).sum())},
    ])
    resumo.to_csv(out / "estrutura_beta_resumo.csv", index=False)
    print(f"{len(df)} modelos 5x3\n" + resumo.to_string(index=False))


if __name__ == "__main__":
    main()
