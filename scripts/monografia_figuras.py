# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "pandas==2.3.3", "matplotlib==3.10.8", "openpyxl>=3.1"]
# ///
"""GOALS 5 item 5.4.1: vector figures for the monograph.

Time series from the partnership data go to <out>/dados/ (gitignored, CLAUDE.md); aggregated figures go to <out>/.

    uv run scripts/monografia_figuras.py --results D:/ProjetosPessoais/TCC/results/e2/t48 --data D:/ProjetosPessoais/TCC/Database
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CM = 1 / 2.54
W = 15 * CM
ROLES = ["train", "val", "testA", "testB", "testC"]
PISTA = {"train": "Wang 2.1\n(treino)", "val": "Wang 2.2\n(validação)", "testA": "Wang 3.1\n(teste A)",
         "testB": "Wang 8.1\n(teste B)", "testC": "Wang 3.2\n(teste C)"}
TEST = ["testA", "testB", "testC"]
VAR = ["A", "B_out", "B_atan2"]
ROT = {"A": "5×2 + atan2", "B_out": "5×3, saída 3", "B_atan2": "5×3 + atan2", "B": "5×3"}
COR = {"A": "#1f4e79", "B_out": "#7b3294", "B_atan2": "#2e7d32", "B": "#7b3294"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def check_final(res):
    estado = (Path(res) / "analise" / "ESTADO.txt").read_text(encoding="utf-8")
    if not estado.startswith("FINAL"):
        sys.exit(f"analise/ESTADO.txt não é FINAL — rode notebooks/E2_analise_beta.ipynb na pasta final:\n{estado}")


def _virgula(fig):
    """Vírgula decimal nos eixos numéricos (o texto do TCC está em português)."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",").replace("-", "−")))


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _virgula(fig)
    fig.savefig(path)
    plt.close(fig)
    print("  ", path)


def sinais_wang21(data, out):
    df = pd.read_excel(Path(data) / "wang21dv_bic_MGGP.xlsx", header=None)
    t = np.arange(len(df)) / 200
    beta = np.degrees(np.arctan2(df[17].to_numpy(float), df[18].to_numpy(float)))
    spec = [(14, r"$a_x$ [m/s²]"), (2, r"$a_y$ [m/s²]"), (22, r"$\omega_f$ [rad/s]"), (23, r"$\omega_r$ [rad/s]"),
            (11, r"$\dot\psi$ [rad/s]"), (18, r"$v_x$ [m/s]"), (17, r"$v_y$ [m/s]"), ("beta", r"$\beta$ [graus]")]
    # GOALS 6.3.1: 4 x 2, column-major (left a-d inputs, right e-h); beta is panel (h), no separate figure
    fig, axes = plt.subplots(4, 2, figsize=(W, 12 * CM), sharex=True, layout="constrained")
    for i, (ax, (c, lab)) in enumerate(zip(axes.T.flat, spec, strict=True)):
        ax.plot(t, beta if c == "beta" else df[c].to_numpy(float), lw=0.5, color="#1f4e79")
        ax.set_ylabel(lab, fontsize=8.5)
        ax.text(0.01, 0.92, f"({'abcdefgh'[i]})", transform=ax.transAxes, va="top", fontsize=9,
                bbox={"fc": "white", "ec": "none", "pad": 1})
        ax.margins(x=0)
    for ax in axes[-1]:
        ax.set_xlabel("tempo [s]")
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    save(fig, out / "dados" / "sinais_wang21.pdf")


def evolucao_ic(ic_tabelas, out):
    df = pd.read_csv(Path(ic_tabelas) / "dados_consolidados.csv")
    g = df.groupby(["nDelays", "treinamento"])["rmse_medio"].mean().reset_index()
    ordem = sorted(g["nDelays"].unique(), key=lambda s: len(s))
    fig, ax = plt.subplots(figsize=(W, 7 * CM))
    for nd, cor, m in zip(ordem, ["#b2182b", "#1f4e79", "#2e7d32"], ["o", "s", "^"], strict=True):
        s = g[g["nDelays"] == nd].sort_values("treinamento")
        ax.plot(s["treinamento"], s["rmse_medio"], marker=m, ms=3, lw=0.9, color=cor, label=f"nDelays = {nd}")
    ax.set_yscale("log")
    ticks = [0.2, 0.3, 0.5, 1, 2, 5, 10]
    ax.set_yticks(ticks, [f"{v:g}".replace(".", ",") for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("treinamento (1 a 49: nTerms e maxHeight crescentes)")
    ax.set_ylabel("RMSE médio [m/s] (escala log)")
    ax.set_xlim(0, 50)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, out / "evolucao_ic.pdf")


def boxplots(res, out):
    beta = pd.read_csv(Path(res) / "analise" / "por_modelo_beta.csv")
    vel = pd.read_csv(Path(res) / "analise" / "por_modelo_vel.csv")
    beta, vel = beta[beta["ok"]], vel[vel["ok"]]
    base = pd.read_csv(Path(res) / "analise" / "baselines.csv").set_index("role")

    fig, axes = plt.subplots(1, len(ROLES), figsize=(W, 6.5 * CM))
    for ax, role in zip(axes, ROLES, strict=True):
        dados = [beta[(beta["role"] == role) & (beta["variante"] == v)]["rmse"].to_numpy() for v in VAR]
        bp = ax.boxplot(dados, patch_artist=True, widths=0.6, medianprops={"color": "black"},
                        flierprops={"ms": 3})
        for patch, v in zip(bp["boxes"], VAR, strict=True):
            patch.set_facecolor(COR[v])
            patch.set_alpha(0.55)
        ax.axhline(base.loc[role, "rmse_beta_zero"], color="black", ls="--", lw=0.8)
        ax.set_xticks([])
        ax.set_title(PISTA[role], fontsize=8.5)
    axes[0].set_ylabel(r"RMSE de $\beta$ [graus]")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COR[v], alpha=0.55) for v in VAR]
    handles.append(plt.Line2D([], [], color="black", ls="--", lw=0.8))
    fig.legend(handles, [ROT[v] for v in VAR] + [r"$\hat\beta = 0$"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    save(fig, out / "rmse_beta_boxplot.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(W, 6.5 * CM))
    for ax, (col, lab) in zip(axes, [("rmse_vx", r"RMSE de $v_x$ [m/s]"), ("rmse_vy", r"RMSE de $v_y$ [m/s]")],
                              strict=True):
        pos, dados, cores = [], [], []
        for i, role in enumerate(ROLES):
            for j, arm in enumerate(["5x2", "5x3"]):
                pos.append(i * 3 + j)
                dados.append(vel[(vel["role"] == role) & (vel["arm"] == arm)][col].to_numpy())
                cores.append(COR["A"] if arm == "5x2" else COR["B"])
        bp = ax.boxplot(dados, positions=pos, patch_artist=True, widths=0.8, medianprops={"color": "black"},
                        flierprops={"ms": 3})
        for patch, c in zip(bp["boxes"], cores, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(ROLES))], [PISTA[r].split("\n")[0] for r in ROLES], rotation=30)
        ax.set_ylabel(lab)
        # one 5x3 model has a huge error on Wang 2.1 (limitations section): clip it and say so on the plot
        top = np.nanpercentile(vel[col], 99) * 1.15
        extremo = vel[col] > top
        ax.set_ylim(0, top)
        if extremo.any():
            onde = ", ".join(sorted({PISTA[r].split("\n")[0] for r in vel.loc[extremo, "role"]}))
            ax.text(0.02, 0.97, f"{int(extremo.sum())} valor(es) fora da escala ({onde})", transform=ax.transAxes,
                    va="top", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COR[c], alpha=0.55) for c in ("A", "B")]
    fig.legend(handles, ["5×2", "5×3"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    save(fig, out / "rmse_vxvy_boxplot.pdf")


def sobreposicao(res, out):
    beta = pd.read_csv(Path(res) / "analise" / "por_modelo_beta.csv")
    beta = beta[beta["ok"]]
    dificil = beta[(beta["variante"] == "A") & beta["role"].isin(TEST)].groupby("role")["rmse"].median().idxmax()

    def mediano(arm, var):
        s = beta[(beta["arm"] == arm) & (beta["variante"] == var) & (beta["role"] == dificil)]
        return s.iloc[(s["rmse"] - s["rmse"].median()).abs().argsort().iloc[0]]["modelo"]

    mA, mB = mediano("5x2", "A"), mediano("5x3", "B_out")
    zA = np.load(Path(res) / "5x2" / "predictions" / f"{mA}.npz", allow_pickle=False)
    zB = np.load(Path(res) / "5x3" / "predictions" / f"{mB}.npz", allow_pickle=False)
    # each model drops its own max lag at the start: align both on the last n samples of the track
    n = min(len(zA[f"{dificil}_yd"]), len(zB[f"{dificil}_yd"]))
    yd = zA[f"{dificil}_yd"][-n:]
    t = np.arange(len(yd)) / 200
    b = np.degrees(np.arctan2(yd[:, 1], yd[:, 0]))
    bA = np.degrees(np.arctan2(zA[f"{dificil}_yp"][-n:, 1], zA[f"{dificil}_yp"][-n:, 0]))
    bB = zB[f"{dificil}_yp"][-n:, 2]
    fig, axes = plt.subplots(2, 1, figsize=(W, 9 * CM), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, b, color="black", lw=0.8, label=r"$\beta$ de referência")
    axes[0].plot(t, bA, color=COR["A"], lw=0.7, ls="--", label="5×2 + atan2 (modelo mediano)")
    axes[0].plot(t, bB, color=COR["B_out"], lw=0.7, label="5×3, saída 3 (modelo mediano)")
    axes[0].set_ylabel(r"$\beta$ [graus]")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)  # fora das curvas
    axes[1].plot(t, b - bA, color=COR["A"], lw=0.7, ls="--")
    axes[1].plot(t, b - bB, color=COR["B_out"], lw=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("erro [graus]")
    axes[1].set_xlabel("tempo [s]")
    for ax in axes:
        ax.margins(x=0)
    save(fig, out / "dados" / "sobreposicao_beta.pdf")
    print(f"   pista mais difícil para o A: {dificil}; modelos medianos {mA} (5x2) e {mB} (5x3)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results/e2/t48"))
    ap.add_argument("--data", default=str(ROOT / "Database"))
    ap.add_argument("--ic-tabelas", default="D:/ProjetosPessoais/IC/TreinamentoMGGP_2_0/Relatorio_resultados/tabelas")
    ap.add_argument("--out", default=str(ROOT / "monografia/figuras"))
    a = ap.parse_args()
    out = Path(a.out)
    check_final(a.results)
    sinais_wang21(a.data, out)
    evolucao_ic(a.ic_tabelas, out)
    boxplots(a.results, out)
    sobreposicao(a.results, out)


if __name__ == "__main__":
    main()
