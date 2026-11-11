# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pandas==2.3.3"]
# ///
"""GOALS 5 item 5.4.2: LaTeX tables for the monograph, generated from the analysis CSVs (never typed by hand).

    uv run scripts/monografia_tabelas.py --results D:/ProjetosPessoais/TCC/results/e2/t48
reads  <results>/analise/*.csv, <results>/params.json, and <results>/../../{e0,e1,e1b}/*.csv
writes monografia/tabelas/{parametros,pistas,divergencias,resumo_beta,resumo_vel,testes,referencias_beta,
       estrutura_beta}.tex
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
ROLES = ["train", "val", "testA", "testB", "testC"]
PISTA = {"train": "Wang 2.1 (treino)", "val": "Wang 2.2 (validação)", "testA": "Wang 3.1 (teste A)",
         "testB": "Wang 8.1 (teste B)", "testC": "Wang 3.2 (teste C)"}
TEST = ["testA", "testB", "testC"]
VAR = {"A": "5×2 + atan2", "B_out": "5×3, saída 3", "B_atan2": "5×3 + atan2", "B": "5×3"}
FONTE = r"\fonte{Elaboração própria.}"


def num(x, d=4):
    if pd.isna(x):
        return "--"
    return f"{x:.{d}f}".replace(".", ",").replace("-", "$-$")


def pval(p):
    if pd.isna(p):
        return "--"
    return "$<$0,0001" if p < 1e-4 else num(p, 4)


def table(caption, label, colspec, header, rows, nota=None, size=r"\small", resize=False):
    body = "\n".join("        " + " & ".join(r) + r" \\" for r in rows)
    nota = f"\n    \\par\\footnotesize Nota: {nota}" if nota else ""
    ini, fim = ("\\resizebox{\\textwidth}{!}{%\n    ", "}") if resize else ("", "")  # evita estouro da margem
    return (f"\\begin{{table}}[htbp]\n    \\centering\n    \\caption{{{caption}}}\n    \\label{{{label}}}\n"
            f"    {size}\n    {ini}\\begin{{tabular}}{{{colspec}}}\n        \\toprule\n        {header} \\\\\n"
            f"        \\midrule\n{body}\n        \\bottomrule\n    \\end{{tabular}}{fim}{nota}\n    {FONTE}\n\\end{{table}}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results/e2/t48"))
    ap.add_argument("--out", default=str(ROOT / "monografia/tabelas"))
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    an = res / "analise"
    estado = (an / "ESTADO.txt").read_text(encoding="utf-8")
    if not estado.startswith("FINAL"):
        sys.exit(f"analise/ESTADO.txt não é FINAL:\n{estado}")
    out.mkdir(parents=True, exist_ok=True)
    tabs = {}

    p = json.loads((res / "params.json").read_text(encoding="utf-8"))
    pp = p["params"]
    rows = [["Número de gerações", str(pp["generations"])], ["Tamanho da população", str(pp["populationSize"])],
            ["Atrasos (\\textit{nDelays})", str(pp["nDelays"])], ["Termos por saída (\\textit{nTerms})", str(pp["nTerms"])],
            ["Profundidade máxima (\\textit{maxHeight})", str(pp["maxHeight"])],
            ["Regime de avaliação no treinamento", f"\\textit{{Multi-shooting}} ($K = {pp['k']}$)"],
            ["Regime de avaliação no teste", "\\textit{Free-run}"],
            ["Taxa de mutação", num(pp["mutationRate"], 1)], ["Taxa de cruzamento", num(pp["crossoverRate"], 1)],
            ["Elitismo", f"{pp['elitePercentage']}\\%"], ["Operadores", "$+$, $-$, $\\times$"],
            ["Aptidão", "média simples do RMSE das saídas"],
            ["Unidade de $\\beta$ no braço 5×3", "graus"],
            ["Execuções independentes por braço", f"{p['runs']} (semente $= {p['base_seed']} +$ tentativa)"],
            ["Critério de aprovação", f"$0 < \\mathrm{{RMSE}}(\\text{{Wang 2.2}}, \\textit{{Free-run}}) < {p['gate_max']:g}$"],
            ["Regra de fracasso", f"{p['max_consecutive_fail']} reprovações seguidas"]]
    tabs["parametros"] = table("Parâmetros do MGGP (configuração T48 da IC) e do protocolo de treinamento.",
                               "tab:parametros", "ll", r"\textbf{Parâmetro} & \textbf{Valor}", rows)

    base = pd.read_csv(an / "baselines.csv").set_index("role")
    rows = [[PISTA[r], {"train": "treinamento", "val": "validação (aprovação)"}.get(r, "teste"),
             f"{int(base.loc[r, 'amostras']):d}", num(base.loc[r, "amostras"] / 200, 1),
             num(base.loc[r, "beta_media"], 2), num(base.loc[r, "beta_dp"], 2), num(base.loc[r, "rmse_beta_zero"], 2),
             num(base.loc[r, "rmse_beta_media_treino"], 2)] for r in ROLES]
    tabs["pistas"] = table(
        "Pistas da base de dados, papel no experimento e escala do ângulo de deriva de referência.", "tab:pistas",
        "llrrrrrr", r"\textbf{Pista} & \textbf{Papel} & \textbf{Amostras} & \textbf{Duração [s]} & "
        r"$\bar\beta$ [°] & $\sigma_\beta$ [°] & \multicolumn{2}{c}{RMSE de linha de base [°]}", rows,
        nota=r"amostras após descartar o maior atraso do modelo (200~Hz). Linhas de base: $\hat\beta = 0$ e "
             r"$\hat\beta$ = média de $\beta$ na pista de treino.", size=r"\footnotesize", resize=True)

    div = pd.read_csv(an / "divergencias.csv")
    rows = []
    for r in ROLES:
        d = div[div["role"] == r].set_index("arm")
        rows.append([PISTA[r], f"{int(d.loc['5x2', 'divergiram'])}/{int(d.loc['5x2', 'modelos'])}",
                     f"{int(d.loc['5x3', 'divergiram'])}/{int(d.loc['5x3', 'modelos'])}"])
    tabs["divergencias"] = table("Pares modelo × pista cuja simulação livre divergiu (excluídos da estatística da pista).",
                                 "tab:divergencias", "lcc", r"\textbf{Pista} & \textbf{5×2} & \textbf{5×3}", rows)

    rp = pd.read_csv(an / "resumo_por_pista.csv")
    rb = rp[rp["metrica"].str.startswith("RMSE_beta")]
    rows = []
    for r in ROLES:
        for v in ["A", "B_out", "B_atan2"]:
            s = rb[(rb["role"] == r) & (rb["variante"] == v)].iloc[0]
            rows.append([PISTA[r] if v == "A" else "", VAR[v], str(int(s["n"])), num(s["mediana"], 3),
                         f"[{num(s['q1'], 3)}; {num(s['q3'], 3)}]", num(s["media"], 3), num(s["dp"], 3)])
    tabs["resumo_beta"] = table("RMSE de $\\beta$ (graus) por pista e variante: mediana, intervalo interquartil, média e "
                                "desvio-padrão entre os modelos.", "tab:resumo_beta", "llrcccc",
                                r"\textbf{Pista} & \textbf{Variante} & $n$ & \textbf{Mediana} & \textbf{[Q1; Q3]} & "
                                r"\textbf{Média} & \textbf{DP}", rows, size=r"\footnotesize")

    rv = rp[~rp["metrica"].str.startswith("RMSE_beta")]
    rows = []
    for r in ROLES:
        for v in ["A", "B"]:
            sx = rv[(rv["role"] == r) & (rv["variante"] == v) & (rv["metrica"] == "RMSE_Vx")].iloc[0]
            sy = rv[(rv["role"] == r) & (rv["variante"] == v) & (rv["metrica"] == "RMSE_Vy")].iloc[0]
            rows.append([PISTA[r] if v == "A" else "", "5×2" if v == "A" else "5×3", str(int(sx["n"])),
                         num(sx["mediana"], 3), f"[{num(sx['q1'], 3)}; {num(sx['q3'], 3)}]",
                         num(sy["mediana"], 3), f"[{num(sy['q1'], 3)}; {num(sy['q3'], 3)}]"])
    tabs["resumo_vel"] = table("RMSE de $v_x$ e $v_y$ (m/s) por pista e braço: mediana e intervalo interquartil entre os "
                               "modelos.", "tab:resumo_vel", "llrcccc",
                               r"\textbf{Pista} & \textbf{Braço} & $n$ & \multicolumn{2}{c}{$v_x$} & "
                               r"\multicolumn{2}{c}{$v_y$}", rows, size=r"\footnotesize")

    te = pd.read_csv(an / "testes.csv")
    te = te[te["confirmatorio"]]
    nome = {("RMSE_beta", "B_out", "A"): r"$\beta$: 5×3 saída 3 $-$ 5×2",
            ("RMSE_beta", "B_atan2", "A"): r"$\beta$: 5×3 atan2 $-$ 5×2",
            ("RMSE_Vx", "B", "A"): r"$v_x$: 5×3 $-$ 5×2", ("RMSE_Vy", "B", "A"): r"$v_y$: 5×3 $-$ 5×2",
            ("RMSE_beta", "B_out", "B_atan2"): r"$\beta$: saída 3 $-$ atan2 (5×3)"}
    rows = []
    for fam in ["Q1", "Q2", "Q3"]:
        for _, t in te[te["familia"] == fam].iterrows():
            rows.append([fam, nome[(t["metrica"], t["g1"], t["g2"])], PISTA[t["pista"]].split(" (")[0],
                         f"{int(t['n1'])}/{int(t['n2'])}", num(t["dif_mediana"], 3),
                         f"[{num(t['ic95_inf'], 3)}; {num(t['ic95_sup'], 3)}]", num(t["efeito"], 2),
                         pval(t["p_holm"])])
    tabs["testes"] = table(
        "Testes confirmatórios nas pistas de teste: diferença de medianas com IC95\\% \\textit{bootstrap}, tamanho de "
        "efeito e valor-p corrigido por Holm em cada família.", "tab:testes", "lllcccrc",
        r"\textbf{Q} & \textbf{Comparação} & \textbf{Pista} & $n_1/n_2$ & \textbf{Dif.} & \textbf{IC95\%} & "
        r"\textbf{Efeito} & \textbf{p (Holm)}", rows,
        nota=r"Q1 e Q2: Mann--Whitney, efeito = $\delta$ de Cliff (negativo: o 5×3 erra menos). Q3: Wilcoxon "
             r"pareado por modelo, diferença = mediana das diferenças, efeito = fração de modelos em que a saída 3 "
             r"erra menos. RMSE de $\beta$ em graus; de $v_x$, $v_y$ em m/s.", size=r"\scriptsize")

    e = res.parent.parent
    ic = pd.read_csv(e / "e1" / "rmse_sa.csv")
    cin = pd.read_csv(e / "e1b" / "rmse_beta_cinematico.csv")
    cin = cin[cin["escolhido"]]
    rows = []
    for r in ROLES:
        def icv(m, r=r):
            return num(ic[(ic["modelo"] == m) & (ic["papel"] == r)]["rmse_beta_deg"].iloc[0], 3)
        rows.append([PISTA[r], num(cin[cin["papel"] == r]["rmse_beta_deg"].iloc[0], 3), icv("T48"), icv("T19"),
                     num(rb[(rb["role"] == r) & (rb["variante"] == "A")]["mediana"].iloc[0], 3),
                     num(rb[(rb["role"] == r) & (rb["variante"] == "B_out")]["mediana"].iloc[0], 3)])
    tabs["referencias_beta"] = table(
        "RMSE de $\\beta$ (graus) do estimador cinemático, dos modelos de referência da IC (T48 e T19, $\\beta$ pós-calculado) "
        "e mediana dos modelos da campanha.", "tab:referencias_beta", "lccccc",
        r"\textbf{Pista} & \textbf{Cinemático} & \textbf{IC T48} & \textbf{IC T19} & \textbf{5×2 + atan2} & "
        r"\textbf{5×3, saída 3}", rows,
        nota=r"cinemático: integração de $\dot\beta = a_y/\hat v_x + \dot\psi$ com $\hat v_x$ proporcional à "
             r"velocidade da roda traseira, sem realimentação de medição; sinal do termo de guinada escolhido na "
             r"pista de treino. T48 e T19 foram selecionados pela IC olhando as pistas Wang 2.1 a 8.1.",
        size=r"\footnotesize")

    eb = pd.read_csv(an / "estrutura_beta_resumo.csv")
    rows = [[g.replace("Σθ", r"$\sum\theta$").replace("Σ|θ|", r"$\sum|\theta|$").replace("|θ|", r"$|\theta|$")
             .replace("ψ̇", r"$\dot\psi$").replace("a_y", "$a_y$").replace("v_y", "$v_y$").replace("v_x", "$v_x$")
             .replace("β", r"$\beta$").replace("≈", r"$\approx$").replace("±", r"$\pm$"),
             m.replace(".", ",").replace("+", ""), str(c)] for g, m, c in eb.itertuples(index=False)]
    tabs["estrutura_beta"] = table(
        "Estrutura da equação de $\\beta$ nos modelos 5×3: soma dos coeficientes dos monômios lineares de cada "
        "variável (qualquer atraso) após a expansão algébrica dos genes.", "tab:estrutura_beta", "p{7.2cm}cc",
        r"\textbf{Grandeza} & \textbf{Mediana [Q1; Q3]} & \textbf{Modelos}", rows, size=r"\footnotesize")

    # --- tabelas da correção do TCC (precisam de scripts/e3_referencia_corrigida_e_linear.py) --------------------
    reg = pd.read_csv(an / "erro_por_regime.csv")
    rows = []
    for eixo, nome in (("abs_beta", r"$|\beta|$"), ("vx", r"$v_x$")):
        for q in (1, 2, 3, 4):
            s = reg[(reg["eixo"] == eixo) & (reg["quartil"] == q)].set_index("variante")["mediana"]
            rows.append([nome if q == 1 else "", f"Q{q}", num(s["A"], 3), num(s["B_out"], 3), num(s["B_atan2"], 3)])
    tabs["regime"] = table(
        "RMSE de $\\beta$ (graus) por quartil de $|\\beta|$ e de $v_x$ de cada pista: mediana entre modelos e pistas de teste.",
        "tab:regime", "llccc", r"\textbf{Eixo} & \textbf{Quartil} & \textbf{5×2 + atan2} & \textbf{5×3, saída 3} & "
        r"\textbf{5×3 + atan2}", rows,
        nota=r"os quartis são os da própria pista (Q1 = menores valores); resultados descritivos, fora do plano de análise.")

    ex = res / "analise_extra"
    if (ex / "reavaliacao_resumo.csv").exists():
        rr = pd.read_csv(ex / "reavaliacao_resumo.csv").set_index(["role", "variante"])
        rows = []
        for r in TEST:
            for i, v in enumerate(["A", "B_out", "B_atan2"]):
                x = rr.loc[(r, v)]
                rows.append([PISTA[r] if i == 0 else "", VAR[v], num(x["rmse_orig"], 3), num(x["rmse_semvies_orig"], 3),
                             num(x["rmse_corr"], 3), num(x["rmse_semvies_corr"], 3)])
        tabs["reavaliacao"] = table(
            "RMSE de $\\beta$ (graus) nas pistas de teste, mediana entre modelos, contra a referência original e contra a "
            "referência com o deslocamento de $v_y$ removido; com e sem o viés.", "tab:reavaliacao", "llcccc",
            r"\textbf{Pista} & \textbf{Variante} & \multicolumn{2}{c}{\textbf{Referência original}} & "
            r"\multicolumn{2}{c}{\textbf{Referência corrigida}} \\ & & RMSE & RMSE sem viés & RMSE & RMSE sem viés", rows,
            nota=r"deslocamentos de $v_y$ removidos: 0,31~m/s (Wang 3.1), 0,37~m/s (Wang 8.1) e 0,31~m/s (Wang 3.2).")
        lin = pd.read_csv(ex / "linear_arx.csv")
        rows = []
        for r in ROLES:
            f = lambda m, v, c, r=r: lin[(lin["modelo"] == m) & (lin["role"] == r) & (lin["variante"] == v)][c].iloc[0]  # noqa: E731
            g = lambda v, c, r=r: rb[(rb["role"] == r) & (rb["variante"] == v)][c].iloc[0]  # noqa: E731
            rows.append([PISTA[r], num(f("FIR linear 5x2", "A", "rmse_beta_orig"), 3),
                         num(f("FIR linear 5x3", "B_out", "rmse_beta_orig"), 3), num(g("A", "mediana"), 3),
                         num(g("B_out", "mediana"), 3)])
        tabs["linear"] = table(
            "RMSE de $\\beta$ (graus) de um modelo linear de mesmas entradas e atrasos, sem realimentação das saídas, e "
            "mediana dos modelos MGGP.", "tab:linear", "lcccc",
            r"\textbf{Pista} & \textbf{Linear 5×2 + atan2} & \textbf{Linear 5×3, saída 3} & \textbf{MGGP 5×2 + atan2} & "
            r"\textbf{MGGP 5×3, saída 3}", rows,
            nota=r"o modelo linear usa as entradas nos atrasos $[1, 2, 5, 10, 25, 50]$ ajustado por mínimos quadrados "
                 r"na pista de treino; a versão com realimentação das saídas divergiu na simulação livre.",
            size=r"\footnotesize")

    for k, v in tabs.items():
        (out / f"{k}.tex").write_text(v, encoding="utf-8")
        print("  ", out / f"{k}.tex")


if __name__ == "__main__":
    main()
