# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Coleta, no PC que treinou, os dados que faltam nos trechos [PENDENTE] do TCC.

Só lê metadados (ledgers, params.json, git log, hardware). Não copia séries nem linhas da base da parceria:
o relatório pode ser colado no chat. Se numpy/pandas/openpyxl existirem, confere também o sinal da taxa de guinada
(correlação entre a_y e v_x·taxa de guinada, só um número).

Uso (na raiz do repositório, no PC do treino):
    python scripts/coletar_pendencias.py --results results/e2/t48 --data Database
Escreve pendencias_relatorio.txt e imprime o mesmo texto.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

OUT = []
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=isinstance(cmd, str)).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(falhou: {e})"


def maquina():
    p("## 4.1 Máquina")
    p("SO:", platform.platform())
    p("Python:", sys.version.split()[0])
    p("Processadores lógicos (os.cpu_count):", os.cpu_count())
    if platform.system() == "Windows":
        p("CPU:", run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"'))
        p("Núcleos físicos:", run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"'))
        ram = run('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"')
        p("RAM (GB):", ram)
    elif Path("/proc/cpuinfo").exists():
        nome = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), "?")
        p("CPU:", nome)
        mem = next((x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal")), "")
        p("RAM:", mem)
    else:
        p("CPU:", platform.processor(), "|", run("sysctl -n machdep.cpu.brand_string"))
    for pkg in ("numpy", "numba", "joblib", "scipy", "pandas"):
        try:
            m = __import__(pkg)
            p(f"{pkg}:", m.__version__)
        except Exception:  # noqa: BLE001
            p(f"{pkg}: não instalado")


def ts(x):
    try:
        return datetime.fromisoformat(x)
    except Exception:  # noqa: BLE001
        return None


def ledger(path):
    if not path.exists():
        p("  (não existe)", path)
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    cont = {}
    for r in rows:
        cont[r["status"]] = cont.get(r["status"], 0) + 1
    p(f"  {path.name}: {len(rows)} linhas, status = {cont}")
    ini = [t for t in (ts(r.get("started", "")) for r in rows) if t]
    fim = [t for t in (ts(r.get("finished", "")) for r in rows) if t]
    if fim:
        p(f"  primeira/última conclusão: {min(fim)} / {max(fim)}" + (f"; primeiro início: {min(ini)}" if ini else ""))
    seg = sorted(float(r["seconds"]) for r in rows if r.get("seconds") not in (None, ""))
    if seg:
        p(f"  duração por treino: mediana {seg[len(seg) // 2] / 3600:.1f} h, mín {seg[0] / 3600:.1f} h, máx {seg[-1] / 3600:.1f} h")
    for r in rows:
        if r["status"] != "approved" or r.get("error"):
            p(f"  tentativa {r['attempt']} semente {r['seed']}: {r['status']} rmse_val={r.get('rmse_val')} erro={r.get('error', '')[:120]!r}")
    p("  tentativas listadas:", ",".join(sorted({str(int(r['attempt'])) for r in rows}, key=int)))


def treino(res):
    p("\n## 3.5 Falha do braço 5x3 e 4.1 paralelismo")
    for nome in ("params.json", "5x3/params_retomada_2026-10-28.json"):
        f = res / nome
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            p(f"  {nome}: workers={j.get('workers')} max_consecutive_fail={j.get('max_consecutive_fail')} base_seed={j.get('base_seed')} arms={j.get('arms')} runs={j.get('runs')}")
    for arm in ("5x2", "5x3"):
        p(f"[{arm}]")
        st = res / arm / "STATUS.txt"
        p("  STATUS:", st.read_text(encoding="utf-8").strip().replace("\n", " | ") if st.exists() else "(sem STATUS.txt)")
        for f in sorted((res / arm).glob("attempts*.csv")):
            ledger(f)
        d = res / arm / "attempts"
        if d.exists():
            pastas = sorted(x.name for x in d.iterdir() if x.is_dir())
            p(f"  pastas de tentativa: {len(pastas)} ({pastas[0]} .. {pastas[-1]})")
            for pasta in pastas:
                log = d / pasta / "log.txt"
                if log.exists() and "CRASH" in log.read_text(encoding="utf-8", errors="replace"):
                    ultimas = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "CRASH" in x]
                    p(f"  {pasta}: {ultimas[-1][:160]}")
    for f in sorted(res.glob("**/*retomada*.csv"))[:1]:
        p("  arquivo de retomada:", f)


def plano():
    p("\n## 3.6 Registro do plano de análise (git)")
    for alvo in ("notebooks/E2_analise_beta.ipynb", "GOALS.md", "notebooks/E2_treinamento_validacao.ipynb"):
        p(f"[{alvo}]")
        p(run(["git", "log", "--format=%h %ad %s", "--date=short", "-n", "6", "--", alvo]) or "  (sem commits)")
    p("HEAD:", run(["git", "log", "-1", "--format=%H %ad", "--date=iso"]))
    p("git status resumido:", run(["git", "status", "--short"])[:400] or "limpo")


def guinada(data):
    p("\n## 4.2 Convenção de sinal da taxa de guinada")
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # noqa: BLE001
        p("  numpy/pandas não instalados: pulado")
        return
    f = Path(data) / "wang21dv_bic_MGGP.xlsx"
    if not f.exists():
        p("  base não encontrada:", f)
        return
    d = pd.read_excel(f, header=None).to_numpy(float)
    ay, vx, r = d[:, 2], d[:, 18], d[:, 11]
    c = np.corrcoef(ay, vx * r)[0, 1]
    c2 = np.corrcoef(d[:, 2], d[:, 12])[0, 1]
    p(f"  corr(a_y, v_x·taxa de guinada) = {c:+.3f}  (positivo: a_y e guinada têm o mesmo sinal na curva; a_y = v_x·psi_ponto vale para o eixo z da guinada no mesmo sentido do eixo y)")
    p(f"  corr(a_y, ay_true col 12) = {c2:+.3f}")
    bd = np.degrees(np.arctan2(d[:, 17], d[:, 18]))
    dbeta = np.gradient(np.convolve(bd, np.ones(41) / 41, "same"), 1 / 200)
    pred = ay / np.maximum(vx, 3) * 180 / np.pi
    for s in (+1, -1):
        e = dbeta - (pred + s * np.degrees(r))
        p(f"  RMSE de dβ/dt com a_y/v_x {'+' if s > 0 else '-'} psi_ponto (graus/s, v_x>3): {np.sqrt(np.mean(e[vx > 3] ** 2)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/e2/t48")
    ap.add_argument("--data", default="Database")
    a = ap.parse_args()
    p("# Relatório de pendências do TCC —", datetime.now().isoformat(timespec="seconds"))
    maquina()
    treino(Path(a.results))
    plano()
    guinada(a.data)
    Path("pendencias_relatorio.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print("\nSalvo em pendencias_relatorio.txt")


if __name__ == "__main__":
    main()
