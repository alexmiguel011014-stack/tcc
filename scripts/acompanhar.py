"""Painel de progresso da campanha E2 para quem roda no terminal (sem o painel do Jupyter).

Só LÊ os logs: não mexe no treino. Use em um terminal separado, na raiz do repositório:
    python scripts/acompanhar.py              # atualiza a cada 5 s (Ctrl+C sai)
    python scripts/acompanhar.py --once       # imprime uma vez
Só biblioteca padrão.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LINHA_GEN = re.compile(r"^(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)$")   # logbook da lib: gen evals min avg max
BARRA = re.compile(r"Evaluating Population:\s+(\d+)%.*?(\d+)/(\d+)")
CABECALHO = re.compile(r"^=== (\S+) tentativa (\d+) seed (\d+)")
GERACOES = 300


def mmss(seg):
    h, r = divmod(int(max(0, seg)), 3600)
    return f"{h}:{r // 60:02d}:{r % 60:02d}" if h else f"{r // 60:02d}:{r % 60:02d}"


def ler_log(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        tam = f.tell()
        f.seek(max(0, tam - 200_000))
        bruto = f.read().decode("utf-8", errors="replace")
    with open(path, "rb") as f:
        primeira = f.readline().decode("utf-8", errors="replace")
    linhas = re.split(r"[\r\n]", bruto)
    gen = None
    barra = None
    for ln in linhas:
        m = LINHA_GEN.match(ln.strip())
        if m:
            gen = m.groups()
        b = BARRA.search(ln)
        if b:
            barra = b.groups()
    h = CABECALHO.match(primeira)
    return {"gen": gen, "barra": barra, "fim_treino": "Executed in:" in bruto,
            "seed": h.group(3) if h else "?"}


def ledger(arm_dir):
    csv_path = arm_dir / "attempts.csv"
    if not csv_path.exists():
        return 0, 0
    with open(csv_path, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return sum(r["status"] == "approved" for r in linhas), len(linhas)


def montar(base, meta):
    agora = time.time()
    out = [f"campanha em {base}   ·   {time.strftime('%H:%M:%S')}", ""]
    for arm in ("5x3", "5x1", "5x2"):
        arm_dir = base / arm
        if not arm_dir.exists():
            continue
        aprov, total = ledger(arm_dir)
        status = (arm_dir / "STATUS.txt").read_text(encoding="utf-8").splitlines()[0] if (arm_dir / "STATUS.txt").exists() else "-"
        out.append(f"[{arm}] {aprov}/{meta} aprovados · {total} tentativa(s) no ledger · STATUS {status}")
        ativos = []
        for d in sorted((arm_dir / "attempts").glob("attempt_*")):
            log = d / "log.txt"
            if log.exists() and not (d / "result.json").exists() and agora - log.stat().st_mtime < 900:
                ativos.append((d, log))
        if not ativos:
            out.append("    nenhuma tentativa rodando agora")
        for d, log in ativos:
            info = ler_log(log)
            parado = agora - log.stat().st_mtime
            decorrido = agora - log.stat().st_ctime
            n = d.name.removeprefix("attempt_")
            if info["fim_treino"]:
                out.append(f"    {n} seed {info['seed']}  treino terminado, validando as 5 pistas (fica um tempo sem log)")
                continue
            if info["gen"]:
                g, _, mn, av, mx = info["gen"]
                g = int(g)
                eta = mmss(decorrido * (GERACOES - g) / g) if g > 0 else "--:--"
                barra_p = "█" * (g * 20 // GERACOES) + "░" * (20 - g * 20 // GERACOES)
                aviso = f"   ⚠ {parado:.0f} s sem log (travou?)" if parado > 600 else ""
                out.append(f"    {n} seed {info['seed']}  geração {g:>3}/{GERACOES} {barra_p} {g * 100 // GERACOES:>3}%  "
                           f"melhor {mn}  média {av}  · decorrido {mmss(decorrido)} · falta ~{eta}{aviso}")
            else:
                b = info["barra"]
                pop = f"{b[1]}/{b[2]} ({b[0]}%)" if b else "..."
                out.append(f"    {n} seed {info['seed']}  avaliando a população inicial {pop} · decorrido {mmss(decorrido)}")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--config", default="t48")
    ap.add_argument("--base", help="pasta que contém as pastas 5x3/5x1/5x2 (padrão: <root>/results/e2/<config>)")
    ap.add_argument("--meta", type=int, default=30)
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    base = Path(a.base) if a.base else Path(a.root) / "results" / "e2" / a.config
    if not base.exists():
        sys.exit(f"não existe {base}")
    try:
        while True:
            texto = montar(base, a.meta)
            if not a.once:
                os.system("cls" if os.name == "nt" else "clear")
            print(texto)
            if a.once:
                return
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
