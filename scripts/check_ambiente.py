# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "pandas==2.3.3", "openpyxl", "deap==1.4.3", "dill", "joblib==1.5.3",
#                 "numba==0.64.0", "matplotlib==3.10.8", "scipy==1.15.3", "scikit-learn==1.7.2", "tqdm==4.67.3"]
# ///
"""Checklist automático antes de rodar a campanha E2 (outro PC / PC do laboratório).

Só LÊ: não treina, não escreve nada. Sai com código 1 se algo bloqueante falhar.
Rode da raiz do repositório:  uv run scripts/check_ambiente.py        (ou: python scripts/check_ambiente.py)
Outra raiz (ex.: worktree sem Database/):  --root <pasta>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PINS = {"numpy": "2.2.6", "pandas": "2.3.3", "deap": "1.4.3", "numba": "0.64.0", "joblib": "1.5.3",
        "scipy": "1.15.3", "scikit-learn": "1.7.2", "matplotlib": "3.10.8", "tqdm": "4.67.3"}
SEM_PIN = ["openpyxl", "dill"]
TRACKS = ["wang21dv_bic_MGGP.xlsx", "wang22dv_bic_MGGP.xlsx", "wang31dv_bic_MGGP.xlsx",
          "wang81dv_bic_MGGP.xlsx", "wang32dv_bic_MGGP.xlsx"]
# mesmos hashes do notebook (lib da IC @7514bfb, fim de linha normalizado)
LIB_SHA256 = {"base": "ba87864422671f9a", "crossings": "fd3ff39ab3e72615", "mggp": "98afae10530b2675",
              "mutations": "91b2d5c1335f3b0d", "predictors": "cadc5e99b1cf43af"}
META = {"5x2": 30, "5x3": 30, "5x1": 30}

falhas, avisos = [], []


def ok(msg):
    print(f"  [ok]     {msg}")


def erro(msg):
    falhas.append(msg)
    print(f"  [FALHA]  {msg}")


def aviso(msg):
    avisos.append(msg)
    print(f"  [aviso]  {msg}")


def secao(t):
    print(f"\n== {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--config", default="t48")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    print(f"raiz: {root}")

    secao("1. Python")
    v = sys.version_info
    (ok if (v.major, v.minor) == (3, 12) else erro)(f"Python {v.major}.{v.minor}.{v.micro} (precisa ser 3.12.x)")

    secao("2. Pacotes (versões da IC)")
    for pkg, want in PINS.items():
        try:
            have = metadata.version(pkg)
            (ok if have == want else erro)(f"{pkg} {have}" + ("" if have == want else f" (esperado {want})"))
        except metadata.PackageNotFoundError:
            erro(f"{pkg} não instalado (esperado {want})")
    for pkg in SEM_PIN:
        try:
            ok(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            erro(f"{pkg} não instalado")
    for pkg in ("jupyter_core", "nbconvert", "ipykernel"):
        try:
            importlib.import_module(pkg)
            ok(f"{pkg} disponível (para rodar o notebook)")
        except ImportError:
            aviso(f"{pkg} ausente — precisa para abrir/rodar o notebook (uv sync --group dev, ou pip install jupyterlab)")

    secao("3. Dados (Database/)")
    faltando = [f for f in TRACKS if not (root / "Database" / f).exists()]
    (erro if faltando else ok)(f"faltam {len(faltando)}/5: {', '.join(faltando)}" if faltando
                               else "as 5 pistas Wang estão em Database/")

    secao("4. Lib MGGP (src/ idêntica à da IC @7514bfb)")
    for nome, esperado in LIB_SHA256.items():
        f = root / "src" / f"{nome}.py"
        if not f.exists():
            erro(f"src/{nome}.py ausente")
            continue
        h = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
        (ok if h == esperado else erro)(f"src/{nome}.py {'idêntico à IC' if h == esperado else 'DIFERENTE da IC'}")
    sys.path.insert(0, str(root))
    try:
        from src.mggp import MGGP  # noqa: F401
        ok("from src.mggp import MGGP funciona")
    except Exception as e:
        erro(f"import da lib falhou: {type(e).__name__}: {e}")

    secao("5. Máquina")
    n = os.cpu_count() or 1
    ok(f"{n} núcleos lógicos (5x1 ≈ 1,7 h/run e 5x3 ≈ 5 h/run com 1 worker sozinho; mais carga = mais lento)")
    livre = shutil.disk_usage(root).free / 2**30
    (ok if livre > 20 else aviso)(f"{livre:.0f} GB livres em disco (cada tentativa guarda ~MBs; folga recomendada > 20 GB)")
    if not (root / "notebooks" / "E2_treinamento_validacao.ipynb").exists():
        erro("notebooks/E2_treinamento_validacao.ipynb ausente")
    else:
        nb = json.loads((root / "notebooks" / "E2_treinamento_validacao.ipynb").read_text(encoding="utf-8"))
        cfg = "".join(nb["cells"][2]["source"])
        (ok if "RUNS_POR_BRACO" in cfg and "5x1" in cfg else erro)(
            "notebook é a versão com o braço 5x1" if "RUNS_POR_BRACO" in cfg and "5x1" in cfg
            else "notebook é a versão ANTIGA (sem 5x1) — faça git pull")

    secao(f"6. Estado da campanha (results/e2/{a.config}/)")
    base = root / "results" / "e2" / a.config
    if not base.exists():
        aviso(f"{base} não existe — a campanha começaria do zero. Para completar o 5x3, copie results/e2/{a.config}/ do PC de origem")
    else:
        import pandas as pd

        for arm, meta in META.items():
            d = base / arm
            led = d / "attempts.csv"
            n_ok = n_rej = n_pkl = 0
            if led.exists():
                df = pd.read_csv(led)
                n_ok, n_rej = int((df.status == "approved").sum()), int((df.status != "approved").sum())
            if (d / "models").exists():
                n_pkl = len(list((d / "models").glob("modelo_rmse_*.pkl")))
            if not led.exists() and arm == "5x1":
                ok("5x1: sem histórico (campanha ainda não começou — esperado)")
                continue
            if not led.exists():
                (aviso if arm == "5x3" else erro)(f"{arm}: sem attempts.csv")
                continue
            msg = f"{arm}: {n_ok}/{meta} aprovados no ledger · {n_pkl} .pkl · {n_rej} reprovada(s)"
            (ok if n_ok == n_pkl else erro)(msg + ("" if n_ok == n_pkl else "  ← ledger e models/ não batem"))
        d3 = base / "5x3" / "attempts.csv"
        if d3.exists():
            n3 = int((pd.read_csv(d3).status == "approved").sum())
            if n3 == 28:
                ok("5x3 está em 28/30: a retomada vai lançar exatamente 2 tentativas")
            elif n3 < 28:
                aviso(f"5x3 tem só {n3} aprovados: a retomada lançaria {30 - n3} tentativas, não 2 — "
                      "confira se copiou a pasta results/e2/t48/5x3 completa deste PC")
            elif n3 >= 30:
                aviso("5x3 já tem 30/30 — nada a retomar")
        for arm in ("5x2", "5x3", "5x1"):
            if (base / arm / "STATUS.txt").exists():
                print(f"           {arm} STATUS: {(base / arm / 'STATUS.txt').read_text(encoding='utf-8').splitlines()[0]}")

    print("\n" + "=" * 60)
    if falhas:
        print(f"BLOQUEANTE: {len(falhas)} falha(s) — arrume antes de lançar:")
        for f in falhas:
            print(f"  - {f}")
    else:
        print("Nenhuma falha bloqueante." + (f" {len(avisos)} aviso(s) acima." if avisos else ""))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
