# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pypdf>=6"]
# ///
"""GOALS 6 item 6.0.1: measure the monograph between Overleaf compiles.

    uv run scripts/monografia_paginas.py palavras [--orcamento]   words per section of monografia/secoes/*.tex
    uv run scripts/monografia_paginas.py pdf [--pdf PATH]         page map of the Overleaf PDF (blank pages, '??')
    uv run scripts/monografia_paginas.py refs                     \\ref/\\cite targets without \\label/BibTeX entry
    uv run scripts/monografia_paginas.py siglas                   acronyms not written in full at first use
"""

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "monografia"
SECOES = ["introducao", "referencial", "metodos", "resultados", "conclusao"]
ORCAMENTO = {"introducao": 900, "referencial": 1780, "metodos": 1600, "resultados": 1720, "conclusao": 430,
             "resumo": 300, "abstract": 300}
MINIMO = {"resumo": 150, "abstract": 150}
COMENTARIO = re.compile(r"(?<!\\)%.*")
TITULO = re.compile(r"(\\(?:chapter|section|subsection)\{[^}]*\})")


def clean(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " EQ ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", "X", s)
    s = re.sub(r"\\(cite|citeonline|autoref|ref|label|input|url)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(textit|emph|textbf|textsubscript|texttt)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return s


def sem_comentarios(texto):
    return "\n".join(COMENTARIO.sub("", ln) for ln in texto.splitlines())


def palavras(orcamento):
    totais = {}
    for nome in SECOES:
        bruto = (MONO / "secoes" / f"{nome}.tex").read_text(encoding="utf-8")
        atual, blocos = "(antes do título)", {}
        for parte in TITULO.split(bruto):
            m = re.match(r"\\(?:chapter|section|subsection)\{([^}]*)\}", parte)
            if m:
                atual = m.group(1)
                blocos.setdefault(atual, 0)
                continue
            blocos[atual] = blocos.get(atual, 0) + len(clean(parte).split())
        for titulo, n in blocos.items():
            print(f"{nome}\t{titulo}\t{n}")
        totais[nome] = sum(blocos.values())
        print(f"{nome}\tTOTAL\t{totais[nome]}")
    main = (MONO / "main.tex").read_text(encoding="utf-8")
    for nome in ("resumo", "abstract"):
        m = re.search(re.escape("\\" + nome + "{%") + r"\n(.*?)\n\}", main, re.S)
        totais[nome] = len(clean(m.group(1)).split()) if m else 0
        print(f"{nome}\tTOTAL\t{totais[nome]}")
    print(f"TODAS\tTOTAL\t{sum(totais[n] for n in SECOES)}")
    if not orcamento:
        return 0
    falhas = 0
    for nome, limite in ORCAMENTO.items():
        if totais[nome] > limite:
            print(f"ESTOURO\t{nome}\t{totais[nome]}\t{limite}")
            falhas += 1
        if totais[nome] < MINIMO.get(nome, 0):
            print(f"ESTOURO\t{nome}\t{totais[nome]}\tmínimo {MINIMO[nome]}")
            falhas += 1
    return 1 if falhas else 0


INICIO_NOMEADO = re.compile(
    r"^(\d+\s+)?(RESUMO|ABSTRACT|SUMÁRIO|AGRADECIMENTOS|LISTA DE [A-ZÀ-Ü]+|REFERÊNCIAS|APÊNDICES?)\b")
INICIO_SECAO = re.compile(r"^(\d+\s+)?([1-9])\s+([A-ZÀ-Ü][A-ZÀ-Ü ,\-]{3,})$")


def pontilhado(linha):
    return ". ." in linha or "..." in linha


def pdf(caminho):
    from pypdf import PdfReader

    paginas = PdfReader(caminho).pages
    brancas, indefinidas, inicios = [], 0, []
    for i, pagina in enumerate(paginas, 1):
        texto = pagina.extract_text() or ""
        chars = sum(not c.isspace() for c in texto)
        linhas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
        branca = chars < 40
        if branca:
            brancas.append(i)
        indefinidas += texto.count("??")
        print(f"{i}\t{chars}\t{'BRANCA' if branca else ''}\t{(linhas[0] if linhas else '')[:60]}")
        # sumário e listas: as linhas com pontilhado são entradas, não inícios de seção
        pontilhada = sum(pontilhado(ln) for ln in linhas) >= 3
        for ln in linhas[:3]:
            if pontilhado(ln):
                continue
            m = INICIO_NOMEADO.match(ln)
            if m:
                inicios.append((m.group(2), i))
                break
            m = None if pontilhada else INICIO_SECAO.match(ln)
            if m:
                inicios.append((f"{m.group(2)} {m.group(3).strip()}", i))
                break
    print(f"paginas\t{len(paginas)}")
    print(f"brancas\t{len(brancas)}\t{','.join(map(str, brancas))}")
    print(f"indefinidas\t{indefinidas}")
    for k, (nome, ini) in enumerate(inicios):
        fim = inicios[k + 1][1] if k + 1 < len(inicios) else len(paginas) + 1
        print(f"{nome}\t{ini}\t{fim - ini}")
    return 0


def arquivos_tex():
    return ([MONO / "main.tex"] + sorted((MONO / "secoes").glob("*.tex")) + sorted((MONO / "tabelas").glob("*.tex"))
            + sorted((MONO / "apendices").glob("*.tex")))


def refs():
    labels, usos, citas = set(), [], []
    for f in arquivos_tex():
        texto = sem_comentarios(f.read_text(encoding="utf-8"))
        labels |= set(re.findall(r"\\label\{([^}]+)\}", texto))
        usos += [(alvo, f.name) for alvo in re.findall(r"\\(?:auto|page)?ref\{([^}]+)\}", texto)]
        for grupo in re.findall(r"\\cite(?:online)?\{([^}]+)\}", texto):
            citas += [(k.strip(), f.name) for k in grupo.split(",")]
    bib = set(re.findall(r"^@\w+\{([^,]+),", (MONO / "referencias.bib").read_text(encoding="utf-8"), re.M))
    falhas = 0
    for alvo, nome in usos:
        if alvo not in labels:
            print(f"ref sem label\t{alvo}\t{nome}")
            falhas += 1
    for chave, nome in citas:
        if chave not in bib:
            print(f"citacao sem entrada\t{chave}\t{nome}")
            falhas += 1
    print(f"{len(usos)} referências, {len(citas)} citações, {falhas} problema(s)")
    return 1 if falhas else 0


def siglas():
    curtas = re.findall(r"\\newacronym\{[^}]*\}\{([^}]*)\}", (MONO / "glossarios" / "siglas.tex").read_text(encoding="utf-8"))
    partes = [(n, sem_comentarios((MONO / "secoes" / f"{n}.tex").read_text(encoding="utf-8"))) for n in SECOES]
    texto = "\n".join(t for _, t in partes)
    # títulos de seção ficam curtos: a sigla é expandida na primeira frase sob eles (mesmo comprimento, mesmas posições)
    texto = TITULO.sub(lambda m: " " * len(m.group()), texto)
    fronteiras, pos = [], 0
    for n, t in partes:
        fronteiras.append((pos, n))
        pos += len(t) + 1
    falhas = 0
    for curta in curtas:
        m = re.search(r"(?<![A-Za-z])" + re.escape(curta) + r"(?![A-Za-z])", texto)
        if not m:
            print(f"nao usada\t{curta}")
            continue
        antes = texto[:m.start()]
        abre = antes.rfind("(")
        entre = antes[abre + 1:] if abre >= 0 else antes
        if abre < 0 or ")" in entre or len(entre) > 12:
            arquivo = [n for p, n in fronteiras if p <= m.start()][-1]
            print(f"sem expansao\t{curta}\t{arquivo}")
            falhas += 1
    print(f"{len(curtas)} siglas, {falhas} sem expansão na primeira ocorrência")
    return 1 if falhas else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("palavras")
    p.add_argument("--orcamento", action="store_true", help="falha se uma seção passar do orçamento de GOALS 6")
    p = sub.add_parser("pdf")
    p.add_argument("--pdf", default=str(MONO / "main.pdf"))
    sub.add_parser("refs")
    sub.add_parser("siglas")
    a = ap.parse_args()
    if a.cmd == "palavras":
        return palavras(a.orcamento)
    if a.cmd == "pdf":
        return pdf(a.pdf)
    return refs() if a.cmd == "refs" else siglas()


if __name__ == "__main__":
    sys.exit(main())
