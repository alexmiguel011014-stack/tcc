# TCC — Ângulo de deriva como terceira saída MIMO do sensor virtual MGGP

Extensão da IC *"Estimação de Velocidades em Veículos Inteligentes utilizando Programação Genética
Multigene e Modelo de Bicicleta"* (UFLA, Depto. de Automática). A IC estimou `v_x`, `v_y` a partir de
IMU + velocidade de rodas com a lib Python `mggp` (5 entradas × 2 saídas). O TCC testa o ângulo de
deriva `β = atan2(v_y, v_x)` como terceira saída (5×3) contra o `β` pós-calculado do modelo 5×2.

- Contexto da IC: [`CONTEXTO_IC.md`](CONTEXTO_IC.md) · plano completo: [`GOALS.md`](GOALS.md)
- Resultado do GOALS 1 (o `β` da base é **computado**, não medido): [`docs/slip_angle_provenance.md`](docs/slip_angle_provenance.md)
- Legenda das 24 colunas do dataset: [`docs/column_inventory.md`](docs/column_inventory.md)

## 1. Dados (não estão no repositório)

A base Wang/Jilin é de parceria e **nunca é commitada** (`Database/` está no `.gitignore`).
Copie os cinco arquivos para `Database/` na raiz do projeto:

```
Database/
  wang21dv_bic_MGGP.xlsx   # treino
  wang22dv_bic_MGGP.xlsx   # validação
  wang31dv_bic_MGGP.xlsx   # teste A
  wang81dv_bic_MGGP.xlsx   # teste B
  wang32dv_bic_MGGP.xlsx   # teste C (não usado na IC)
```

Origem: pasta `IC\Database\` do PC principal ou o PC do laboratório. Conferir:

```bash
python -c "import glob; print(sorted(glob.glob('Database/wang*.xlsx')))"
```

## 2. Instalar

Precisa de Python **3.12** (a lib da IC está pinada em `numpy 2.2.6` / `numba 0.64`, que não rodam no 3.13+).

### Opção A — `uv` (recomendado; baixa o Python 3.12 e as dependências sozinho)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/alexmiguel011014-stack/tcc.git
cd tcc
uv run scripts/run_e2_mimo3.py --help     # primeira execução cria o ambiente (~1 min)
```

Os scripts têm as dependências declaradas inline (PEP 723): `uv run scripts/<nome>.py` resolve tudo,
sem `uv sync`. Para os testes/lint do projeto use `uv sync` (lê `pyproject.toml` + `uv.lock`).

### Opção B — `pip` (PC sem `uv`)

```bash
py -3.12 -m venv .venv            # Windows; Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_e2_mimo3.py --help
```

Nos dois casos rode os scripts **a partir da raiz do repositório** (eles procuram `Database/` e `src/` por lá).

## 3. Rodar

### Experimento E2 — 30 modelos 5×2 + 30 modelos 5×3 do T48, em paralelo

```bash
uv run scripts/run_e2_mimo3.py --config t48 --arms 5x2,5x3 --runs 30
# sem uv:  python scripts/run_e2_mimo3.py --config t48 --arms 5x2,5x3 --runs 30
```

Porta do `TrainingMGGP_LOOP.ipynb` da IC: mesma lib, mesmos parâmetros do T48 (`nDelays [1,2,5,10,25,50]`,
7 termos, `maxHeight 6`, 300 gerações × 300 indivíduos, MShooting → Free-Run), mesma regra de aprovação
(`0 < RMSE(Wang 2.2, Free-Run) < 100`). O que muda:

- **Paralelo de verdade.** A lib é single-thread, então o driver dispara **um processo por tentativa** e mantém
  `--workers` deles ocupados (padrão: todos os núcleos lógicos → CPU a 100 %). Cada worker roda com BLAS/numba
  em 1 thread para não haver oversubscription. Com núcleos suficientes, as 60 tentativas iniciais sobem de uma vez.
- **Dois braços numa campanha só** (`--arms 5x2,5x3`): fila única de jobs, contadores independentes por braço.
  `5x2` reproduz a IC (β = `atan2(v̂y, v̂x)` pós-hoc); `5x3` treina β como 3ª saída (`atan2(col 17, col 18)`, em
  graus por padrão — `--beta-unit`).
- **Regra de fracasso** (`--max-consecutive-fail 5`): tentativa falha = free-run da Wang 2.2 explode (inf/NaN/≥ gate)
  ou o worker quebra. **5 falhas seguidas ⇒ o braço vira `FRACASSO`**: nada mais é lançado para ele e os jobs em voo
  são encerrados. Uma aprovação zera o contador (não é acumulativo). "Seguidas" = na ordem em que terminam.
- **Sem desperdício**: em voo ≤ `runs − aprovados`; nunca treina além da meta.
- **Retomável**: `Ctrl+C` encerra os workers; o mesmo comando continua do `attempts.csv`. Caiu a luz, idem.
- Cada modelo aprovado é validado Free-Run nas **5 pistas** dentro do próprio worker, e as predições ficam salvas
  (`predictions/*.npz`) para a análise posterior sem refazer free-run.

Medir antes de comprometer a máquina: `--runs 1 --workers 1` e ler `seconds` em `attempts.csv`.

Saída em `results/e2/t48/<braço>/` (ignorado pelo git):

| arquivo | conteúdo |
|---|---|
| `models/modelo_rmse_N.pkl` | modelo aprovado N (`dill`) |
| `predictions/modelo_rmse_N.npz` | `{papel}_yd` / `{papel}_yp` por pista — séries reais e preditas de todas as saídas |
| `fig/modelo_rmse_N.png` | Free-Run nas 5 pistas: v_x, v_y, β (saída 3 e `atan2(v̂_y, v̂_x)` sobrepostos) |
| `relatorio_validacao_rmse.csv` | RMSE por modelo × pista: `Vx`, `Vy`, `beta_out_deg` (só 5x3), `beta_atan2_deg` |
| `summary_agg.csv` | média ± desvio entre modelos, por pista |
| `attempts.csv`, `STATUS.txt` | ledger de todas as tentativas (seed, status, RMSE, tempo) e estado do braço |
| `attempts/attempt_XXX/log.txt` | log completo do treinamento daquela tentativa (`tail -f` para acompanhar) |

Para trazer os resultados de volta: copie `results/e2/t48/` inteira (pendrive/drive) — nada disso vai pelo git.

### GOALS 1 — análise do dataset (já feita; reproduzível)

```bash
uv run scripts/inspect_columns.py      # -> docs/column_inventory.md
uv run scripts/check_slip_angle.py     # -> docs/slip_angle_provenance.md + docs/fig/
```

## 4. Estrutura

```
src/          lib mggp vendorizada da IC (@7514bfb) — ver src/VENDORED.md; não editar sem registrar lá
scripts/      análises e o loop de treinamento
docs/         relatórios gerados pelos scripts
Database/     dados (git-ignored)
results/      saídas de treinamento (git-ignored)
```

## 5. Testes e lint (ambiente do projeto)

```bash
uv sync
uv run ruff check .
uv run pytest            # testes sem dataset; os marcados `dataset` pulam se Database/ não existir
```
