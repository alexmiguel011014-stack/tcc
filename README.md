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
uv sync                                   # cria o ambiente com Python 3.12 (~1-2 min)
```

Os scripts têm as dependências declaradas inline (PEP 723): `uv run scripts/<nome>.py` resolve tudo,
sem `uv sync`. Para os testes/lint do projeto use `uv sync` (lê `pyproject.toml` + `uv.lock`).

### Opção B — `pip` (PC sem `uv`)

```bash
py -3.12 -m venv .venv            # Windows; Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab notebooks/E2_treinamento_validacao.ipynb
```

O notebook acha a raiz do repositório sozinho (procura `src/mggp.py` subindo na árvore); os scripts do GOALS 1 devem ser rodados a partir da raiz.

## 3. Rodar

### Experimento E2 — 30 modelos 5×2 + 30 modelos 5×3 do T48

Um notebook único faz **treinamento e validação**: [`notebooks/E2_treinamento_validacao.ipynb`](notebooks/E2_treinamento_validacao.ipynb).

```bash
uv run jupyter lab notebooks/E2_treinamento_validacao.ipynb
```

Toda a configuração está na primeira célula de código (`CONFIG`, `ARMS`, `RUNS`, `WORKERS`,
`MAX_CONSEC_FAIL`, `BETA_UNIT`, `SEED`). Rode as células na ordem; a campanha é a célula 5 e o
resumo com gráficos é a célula 6.

Porta do `TrainingMGGP_LOOP.ipynb` da IC: mesma lib, mesmos parâmetros do T48 (`nDelays [1,2,5,10,25,50]`,
7 termos, `maxHeight 6`, 300 gerações × 300 indivíduos, MShooting → Free-Run), mesma regra de aprovação
(`0 < RMSE(Wang 2.2, Free-Run) < 100`). O que muda:

- **Paralelo de verdade.** A lib é single-thread, então cada tentativa roda num processo próprio
  (joblib/loky) e `WORKERS` deles ficam ocupados — padrão: todos os núcleos lógicos, CPU a 100 %.
  Cada processo usa 1 thread de BLAS/numba, sem oversubscription.
- **Dois braços na mesma campanha** (`ARMS = ["5x2", "5x3"]`, nessa ordem): `5x2` reproduz a IC
  (β = `atan2(v̂y, v̂x)` pós-hoc); `5x3` treina β como 3ª saída (`atan2(col 17, col 18)`, em graus).
- **Regra de fracasso** (`MAX_CONSEC_FAIL = 5`): tentativa falha quando o Free-Run da Wang 2.2 explode
  (inf/NaN/≥ `GATE_MAX`) ou o processo quebra. **5 falhas seguidas ⇒ o braço vira `FRACASSO`**: nada mais
  é lançado e o que está rodando é cancelado. Uma aprovação zera o contador (não é acumulativo).
  "Seguidas" = na ordem em que as tentativas terminam.
- **Sem desperdício**: cada leva submete exatamente `RUNS − aprovados` tentativas.
- **Retomável**: pare o kernel (ou caia a luz) e rode tudo de novo — ele continua do `attempts.csv`.
- **Painel ao vivo** (`PAINEL_MODO`): por padrão (`"stream"`) mostra um log por vez, como a lib mostra rodando sozinha — as últimas gerações (`gen evals min avg max`) e a barra da avaliação em andamento, da tentativa `PAINEL_SEGUIR` (padrão: a primeira em execução) — mais uma linha `outras` com a geração das demais. `"tabela"` mostra uma linha por tentativa. `⚠ ... sem log (travou?)` aparece se um log para de crescer por mais de `STALL_SEGUNDOS` (120 s). Só lê o `log.txt` de cada tentativa.
- **Libera a CPU sozinho**: ao terminar cada braço, ao fim da campanha e se você interromper o kernel, `liberar_recursos()` encerra os workers (inclusive tentativas em curso), fecha as figuras e libera a memória. Se o kernel for reiniciado ou fechado no meio, cada worker percebe em até 5 s que ficou órfão e se encerra sozinho.
- Cada modelo aprovado é validado Free-Run nas **5 pistas** dentro do próprio processo, e as predições
  ficam salvas (`predictions/*.npz`) para a análise posterior sem refazer free-run.

Antes de lançar a campanha o notebook confere as 5 pistas em `Database/` e **para na hora** se faltar
alguma (sem isso, cada tentativa quebraria e o braço marcaria `FRACASSO` por falta de dados).

Medir antes de comprometer a máquina: rode uma vez com `RUNS = 1` e `WORKERS = 1`, e leia `seconds`
em `attempts.csv`.

Saída em `results/e2/t48/<braço>/` (ignorada pelo git):

| arquivo | conteúdo |
|---|---|
| `models/modelo_rmse_N.pkl` | modelo aprovado N (`dill`) |
| `predictions/modelo_rmse_N.npz` | `{papel}_yd` / `{papel}_yp` por pista — séries reais e preditas de todas as saídas |
| `fig/modelo_rmse_N.png` | Free-Run nas 5 pistas: v_x, v_y, β (saída 3 e `atan2(v̂_y, v̂_x)` sobrepostos) |
| `relatorio_validacao_rmse.csv` | RMSE por modelo × pista: `Vx`, `Vy`, `beta_out_deg` (só 5x3), `beta_atan2_deg` |
| `summary_agg.csv` | média ± desvio entre modelos, por pista |
| `attempts.csv`, `STATUS.txt` | ledger de todas as tentativas (seed, status, RMSE, tempo) e estado do braço |
| `attempts/attempt_XXX/log.txt` | log completo do treinamento daquela tentativa |

Para trazer os resultados de volta: copie `results/e2/t48/` inteira (pendrive/drive) — nada disso vai pelo git.

> Ao commitar o notebook, limpe as saídas antes (**Kernel → Restart Kernel and Clear Outputs**):
> as tabelas e gráficos gerados vêm dos dados da parceria.

### Exportar no formato do trabalho da IC

[`notebooks/E2_exportar_formato_IC.ipynb`](notebooks/E2_exportar_formato_IC.ipynb) converte `results/e2/t48/` para a
estrutura `RESULTADOS-N/Treinamento_ID` da IC (`TRAINING/*.pkl`, `modelo_rmse_N.png`, `relatorio_validacao_rmse.csv`
com `;` e vírgula decimal, `parametros_utilizados.csv`, `checklist_local.xlsx`), em
`results/e2/t48_formato_IC/RESULTADOS-E2/` — `Treinamento_1` = 5×2, `Treinamento_2` = 5×3. Assim o `BOXPLOT.ipynb` e o
`mineracao/` antigos leem os resultados novos sem mudança. É **somente leitura** na fonte: pode rodar com a campanha em
andamento. Extras: `beta_validacao.csv` (RMSE do β) e `tentativas.csv` (seeds e status).

### GOALS 1 — análise do dataset (já feita; reproduzível)

```bash
uv run scripts/inspect_columns.py      # -> docs/column_inventory.md
uv run scripts/check_slip_angle.py     # -> docs/slip_angle_provenance.md + docs/fig/
```

## 4. Estrutura

```
src/          lib mggp vendorizada da IC (@7514bfb) — ver src/VENDORED.md; não editar sem registrar lá
notebooks/    E2_treinamento_validacao.ipynb — treino + validação (campanha)
scripts/      análises do dataset (GOALS 1)
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
