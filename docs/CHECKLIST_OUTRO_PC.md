# Checklist — rodar a campanha E2 no outro PC

Objetivo: completar o **5x3** (faltam o `modelo_rmse_29` e o `30`) e treinar o **5x1** (β sozinho).
Estado de partida (PC de origem): 5x2 = 30/30 · 5x3 = 28/30 · 5x1 = 0.

## 1. Trazer o código
```powershell
git pull            # ou: git clone <repo> e depois git checkout <branch>
```
`Database/` e `results/` **não** vêm pelo git (dados da parceria; estão no `.gitignore`).

## 2. Rodar o verificador automático
```powershell
uv sync
uv run scripts/check_ambiente.py
```
(sem `uv`: `python scripts/check_ambiente.py` dentro do ambiente Python 3.12.)

Ele confere, e sai com `BLOQUEANTE` se algo estiver errado:

| # | Item | Se falhar |
|---|---|---|
| 1 | Python 3.12.x | instalar 3.12 (`uv` baixa sozinho) |
| 2 | numpy 2.2.6, numba 0.64.0, deap 1.4.3, pandas 2.3.3, joblib 1.5.3… + jupyter/nbconvert | `uv sync` |
| 3 | 5 arquivos `Database/wang{21,22,31,81,32}dv_bic_MGGP.xlsx` | copiar do PC de origem |
| 4 | `src/` idêntico ao MGGP da IC @7514bfb (hash dos 5 arquivos) | `git checkout -- src/` — nunca editar a lib |
| 5 | notebook é a versão com o braço 5x1 | `git pull` |
| 6 | `results/e2/t48/5x3` com **28 aprovados** e 28 `.pkl` | copiar a pasta `results/e2/t48/` inteira do PC de origem |

O item 6 é o que decide quantas tentativas o 5x3 vai lançar: com 28/30 ele lança **exatamente 2**.
Se o PC novo tiver menos de 28, o verificador avisa — não deixe lançar assim, ele refaria o que já existe.

## 3. Lançar (um kernel só, os dois braços juntos)
O notebook agora roda **todos os braços de `ARMS` ao mesmo tempo**, dividindo as `WORKERS` vagas de CPU
(padrão: todos os núcleos). Não precisa trocar de braço nem abrir dois kernels:

- `ARMS = ["5x3", "5x1"]` (padrão): o 5x3 pega 2 vagas (só faltam 2 modelos) e o 5x1 pega o resto; quando o 5x3
  termina, as vagas dele passam sozinhas para o 5x1. A ordem em `ARMS` é a prioridade.
- `RUNS = 30` (ou `RUNS_POR_BRACO = {"5x1": 10}` para uma leva só). `CONFIG`, `SEED`, `GATE_MAX`, `BETA_UNIT`: não mexer.
- Não use `GENERATIONS`/`POPULATION`/`MAX_SAMPLES` (só para teste rápido) — deixe `None`.

**Jupyter (VS Code):** abra `notebooks/E2_treinamento_validacao.ipynb`, kernel do `.venv`, **Run All**.

**Terminal (sem Jupyter):**
```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/E2_treinamento_validacao.ipynb --output _out_campanha.ipynb --ExecutePreprocessor.timeout=-1
```
- Num terminal não há painel do Jupyter: use `python scripts/acompanhar.py` em outro terminal.
- Fechar o terminal/VS Code mata o treino (os workers se encerram sozinhos em ~5 s); rodar de novo retoma do `attempts.csv`.
- `notebooks/_out_*.ipynb` fica fora do git (tem saídas derivadas dos dados).
- O atalho `PERFIL = "5x3"` / `"5x1"` (ou `E2_PERFIL`) continua existindo, para rodar um braço só com vagas fixas.

**Regras que a campanha aplica sozinha (testadas):**
- Cada tentativa roda num **processo novo** que morre ao terminar (RAM devolvida; nada passa de uma para a outra).
- Tentativa **reprovada** = RMSE da Wang 2.2 em Free-Run infinito, NaN, ≤ 0 ou ≥ 100, ou crash.
- **5 falhas seguidas ⇒ FRACASSO do braço** (para de lançar e mata o que está rodando). Uma aprovação **zera** o
  contador: 4 falhas + 1 aprovação + 4 falhas = continua; 4 + 1 + 5 = para. A contagem segue a ordem em que as
  tentativas terminam e continua entre retomadas (via `attempts.csv`).
- Worker morto à força (ex.: falta de memória) não conta como falha: a leva é relançada (até 3 vezes seguidas).

## 4. Conferir nos primeiros minutos
- [ ] A célula de preflight imprime `dataset OK` e `lib OK: src/ idêntica ao MGGP da IC @7514bfb`.
- [ ] Aparece `[5x3] RUNNING — 28/30 aprovados` e `leva de 2 tentativa(s)`, e `[5x1] RUNNING — 0/30`.
- [ ] O painel mostra `geração N/300` andando; log em `results/e2/t48/<braço>/attempts/attempt_XXX/log.txt`.
- [ ] Cada `attempt_XXX/log.txt` começa com `System Mode: MIMO … N° Outputs: 3` (5x3) ou `System Mode: MISO … N° Outputs: 1` (5x1).

## 5. Tempos esperados (medidos num PC de 12 núcleos lógicos, 1 worker sozinho)
- 5x1: ≈ 20 s/geração → **~1,7 h por run** (30 runs em 10 workers ≈ 3 levas ≈ 5–9 h).
- 5x3: ≈ 62 s/geração → **~5 h por run** (7–8 h sob carga).
- Pode parar e rodar de novo a qualquer momento: a campanha retoma do `attempts.csv`.
- Tentativa reprovada (Free-Run da Wang 2.2 explodiu) conta como falha; **5 falhas seguidas ⇒ FRACASSO** do braço.

## 6. Ao terminar
- [ ] `results/e2/t48/5x3/STATUS.txt` = `DONE`, `aprovados=30/30`; `results/e2/t48/5x1/STATUS.txt` = `DONE`.
- [ ] `models/` tem `modelo_rmse_1..30.pkl` em cada braço.
- [ ] Copiar `results/e2/t48/` de volta para o PC de análise (**não** commitar: `results/` é gitignored).
- [ ] Rodar a última célula do notebook (liberar recursos) e não commitar o notebook com saídas
      (Restart Kernel and Clear Outputs antes de qualquer commit).
