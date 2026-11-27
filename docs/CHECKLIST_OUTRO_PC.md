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

## 3. Configurar e lançar (duas execuções em paralelo)
Abra **dois** notebooks/kernels do `notebooks/E2_treinamento_validacao.ipynb`, ambos com a raiz do repositório
como pasta de trabalho. Atalho: na célula 1 (Configuração) use `PERFIL = "5x3"` no kernel A e `PERFIL = "5x1"` no
kernel B — ele já ajusta `ARMS` e `WORKERS` como na tabela abaixo (o resto da célula fica como está):

| | Kernel A (completar o 5x3) | Kernel B (treinar o 5x1) |
|---|---|---|
| `ARMS` | `["5x3"]` | `["5x1"]` |
| `RUNS` | `30` | `30` (ou `RUNS_POR_BRACO = {"5x1": 10}` para uma leva só) |
| `WORKERS` | `2` | núcleos lógicos − 2 (ex.: 10 num PC de 12) |
| `CONFIG`, `SEED`, `GATE_MAX`, `BETA_UNIT` | não mexer (`"t48"`, `2026`, `100.0`, `"deg"`) | idem |

Por que dois kernels: o notebook roda os braços em sequência; o 5x3 só usa 2 núcleos, então o 5x1 ficaria
esperando ~8 h à toa.

Não use `GENERATIONS`/`POPULATION`/`MAX_SAMPLES` (só para teste rápido) — deixe `None`.

### 3b. Alternativa: rodar direto no terminal (sem abrir o Jupyter)
Dois terminais PowerShell, ambos na raiz do repositório (a variável vale só para o terminal onde foi definida):

```powershell
$env:E2_PERFIL = "5x3"; uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute notebooks/E2_treinamento_validacao.ipynb --output _out_5x3.ipynb --ExecutePreprocessor.timeout=-1
```
```powershell
$env:E2_PERFIL = "5x1"; uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute notebooks/E2_treinamento_validacao.ipynb --output _out_5x1.ipynb --ExecutePreprocessor.timeout=-1
```
- Não há painel ao vivo: acompanhe por `results/e2/t48/<braço>/attempts/attempt_XXX/log.txt` e `attempts.csv`.
- Feche o terminal e o treino morre (os workers se encerram sozinhos em ~5 s); rodar o mesmo comando retoma do `attempts.csv`.
- `notebooks/_out_*.ipynb` fica fora do git (tem saídas derivadas dos dados).
- Uso de CPU: 5x3 usa 2 núcleos (só 2 tentativas) e o 5x1 usa núcleos−2; juntos ocupam 100 % **enquanto o 5x3 roda**
  (~5–8 h). Depois disso 2 núcleos ficam livres.

## 4. Conferir nos primeiros minutos
- [ ] A célula de preflight imprime `dataset OK` e `lib OK: src/ idêntica ao MGGP da IC @7514bfb`.
- [ ] Kernel A anuncia `[5x3] RUNNING — 28/30 aprovados` e `leva de 2 tentativa(s)`.
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
