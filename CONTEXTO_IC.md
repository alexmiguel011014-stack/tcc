# Contexto da IC — base para o TCC

> Síntese da Iniciação Científica de **Alexsander Miguel do Sacramento** (UFLA, Depto. de Automática, 2026),
> gerada a partir de `RELATORIOPARCIAL_IC.tex`, `Ufla/IC_MGGP_Artigo.pdf` (jun/2026) e do PDF do relatório
> completo (set/2026). Serve de ponto de partida para definir o escopo do TCC.

## 1. Identificação

| Campo | Valor |
|---|---|
| Título | Estimação de Velocidades em Veículos Inteligentes utilizando Programação Genética Multigene e Modelo de Bicicleta |
| Área | Identificação de sistemas / computação evolutiva / dinâmica veicular |
| Palavras-chave | MGGP; Sensor Virtual; Veículos Inteligentes; Dinâmica Veicular; Computação Evolutiva |
| Linha do grupo | Castro (2021, 2023), Mota (2022), Braz (2018) — MGGP para NARMAX e sensores virtuais na UFLA |
| Ferramenta | Biblioteca Python `mggp` (CastroHc/MGGP, DEAP + NumPy), modo MIMO |

## 2. Problema

- Estimar **velocidade longitudinal (v_x)** e **lateral (v_y)** de um veículo a partir de sensores inerciais baratos (IMU + velocidade de rodas), **sem modelo físico explícito** e sem GPS/INS caro.
- Métodos físicos exigem parâmetros (atrito de pneu, aerodinâmica) difíceis de obter e variáveis no tempo.
- Objetivo: **sensor virtual simbólico, interpretável e parcimonioso**, viável para embarcado.

## 3. Fundamentação usada

- **Modelo de bicicleta (cinemático)**, referência no CG:
  - `dv_x/dt = a_x + psi_dot * v_y`
  - `dv_y/dt = a_y - psi_dot * v_x`
- **MGGP**: cada indivíduo = conjunto de genes (árvores); saída = combinação linear dos genes com coeficientes por **mínimos quadrados (MQO)**. Evolução decide a *estrutura*; MQO ajusta os *parâmetros*.
- Operadores: crossover alto/baixo nível, mutação, torneio, elitismo.
- Regimes de avaliação: **One-step-ahead**, **Free-run** (recursivo, realimenta a própria predição), **Multi-shooting** (blocos de tamanho K reiniciados com valor real).

## 4. Base de dados

| Item | Valor |
|---|---|
| Origem | Parceria com a **Universidade de Jilin (China)**; veículo instrumentado com GPS/INS (referência), DAQ, transdutores de pneu (não usados) |
| Conjuntos | 5 pistas "Random Driving Conditions": **Wang 2.1** (treino), **Wang 2.2** (validação), **Wang 3.1** (teste A), **Wang 8.1** (teste B), Wang 3.2 (**não usado**) |
| Amostragem | **200 Hz** (dt = 5 ms); diferem só na duração |
| Entradas (u1..u5) | u1 = a_x (m/s²), u2 = a_y (m/s²), u3 = vel. roda dianteira (m/s), u4 = vel. roda traseira (m/s), u5 = psi_dot (rad/s) |
| Saídas (y1, y2) | y1 = v_x (m/s), y2 = v_y (m/s) |
| Formato original | Excel `wang21dv_bic_MGGP.xlsx` etc. (estavam em `C:\IC_alex\Database\` — **não presentes nesta máquina**) |

## 5. Metodologia experimental

- Arquitetura **MIMO**: mesmas entradas defasadas alimentam todas as árvores; cada saída tem seus próprios coeficientes.
- **Grade de busca = 147 configurações**: 3 cenários de `nDelays` × 7 `nTerms` (1–7) × 7 `maxHeight` (1–7).
  - Cenário 1: `nDelays = [1]`
  - Cenário 2: `nDelays = [1, 2, 5, 10]`
  - Cenário 3: `nDelays = [1, 2, 5, 10, 25, 50]`
- Parâmetros fixos: 300 gerações, população 300, K = 300, treino **Multi-shooting**, teste **Free-run**, mutação 0,3, crossover 0,8, elitismo 10 %, operadores `{+, -, *}`, `froe_mode = False`.
- Métrica única: **RMSE** (m/s) — penaliza erros grandes, unidade física, estável no cruzamento por zero de v_y.
- Pipeline (notebook `TrainingMGGP_LOOP.ipynb`): carrega Excel → `StandardScaler` → `MGGP(...).run()` → `hof[0]` → `predict('FreeRun')` nas 4 pistas → planilha de controle `controle_treinamentos.xlsx` → modelos serializados com `dill`.
- Total de **1406 equações** geradas ao longo da busca.

## 6. Resultados principais

| Achado | Detalhe |
|---|---|
| `nDelays = [1]` | Instabilidade severa em Free-run; várias execuções divergem (erro → infinito), sobretudo em v_y |
| `nDelays = [1,2,5,10]` | Estabiliza; top-10 com RMSE médio 0,2815–0,3821 |
| `nDelays = [1,2,5,10,25,50]` | Produz o **melhor modelo individual** (RMSE geral **0,2121**), mas top-10 (0,3402–0,3993) não supera consistentemente o cenário 2 → mais atrasos = teto maior, **variância maior** |
| **Melhor absoluto** | Treinamento 48, cenário 3, `nTerms = 7`, `maxH = 6`: RMSE treino 0,2728 (v_x) / 0,2130 (v_y); 0,1224–0,3505 nas 4 pistas |
| **Melhor custo-benefício (eleito)** | Treinamento 19, cenário 2, `nTerms = 3`, `maxH = 5`, `modelo_rmse_6`: RMSE geral **0,2239** (**5,6 %** pior que o melhor com menos da metade dos termos); supera todas as configs de 4, 5 e 6 termos |

Fronteira custo-benefício (melhor por `nTerms`): 1→0,3945 · 2→0,3019 · **3→0,2239** · 4→0,2526 · 5→0,2407 · 6→0,2356 · 7→0,2121.

### Modelo eleito (T19) — coeficientes já ajustados

```
v_x[k] = 0,99981*v_x[k-1]
       + 9,70e-4*a_x[k-1]*a_x[k-3]*a_x[k-5]*a_x[k-9]*a_x[k-12]
       - 7,61e-5*u4[k-3]*psi_dot[k-3]*v_y[k-3]*v_x[k-16]*a_x[k-16]*u4[k-26]
       - 3,70e-5

v_y[k] = 0,49956*(v_y[k-1])^2
       + 1,21e-3*a_y[k-13]*a_y[k-3]*a_y[k-5]
       + 6,30e-4*u4[k-7]*a_y[k-7]*u4[k-7]*psi_dot[k-7]
       - 2,99e-4
```

### Leitura física das equações

- Termo dominante quase universal: **autorregressivo de 1ª ordem** (`v[k-1]`, coef ≈ 1) — coerente com inércia a 200 Hz. A contribuição real do MGGP está nos **termos residuais** (2–4 ordens de grandeza menores) que evitam acúmulo de erro em Free-run.
- Nas 100 melhores equações: **79 %** têm como principal correção de v_x o produto `v_y * u5` (= `psi_dot * v_y`, exatamente o termo de Coriolis de `dv_x/dt = a_x + psi_dot*v_y`); **55 %** têm `u2` (= `a_y`) isolado como correção de v_y (coerente com `dv_y/dt = a_y - psi_dot*v_x`).
- → Evidência de que o MGGP **recupera estrutura física real**, não só ajusta ruído.
- Maioria das equações carrega 1–2 produtos de alta ordem (5–12 fatores) com coef ~1e-6, **sem contribuição prática** (bloat residual).

## 7. Limitações declaradas

1. **6 das 147 configs** não entraram na análise final (3 com `nDelays` errado, 2 interrompidas, 1 sem arquivo de parâmetros).
2. **FROE pruning** (poda estrutural) existe na lib, mas `froe_mode=True` **não funciona em MIMO** → equações não compactadas automaticamente.
3. Wang 3.2 nunca usado.
4. Objetivo específico "selecionar conhecimento prévio e inseri-lo no projeto dos modelos" foi tratado só indiretamente (a física apareceu *a posteriori* nas equações; não foi injetada *a priori*).
5. Sem comparação com baseline (EKF / cinemático puro / LSTM); sem análise de custo computacional real de inferência; sem teste em hardware.

## 8. Trabalhos futuros sugeridos no próprio relatório

1. Completar as configurações pendentes da grade.
2. Estender **FROE para MIMO** e avaliar simplificação sem retreino.
3. Investigação físico-teórica dos acoplamentos identificados vs. parâmetros do modelo de bicicleta.

## 9. Ativos existentes nesta máquina (reaproveitáveis no TCC)

| Ativo | Caminho | Relevância |
|---|---|---|
| Relatório IC (LaTeX, classe `uflamon`, ABNT) | `C:\Users\Alex\Desktop\RELATORIOPARCIAL_IC.tex` | Base textual; faltam `Conteudo/`, `Imagens/`, `tabelapc*.tex`, `referencias.bib` (não estão no Desktop) |
| Relatório IC (PDF, 40 p.) | `C:\Users\Alex\Desktop\Ufla\Estimação_..._(2) (1).pdf` | Versão completa com conclusão |
| Artigo IC (PDF, 9 p., jun/2026) | `C:\Users\Alex\Desktop\Ufla\IC_MGGP_Artigo.pdf` | Versão curta, resultados parciais (ainda apontava T10 como melhor) |
| Vídeo apresentação | `C:\Users\Alex\Desktop\Ufla\artigofinalPtcc.mp4` | — |
| **Port MATLAB do MGGP** (git local) | `C:\Users\Alex\Desktop\sites\MGGP_matlab` | Reimplementação nativa (MISO/MIMO, MShooting, parfor, lsGpu, NSGA-II). **Nunca executado no MATLAB real** (pendência crítica registrada no `PROJETO.md`). Inclui cópias da lib Python (`mggptestePYTHON`, `mggptestePYTHONCUDA`) e o notebook da IC (`dev/comparison/TrainingMGGP_LOOP.ipynb`) |
| Firmware ESP32-S3 + IMU ICM42670P (FreeRTOS, PlatformIO) | `C:\Users\Alex\Desktop\Ufla\integrador-main` | Plataforma embarcada já dominada pelo autor — candidata a alvo de deploy do sensor virtual |
| Simulações 1/2-DOF, PID, Simulink | `C:\Users\Alex\Desktop\Ufla\PROJETO FINAL`, `Trab_final_SIMTO` | Disciplinas correlatas (controle) |
| **Dados Wang + modelos treinados (`.pkl`/`dill`) + `controle_treinamentos.xlsx`** | `C:\IC_alex\...` | **Ausentes nesta máquina** — precisam ser recuperados (PC do laboratório / drive) |

## 10. Perguntas abertas para definir o TCC

Direções compatíveis com a IC (não excludentes):

| # | Direção | Tipo de entrega | Reaproveita |
|---|---|---|---|
| A | **Sensor virtual embarcado**: implementar o modelo T19 (3 termos) em C/C++ no ESP32-S3, inferência recursiva a 200 Hz, validar com replay das pistas Wang e/ou IMU real; medir latência/RAM/precisão numérica (float32 vs. float64) | Firmware + bancada + relatório | integrador-main, T19 |
| B | **FROE pruning para MIMO**: estender a lib `mggp` (Python), reavaliar os top-N modelos podados, quantificar simplificação × RMSE | Software (Python) + experimento | lib mggp, modelos `.pkl` |
| C | **MGGP físico-informado (grey-box)**: injetar termos do modelo de bicicleta (`psi_dot*v_y`, `a_y`, `psi_dot*v_x`) como genes-semente / restrições e comparar com o puro data-driven; fecha o objetivo específico 2 da IC | Software + experimento + análise teórica | lib mggp, dados Wang |
| D | **Benchmark contra baselines**: EKF cinemático (bicicleta), integração direta, LSTM (Kong 2022) vs. MGGP, mesmas pistas, mesma métrica, + custo computacional | Experimento comparativo | dados Wang, modelos IC |
| E | **Validar e usar o port MATLAB**: rodar a suíte de testes, fechar paridade com Python, repetir parte da grade em MATLAB (CPU/GPU), entregar toolbox ao orientador | Software (MATLAB) | MGGP_matlab |

Decisões que só o autor pode tomar:

1. Qual direção (ou combinação) o orientador aceita como TCC?
2. Os dados Wang e os modelos `.pkl` da IC são recuperáveis? (sem eles, B/C/D exigem retreino do zero)
3. Há veículo/bancada física disponível para A, ou a validação seria só por replay?
4. Linguagem/stack principal do TCC: Python (lib original), MATLAB (port), C++ (embarcado) ou combinação?
5. Prazo e formato de entrega (monografia UFLA `uflamon` + artigo? + código no GitHub?).
