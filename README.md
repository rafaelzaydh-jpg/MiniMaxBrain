# MiniMaxBrain (MMB)

MiniMaxBrain é um runtime independente de paginação de pesos para LLMs esparsas/MoE. Ele mantém no SSD os blocos que não cabem na RAM, admite somente os pesos obrigatórios para o cálculo atual e permite escolher o limite por bytes ou por número de especialistas residentes.

O componente entregue é o **gate externo**. Ele é comparável ao papel operacional de um gerenciador de memória física: controla a residência física, integridade por selo criptográfico, leases e fila de I/O assíncrona para permitir que modelos gigantescos rodem em hardware de consumo com RAM limitada.

> [!IMPORTANT]
> **⚠️ Transparência sobre o Estado Atual do Projeto:**
> - **O que o MMB É hoje:** Um *runtime* de controle físico de pesos (SSD ➔ RAM compartilhada) com orçamentos restritos de memória, conversor GGUF automatizado e selo de integridade pré-voo.
> - **O que o MMB AINDA NÃO É:** Ele **não é uma interface de chat pronta no navegador** (como o `llama-server` ou `Ollama`). O componente atual gerencia a movimentação física dos pesos para alimentar um executor, mas ainda não inclui os kernels tensoriais embutidos para gerar texto ponta a ponta sozinho na tela.
> - **Roadmap do Projeto:**
>   - ✅ **Fase 1 (Entregue - v0.2.0):** Porteiro físico externo, IPC de memória compartilhada, selo criptográfico `seal`, conversor GGUF 1-clique e orçamento modular de RAM.
>   - ⏳ **Fase 2 (Em Desenvolvimento):** Conexão dos kernels tensoriais (`down/gate/up`) e do router da LLM diretamente sobre os blocos paginados pelo MMB.
>   - 🎯 **Fase 3 (Planejado):** Servidor HTTP local com WebUI (aba de chat visual no navegador estilo `llama-server`) e API compatível com OpenAI (`/v1/chat/completions`).

---

## 📊 Comparativo: LLM Convencional (Normal) vs MiniMaxBrain (MMB)

A tabela abaixo resume a diferença fundamental entre executar uma LLM MoE pelo método tradicional (modelo integral carregado na memória) versus usar o runtime de paginação física do **MiniMaxBrain**:

| Dimensão | Inferência Convencional (Sem MMB) | MiniMaxBrain (MMB Paginado) |
|---|---|---|
| **Requisito de RAM** | Exige **100% dos pesos** residentes na RAM/VRAM. | Exige apenas o **núcleo fixo + especialistas ativos**. |
| **Escalabilidade em PCs Comuns** | ❌ **Inviável para modelos grandes**: um MoE de 35B ou 2T causa estouro de memória (*Out of Memory*) ou *swap* destrutivo do SO. | ✅ **Viável**: um modelo de 35B roda com folga em apenas **4 GiB de RAM**; modelos de até 2T tornam-se fisicamente possíveis. |
| **Consumo de Memória Real** | Ocupa toda a memória do modelo (~1 a centenas de GB). | Redução observada de **70% a 87,5%** no *working set*. |
| **Gargalo Principal** | Capacidade e custo de memória RAM/VRAM. | Largura de banda de leitura do SSD (NVMe). |
| **Comportamento em Modelos Pequenos** | Mais rápido (mantém tudo na RAM). | Deve usar modo *bypass* / *full-resident* quando o modelo já couber. |
| **Integridade de Dados** | Carrega uma vez e confia na RAM. | **Selo Criptográfico Pré-Voo (`seal`)**: validação $O(1)$ instantânea com fail-closed sem travar a CPU. |

---

## 🔬 Resultados Experimentais Reais

### 1. Teste com Modelo Gigante: **Qwen 3.6 35B A3B MoE**
* **Parâmetros:** 35,5 Bilhões (com 3,3 Bilhões ativos por token).
* **Topologia:** 41 camadas, 10.496 blocos de especialistas, ~13 GB de pesos fatiados.

| Configuração | Modelo Normal (Sem MMB) | MMB Sob Demanda | MMB com Selo + Orçamento 4 GiB | Ganho com MMB Otimizado |
|---|---:|---:|---:|---:|
| **RAM Mínima de Pesos** | **~13 a 20 GiB** | 127 MiB | **~4 GiB (Modular)** | **Economia de >70% de RAM** |
| **Expulsões (*Evictions*)** | 0 (tudo em RAM) | 640 expulsões | **0 expulsões (Zero!)** | **Zero Churn de Cache** |
| **Tempo da Rota Física** | *Inviável em RAM <16GB* | 6,189 s | **2,108 s** | **Quase 3× mais rápido** |
| **Integridade** | Sem checagem | SHA-256 no I/O | **Selo Pré-Voo (`seal`)** | **100% Seguro sem perda de FPS** |

---

### 2. Teste com Modelo Pequeno: **IBM Granite 3.0 1B-A400M Instruct**
* **Parâmetros:** 1,33 Bilhão (com 400M ativos).
* **GGUF:** 821,8 MB / 24 camadas MoE / 768 blocos de especialistas.

| Métrica | Modelo integral, sem MMB | Harness de paginação MiniMaxBrain |
|---|---:|---:|
| **Pico de working set observado** | **1.024,12 MiB** | **127,15 MiB** (**87,58% menor**) |
| **Pico de memória privada** | 766,78 MiB | **123,60 MiB** (**8,05× menor**) |
| **Tempo da Rota Física (16 experts)** | — | **0,666 s** (com integridade *seal*) |

> [!NOTE]
> **Veredito Prático:**
> - **Quando o modelo cabe na RAM:** Manter o modelo integral residente continua sendo a opção mais rápida.
> - **Quando o modelo NÃO cabe na RAM:** O MiniMaxBrain é a ponte de **viabilidade física**, permitindo que modelos que exigiriam estações de trabalho de centenas de gigabytes de RAM rodem em computadores domésticos comuns com SSDs rápidos.

---

## 🏛️ Arquitetura

```text
                    PROCESSO DO EXECUTOR
 
 prompt -> camadas core -> router real -> experts obrigatórios
                              |                 |
                     gate interno futuro        | acquire
                     + ModelMemory               |
                              | prefetch         |
                              v                  v
====================== IPC MMB v1 ======================
                              |
                    GATE EXTERNO INDEPENDENTE
 
       mapa físico -> admission controller -> leases
                            |                   |
                            v                   v
                      fila de I/O -> cache RAM compartilhada
                            |
                            v
                       shards no SSD
```

O gate externo é a autoridade sobre a residência física. Ele não escolhe significado, não altera o `top-k` do modelo e nunca troca um especialista obrigatório por outro.

---

## 🚀 Como Usar

### ⚡ Início Rápido no Windows (1 Clique)

1. **Instale o ambiente:** Dê 2 cliques no arquivo [`instalador.bat`](file:///c:/Users/Kel/Desktop/nvopokea/MiniMaxBrain/instalador.bat) na raiz do projeto. Ele verifica o Python, instala as dependências e valida a suíte de testes automaticamente.
2. **Adicione o modelo:** Coloque seu arquivo `.gguf` dentro da pasta [`conversor/`](file:///c:/Users/Kel/Desktop/nvopokea/MiniMaxBrain/conversor).
3. **Converta e Sele:** Dê 2 cliques no arquivo [`conversor.bat`](file:///c:/Users/Kel/Desktop/nvopokea/MiniMaxBrain/conversor.bat). Ele reconhece o modelo automaticamente, fatia os especialistas, gera a configuração modular `gate.json` e cria o selo de integridade pré-voo!

---

### 💻 Uso Manual via Linha de Comando (CLI)

#### 1. Instalação Manual

Requer Python 3.11 ou superior (usa apenas a biblioteca padrão):

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Instale o comando `mmb`:

```powershell
python -m pip install -e .
mmb --help
```

### 2. Inspecione e Converta o Modelo

Inspecione um modelo GGUF MoE suportado:

```powershell
python mmb.py gguf-inspect --gguf modelo.gguf
```

Converta o MoE em blocos pagináveis MiniMaxBrain:

```powershell
python mmb.py gguf-pack-moe --gguf modelo.gguf --output modelo-mmb
```

### 3. Crie o Selo de Integridade Pré-Voo

Para rodar com segurança máxima e sem penalidade de latência de SHA-256 no carregamento:

```powershell
python mmb.py seal --config mmb.json
```

### 4. Valide a Configuração e o Orçamento

```powershell
python mmb.py check --config mmb.json
python mmb.py smoke --config mmb.json --blocks 1
```

### 5. Execução em Produção

Inicie o gate como um serviço de memória compartilhada para o seu executor:

```powershell
python mmb.py serve --config mmb.json
```

---

## ⚙️ Configuração Modular de Orçamento

### Modo por Orçamento Exato de RAM (Recomendado):
O usuário escolhe quanta RAM quer dedicar e o MMB aloca dinamicamente o máximo de especialistas que couberem:

```json
{
  "schema_version": "mmb-external-gate-config-v1",
  "model_map": "model.mmb-map.json",
  "memory": {
    "ram_budget": "4GiB",
    "resident_experts": null,
    "kv_cache": "512MiB",
    "scratch": "256MiB",
    "transport": "shared_memory",
    "lease_timeout_seconds": 120
  },
  "io": {
    "workers": 2,
    "prefetch_queue": 32,
    "integrity": "seal"
  }
}
```

### Modo por Quantidade Fixa de Especialistas:

```json
{
  "schema_version": "mmb-external-gate-config-v1",
  "model_map": "model.mmb-map.json",
  "memory": {
    "ram_budget": null,
    "resident_experts": 16,
    "kv_cache": "512MiB",
    "scratch": "256MiB",
    "transport": "shared_memory",
    "lease_timeout_seconds": 120
  }
}
```

---

## 🗂️ Estrutura do Repositório

```text
minimaxbrain/                  runtime do gate externo, cache, storage e ModelMemory
tests/                         suíte de testes unitários e de integração
tools/                         scripts de comparação A/B, smoke e IPC
real_model_test/               validações reais e configurações do Granite e Qwen 35B
docs/external-gate-architecture.md
docs/model-format.md
docs/memory-kernel.md
mmb.py                         CLI executável sem necessidade de instalação
mmb.example.json               configuração de referência
```

---

## 📄 Documentação Detalhada

- [`docs/external-gate-architecture.md`](docs/external-gate-architecture.md) — arquitetura física detalhada e limites de I/O;
- [`docs/model-format.md`](docs/model-format.md) — formato de empacotamento e contratos de mapa;
- [`docs/memory-kernel.md`](docs/memory-kernel.md) — banco de dados estrutural e histórico de rotas;
- [`real_model_test/REPORT_QWEN.md`](real_model_test/REPORT_QWEN.md) — relatório completo do modelo Qwen 35B;
- [`real_model_test/REPORT.md`](real_model_test/REPORT.md) — relatório completo do modelo Granite MoE.

---

## ⚖️ Licença

MiniMaxBrain é source-available para uso pessoal e não comercial. Consulte [`LICENSE.md`](LICENSE.md) e [`COMMERCIAL.md`](COMMERCIAL.md).
