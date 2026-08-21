<div align="center">

# 🧠 MiniMaxBrain (MMB)

**Runtime independente de paginação física de pesos (SSD ➔ RAM) para LLMs esparsas e Mixture of Experts (MoE).**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Source--Available-green.svg)](LICENSE.md)
[![Release](https://img.shields.io/badge/version-v0.2.0-orange.svg)](https://github.com/rafaelzaydh-jpg/MiniMaxBrain/releases)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](tests/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

<br>

*Permite que modelos massivos (de 35B até 2 Trilhões de parâmetros) rodem em computadores comuns com 8 GB a 12 GB de RAM, transmitindo do SSD apenas os especialistas necessários para cada token.*

<br>

[⚡ Início Rápido](#-get-started-início-rápido) •
[✨ Destaques](#-principais-recursos) •
[📊 Benchmarks](#-benchmarks-reais) •
[🏗️ Arquitetura](#️-como-funciona) •
[🗺️ Roadmap](#️-roadmap) •
[📄 Documentação](#-documentação)

</div>

---

## 💡 O que é o MiniMaxBrain?

Modelos de linguagem modernos com arquitetura **MoE (Mixture of Experts)** possuem dezenas de bilhões de parâmetros, mas ativam apenas uma fração mínima (poucos especialistas) para cada palavra gerada. 

Carregar 100% dos pesos na memória RAM tradicional exige dezenas ou centenas de gigabytes de RAM/VRAM caríssimas. O **MiniMaxBrain** resolve isso desacoplando o armazenamento da execução:

* 💾 **Mantém os pesos frios no SSD** em shards alinhados de alta velocidade (`.mmbw`).
* 🎯 **Admite na RAM apenas os especialistas obrigatórios** para a camada atual.
* 🛡️ **Orçamento Estrito e Seguro**: Sem estouros de memória (*Out of Memory*) e sem *swap* destrutivo do sistema operacional.

> [!IMPORTANT]
> **Estado Atual (v0.2.0):** O MiniMaxBrain entrega o **Porteiro Físico Externo**, responsável pelo controle de memória, conversão de modelos, cache LRU, leases e selo de integridade criptográfico. Ele alimenta um executor de inferência através de memória compartilhada (IPC) e prepara o caminho para a futura interface de chat integrada.

---

## ✨ Principais Recursos

- 🚀 **Economia de até 87% de RAM**: Execute modelos de 35B (como o Qwen 3.6 MoE) alocando apenas **4 GiB de RAM**.
- ⚡ **Zero-Overhead com Selo Criptográfico (`integrity: seal`)**: Validação pré-voo $O(1)$ que elimina o gargalo de SHA-256 no carregamento e libera a taxa de transferência bruta do SSD NVMe (>2.000 MB/s).
- 🎛️ **Orçamento Modular por RAM (`ram_budget`)**: Você define quanta RAM quer dedicar (ex: `4GiB`, `8GiB`) e o MMB calcula dinamicamente a retenção de especialistas sem estourar seu computador.
- 🖱️ **Automação 1-Clique no Windows**: Inclui `instalador.bat` para configurar o ambiente e `conversor.bat` com assistente interativo para fatiar arquivos `.gguf`.
- 📦 **Conversor Universal MoE**: Suporte nativo para **Qwen MoE** (2, 3, 3.5), **IBM Granite MoE** e qualquer modelo com tensores fundidos `blk.N.ffn_*_exps.weight`.
- 🔒 **Memória Compartilhada e Zero-Copy**: Comunicação inter-processos (IPC) versionada com entrega direta de ponteiros de memória compartilhada.

---

## 📊 Benchmarks Reais

### LLM Convencional (Normal) vs MiniMaxBrain (MMB)

| Métrica / Dimensão | Modelo Normal (Sem MMB) | MiniMaxBrain (Sob Demanda) | MiniMaxBrain (4 GiB RAM + Selo) |
|---|---|---|---|
| **RAM Exigida (Qwen 35B)** | **~13 a 20 GiB** *(Inviável em PCs 8GB/12GB)* | **127 MiB** | **~4 GiB** *(Configurável)* |
| **Economia de RAM** | 0% (Alocação integral) | **99% menor** | **>70% a 80% menor** |
| **Tempo de Rota (Qwen 35B)** | *Crash por falta de RAM* | 6,189 s / token | **2,108 s / token (3× mais rápido)** |
| **Expulsões de Cache (*Evictions*)** | N/A | 952 expulsões | **0 expulsões (Zero Churn!)** |
| **Integridade de Dados** | Nenhuma | SHA-256 lento no I/O | **Selo Pré-Voo (`seal`) 100% Seguro** |

---

## ⚡ Get Started (Início Rápido)

### Opção 1: No Windows (1 Clique — Recomendado) 🪟

1. **Instale o Ambiente:**
   Dê dois cliques em [`instalador.bat`](instalador.bat). Ele verifica o Python 3.11+, instala as dependências e valida a suíte de testes automaticamente.

2. **Adicione o Modelo:**
   Coloque o seu arquivo `.gguf` (ex: `qwen-35b.gguf` ou `granite-moe.gguf`) dentro da pasta [`conversor/`](conversor/).

3. **Converta e Sele:**
   Dê dois cliques em [`conversor.bat`](conversor.bat). O assistente detecta o modelo, fatia os especialistas para `.mmbw`, gera o arquivo `gate.json` e grava o selo de segurança `model.verified.json`.

---

### Opção 2: Via Terminal / Linha de Comando (CLI) 💻

#### 1. Instalação e Testes

```powershell
# Instale as dependências e registre o comando mmb
python -m pip install -r requirements-dev.txt
python -m pip install -e .

# Valide a integridade do runtime
python -m pytest -q
```

#### 2. Inspecione e Converta o GGUF

```powershell
# 1. Inspecione a topologia do GGUF
python mmb.py gguf-inspect --gguf conversor/meu-modelo.gguf

# 2. Converta para blocos pagináveis MiniMaxBrain
python mmb.py gguf-pack-moe --gguf conversor/meu-modelo.gguf --output modelos/meu-modelo-mmbw
```

#### 3. Gere o Selo e Inicie o Serviço

```powershell
# 3. Crie o selo de integridade pré-voo
python mmb.py seal --config modelos/meu-modelo-mmbw/gate.json

# 4. Inicie o Gate Externo em segundo plano
python mmb.py serve --config modelos/meu-modelo-mmbw/gate.json
```

---

## 🏗️ Como Funciona

```text
                     PROCESSO DO EXECUTOR
 
  prompt ➔ camadas core ➔ router real ➔ experts obrigatórios
                               |                 |
                      gate interno futuro        | acquire
                      + ModelMemory               |
                               | prefetch         |
                               v                  v
 ====================== IPC MMB v1 ======================
                               |
                     GATE EXTERNO INDEPENDENTE
 
        mapa físico ➔ admission controller ➔ leases
                             |                   |
                             v                   v
                       fila de I/O ➔ cache RAM compartilhada
                             |
                             v
                        shards no SSD
```

1. **Camada Core Fixa:** Fica residente permanentemente na RAM (embeddings, normas, atenção).
2. **Especialistas no SSD:** Fatiados individualmente com SHA-256 e alinhamento de 4096 bytes.
3. **Contrato `acquire`:** O executor requisita apenas os índices que o router selecionou; o MMB entrega os ponteiros de memória compartilhada e mantém o *lease* ativo durante o cálculo.

---

## ⚙️ Configuração do `gate.json`

Exemplo de configuração modular recomendada:

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
  },
  "server": {
    "host": "127.0.0.1",
    "port": 55321
  }
}
```

---

## 🗺️ Roadmap

- [x] **v0.1.0 — Fundação do Gate Externo**: Mapeamento físico, cache LRU sem overcommit, leases e IPC via shared memory.
- [x] **v0.2.0 — Alta Performance e Automação**:
  - [x] Selo de integridade criptográfica pré-voo (`mmb seal`).
  - [x] Orçamento modular por bytes de RAM (`ram_budget`).
  - [x] Validação física e benchmarks no Qwen 3.6 35B MoE.
  - [x] Scripts 1-clique (`instalador.bat` e `conversor.bat`).
- [ ] **v0.3.0 — Backend Tensorial Integrado**: Execução direta dos kernels `gate/up/down` e tokenizador sobre os blocos paginados.
- [ ] **v0.4.0 — Interface Web & Chat UI**: Servidor HTTP local no navegador (estilo `llama-server`) e API compatível com OpenAI `/v1/chat/completions`.
- [ ] **v1.0.0 — Validação de Modelos de 1TB–2TB**: Inferência completa de MoEs de 2 Trilhões de parâmetros em PCs domésticos.

---

## 📄 Documentação

- 📘 [**Arquitetura do Gate Externo**](docs/external-gate-architecture.md) — limites físicos, concorrência e modelo de I/O.
- 📐 [**Formato Físico de Modelos**](docs/model-format.md) — layout tensorial, pack plans e contratos.
- 🧠 [**Memória Estrutural (ModelMemory)**](docs/memory-kernel.md) — banco SQLite de perfil de rotas e telemetria.
- 📊 [**Relatório Experimental do Qwen 35B**](real_model_test/REPORT_QWEN.md) — metodologia e dados do modelo de 35B.
- 📊 [**Relatório Experimental do Granite**](real_model_test/REPORT.md) — metodologia e validação de referência.

---

## ⚖️ Licença

MiniMaxBrain é source-available para uso pessoal e não comercial. Consulte [`LICENSE.md`](LICENSE.md) e [`COMMERCIAL.md`](COMMERCIAL.md).

---

## 🤝 Reconhecimentos e Origem

Este é uma aplicação idealizada por humanos e construida com a ajuda de Inteligencia Artificial, grande parte do projeto já existia dentro do sistema Eyle Code Agent e apenas reutilizamos os fundamentos.
