# Relatório Experimental — Qwen 3.6 35B A3B no MiniMaxBrain

Validação experimental do **Porteiro Externo MiniMaxBrain** usando pesos reais particionados do **Qwen3.6 35B A3B MoE**.

---

## 📌 Especificações do Modelo Testado

| Propriedade | Valor Observado |
|---|---:|
| **Modelo** | Qwen 3.6 35B A3B |
| **Arquitetura GGUF** | `qwen35moe` |
| **Parâmetros Totais** | 35.505.251.456 (~35,5 Bilhões) |
| **Parâmetros Ativos por Token** | ~3.300.000.000 (~3,3 Bilhões) |
| **Quantização** | GGUF Q4 / ftype 29 (~2,92 bits/peso) |
| **Camadas MoE** | 41 camadas |
| **Total de Blocos Físicos** | 11.126 (630 core + 10.496 experts) |
| **Tamanho dos Pesos no SSD (`.mmbw`)** | 12.953.412.096 bytes (~12,95 GB) |
| **Tamanho Médio do Especialista** | 3.342.336 bytes (~3,34 MB) |

---

## 📊 Comparativo Geral: LLM Convencional vs MiniMaxBrain

| Dimensão | Modelo Normal (Sem Porteiro) | MMB Inicial (16 experts) | MMB Otimizado (4 GiB RAM + Selo) |
|---|---:|---:|---:|
| **Memória RAM Exigida** | **~13 a 20 GiB (Inviável em PCs 8GB/12GB)** | **127 MiB** | **4.096 MiB (Modular)** |
| **Economia de RAM** | 0% (Alocação integral) | **99,0% menor** | **~70% a 80% menor** |
| **Tempo da Rota Física** | *Out of Memory em PC 8GB/12GB* | 6,189 s | **2,108 s** |
| **Expulsões de Cache (*Evictions*)** | 0 | 952 expulsões | **0 expulsões (Zero!)** |
| **Integridade** | Confiança cega | SHA-256 no I/O (gargalo de CPU) | **Selo Criptográfico Pré-Voo (`seal`)** |

---

## 🔬 Bateria de Testes Experimentais

### 1. Teste de Configuração Fixa (16 Especialistas Residentes)
* **Configuração:** `gate.experts.json` com `resident_experts: 16` e integridade `first_load`.
* **Cenário:** Simulação severa de restrição de memória (apenas 16 especialistas simultâneos em RAM para um modelo com 10.496 especialistas).

| Métrica | Somente sob demanda | Prefetch da próxima camada | Efeito do Prefetch |
|---|---:|---:|---:|
| **Tempo mediano da rota** | 6,189 s | **4,877 s** | **−21,2%** (1,269× mais rápido) |
| **Acquire médio por camada** | 71,844 ms | **55,941 ms** | −22,1% |
| **Bytes lidos no SSD** | 2.230.411.776 (~2,23 GB) | 2.565.448.192 (~2,56 GB) | **+15,0%** |
| **Expulsões (*Evictions*)** | 640 | 952 | +48,7% |

* **Conclusão:** O prefetch reduz a latência em mais de 21%, mas o limite de apenas 16 especialistas causa alta rotatividade (*churn*) no cache, forçando o SSD a reler dados expulsos.

---

### 2. Teste Modular com Orçamento de RAM (`4 GiB`) + Selo de Integridade
* **Configuração:** `gate.ram.json` com `ram_budget: "4GiB"` e `integrity: "seal"`.
* **Cenário:** O MMB calcula dinamicamente que 4 GiB comportam até ~1.044 especialistas residentes sem estourar o limite da máquina.

| Métrica | Sem Ajuda (Sob Demanda) | Com Prefetch da Próxima Camada |
|---|---:|---:|
| **Tempo mediano da rota física** | 2,323 s | **2,108 s** |
| **Tempo médio de acquire por camada** | 23,64 ms | **19,72 ms** |
| **Expulsões de cache (*Evictions*)** | **0 expulsões** | **0 expulsões** |
| **Bytes lidos no SSD** | 2.230.411.776 (~2,23 GB) | 2.230.411.776 (~2,23 GB) |
| **Status do Selo** | Válido e Verificado | Válido e Verificado |

* **Conclusão:** Ao conceder 4 GiB de orçamento de RAM e utilizar o selo pré-verificado (`seal`), as expulsões caíram para **zero absoluto** e a latência de rota caiu de **6,18 s para 2,10 s** (quase 3× mais rápido), com 100% de segurança de integridade.

---

## 🛠️ Como Reproduzir os Testes do Qwen

1. **Gere o Selo de Integridade:**
   ```powershell
   python mmb.py seal --config real_model_test\mmb-qwen-pageable\gate.ram.json
   ```

2. **Valide a Configuração e o Selo:**
   ```powershell
   python mmb.py check --config real_model_test\mmb-qwen-pageable\gate.ram.json
   ```

3. **Execute a Comparação A/B com 2 Tokens:**
   ```powershell
   python tools\mmb_ab_compare.py --config real_model_test\mmb-qwen-pageable\gate.ram.json --tokens 2 --rounds 2
   ```

---

## 💡 Lições Arquiteturais do Qwen 35B

1. **Escalabilidade Comprovada:** O Porteiro Externo gerencia com sucesso um grafo de mais de 10.000 especialistas em 13 GB de tensores MoE mantendo o consumo de memória rigorosamente controlado.
2. **Orçamento Modular por RAM vence Slots Fixos:** Dedicar uma fatia de RAM em bytes (`ram_budget: 4GiB`) permite que a cache aproveite a memória disponível para zerar expulsões desnecessárias.
3. **Selo de Integridade Pré-Voo é Mandatório em Modelos Grandes:** Em modelos de dezenas de bilhões de parâmetros, calcular SHA-256 no carregamento de cada bloco estrangula a CPU. O selo pré-computado permite que o NVMe entregue dados em sua velocidade bruta máxima sem abrir mão da segurança contra arquivos corrompidos.
