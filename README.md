# MiniMaxBrain (MMB)

MiniMaxBrain é um runtime independente de paginação de pesos para LLMs esparsas/MoE. Ele mantém no SSD os blocos que não cabem na RAM, admite somente os pesos obrigatórios para o cálculo atual e permite escolher o limite por bytes ou por número de especialistas residentes.

O componente entregue hoje é o **gate externo**. Ele é comparável ao papel operacional de um `llama.exe` apenas no controle físico dos pesos: ainda não contém os kernels tensoriais necessários para gerar tokens diretamente sobre os blocos paginados.

## Estado real do projeto

Implementado e testado:

- mapa físico estrito de shards, offsets, hashes, camadas e especialistas;
- conversor streaming de GGUF GraniteMoE para blocos pagináveis;
- orçamento exato por RAM ou quantidade de especialistas residentes;
- cache LRU com reservas, blocos fixos, leases e sem overcommit oculto;
- leitura SSD → RAM somente dos ranges solicitados;
- prefetch assíncrono, coalescência e promoção para pedido obrigatório;
- processo externo com IPC versionado e entrega por memória compartilhada;
- `ModelMemory`: grafo estrutural persistente, revisionado e paginado;
- perfil físico de hits, misses e espera de prefetch por rota;
- teste com um modelo MoE real e comparação A/B com e sem ajuda.

Ainda não implementado:

- backend tensorial que calcula `down/gate/up` diretamente sobre os blocos do gate;
- ligação dos índices reais do router do modelo ao `acquire_routes`;
- gate interno neural que conhece hidden states, logits e a topologia completa;
- geração ponta a ponta de tokens através do MMB;
- validação física de um modelo de 1–2 TB.

## Arquitetura

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

O gate externo é a autoridade sobre a residência física. Ele não escolhe significado, não altera o `top-k` do modelo e nunca troca um especialista obrigatório por outro. O gate interno será apenas um conselheiro de prefetch; a seleção matemática real continuará vindo do router.

## Comparação real: com e sem o gate

Teste executado em 20 de agosto de 2026 com **IBM Granite 3.0 1B-A400M Instruct Q4_K_M**:

| Propriedade | Valor |
|---|---:|
| Parâmetros contados | 1.334.628.352 |
| Tamanho do GGUF | 821.845.024 bytes |
| Camadas MoE | 24 |
| Especialistas por camada | 32 |
| Especialistas ativos por camada | 8 |
| Blocos MMB | 170 core + 768 experts |

### Modelo integral, sem MMB

O modelo completo, que cabe na máquina de teste, gerou **25,00 ± 7,43 tokens/s** e atingiu aproximadamente **1.024,12 MiB** de working set. Para um modelo que já cabe em RAM, manter tudo residente é a opção correta para velocidade.

### MMB externo, com os mesmos pesos reais

O harness paginou as rotas determinísticas de 2 tokens, 24 camadas e 8 especialistas por camada. Essa etapa mede o trajeto físico dos pesos; ela ainda **não é geração de tokens**.

| Limite | Sob demanda | Prefetch exato da próxima camada | Ganho de latência | Custo em bytes lidos |
|---|---:|---:|---:|---:|
| 16 experts residentes | 1,960 s | **1,466 s** | **25,20%** | +32,46% |
| 192 MiB totais | 1,915 s | **1,256 s** | **34,43%** | +25,11% |

Uma medição isolada do processo de paginação sob demanda observou **127,15 MiB** de pico de working set, contra **1.024,12 MiB** no processo integral: redução observada de **87,58%**. Essa comparação não inclui um backend consumidor materializando KV cache e scratch; portanto não é a RAM final de uma inferência integrada.

O resultado honesto é:

- sem gate venceu em velocidade porque o modelo pequeno cabe inteiro;
- com gate venceu em residência física de pesos;
- ajuda de rota perfeita reduziu a espera entre 25,20% e 34,43%, mas aumentou I/O e expulsões;
- quando o modelo cabe, o produto deverá usar modo full-resident/bypass;
- quando não cabe, o gate é primeiro um mecanismo de viabilidade, não uma aceleração garantida.

Os dados completos, metodologia e comandos estão em [`real_model_test/README.md`](real_model_test/README.md) e [`real_model_test/COMPARISON.md`](real_model_test/COMPARISON.md).

## Um modelo de 1 TB pode rodar em 12 GB?

Pode ser fisicamente possível se todas estas condições forem verdadeiras:

1. o modelo é esparso e seus pesos obrigatórios são particionáveis;
2. a parte compartilhada, o KV cache, scratch e o maior conjunto simultâneo de experts cabem nos 12 GB;
3. o executor informa ao gate quais blocos o router realmente selecionou;
4. o backend aceita executar usando os pesos entregues pelo gate;
5. a localidade e a banda do SSD tornam o volume de misses aceitável.

Não é possível prometer que qualquer modelo denso de 1 TB funcionará bem. Paginar quase 1 TB por token pode fazê-lo “caber”, mas tende a tornar a geração impraticavelmente lenta. Um processo completamente caixa-preta, observando apenas prompt e texto, também não consegue descobrir quais pesos internos são obrigatórios.

Para estimar um limite inferior físico:

```powershell
python mmb.py estimate --parameters 2T --bits 4 --expert-blocks 900 `
  --cold-blocks-per-token 10 --storage-bandwidth 7GB
```

## A memória útil preservada no MMB

`ModelMemory` é a parte útil de memória para este produto. Ela não armazena chat nem injeta conteúdo em prompts. O banco SQLite guarda:

- identidade exata de modelo e revisão do mapa;
- nós compactos de modelo, camada, expert, bloco, tensor e assinatura de rota;
- relações revisionadas entre os nós;
- histórico append-only para auditoria;
- hits, misses, bytes solicitados e latência por `(layer, expert)`;
- consultas explícitas com limite e cursor.

O grafo pode crescer no SSD sem ser carregado inteiro na RAM. Isso permitirá ao gate interno oferecer apenas a vizinhança estrutural relevante ao gate externo.

Ative no arquivo de configuração:

```json
"model_memory": {
  "enabled": true,
  "path": "state/model-memory.sqlite3"
}
```

Veja [`docs/memory-kernel.md`](docs/memory-kernel.md).

## Instalação

Requer Python 3.11 ou superior. O runtime usa somente a biblioteca padrão; `pytest` é necessário para desenvolvimento.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Também é possível instalar o comando `mmb`:

```powershell
python -m pip install -e .
mmb --help
```

## Início rápido

1. Inspecione um GGUF suportado:

```powershell
python mmb.py gguf-inspect --gguf modelo.gguf
```

2. Converta GraniteMoE em blocos MMB:

```powershell
python mmb.py gguf-pack-moe --gguf modelo.gguf --output modelo-mmb
```

3. Copie `mmb.example.json` para `mmb.json`, ajuste `model_map` e o orçamento, e valide:

```powershell
python mmb.py check --config mmb.json
python mmb.py smoke --config mmb.json --blocks 1
```

4. Para uso em outro processo, configure `transport` como `shared_memory` e execute:

```powershell
python mmb.py serve --config mmb.json
```

O executor usa `ExternalGateClient` para `prefetch`, `acquire`/`acquire_routes` e `release`.

## Orçamento

O orçamento precisa satisfazer:

```text
RAM >= core fixo + KV cache + scratch + conjunto obrigatório simultâneo
```

Exemplo por bytes:

```json
"memory": {
  "ram_budget": "12GiB",
  "resident_experts": null,
  "kv_cache": "2GiB",
  "scratch": "1GiB",
  "transport": "shared_memory",
  "lease_timeout_seconds": 120
}
```

Exemplo por quantidade de experts:

```json
"memory": {
  "ram_budget": null,
  "resident_experts": 16,
  "kv_cache": "2GiB",
  "scratch": "1GiB",
  "transport": "shared_memory",
  "lease_timeout_seconds": 120
}
```

No segundo modo, o MMB calcula o orçamento conservador usando o maior bloco de especialista.

## Estrutura do repositório

```text
minimaxbrain/                  runtime do gate externo e ModelMemory
tests/                         testes focados do MMB
tools/                         smoke real, IPC e comparação A/B
real_model_test/               resultados e contratos do modelo real
docs/external-gate-architecture.md
docs/model-format.md
docs/memory-kernel.md
mmb.py                         entrada CLI sem instalação
mmb.example.json               configuração de referência
```

## Limites e segurança

- paths de mapas, shards e banco são confinados ao diretório da configuração;
- offsets, sobreposição, tamanho e SHA-256 são validados;
- bind fora de loopback exige token;
- blocos leased ou core não podem ser expulsos;
- prefetch é conselho falível; `acquire` é necessidade obrigatória;
- versão ou campo desconhecido falha de forma fechada.

Leia [`SECURITY.md`](SECURITY.md) antes de expor o serviço.

## Documentação

- [`docs/external-gate-architecture.md`](docs/external-gate-architecture.md) — arquitetura e limites físicos;
- [`docs/model-format.md`](docs/model-format.md) — mapa, pack plan, configuração e IPC;
- [`docs/memory-kernel.md`](docs/memory-kernel.md) — memória estrutural revisionada;
- [`real_model_test/README.md`](real_model_test/README.md) — teste real completo;
- [`real_model_test/COMPARISON.md`](real_model_test/COMPARISON.md) — comparação com e sem ajuda.

## Licença

MiniMaxBrain é source-available para uso pessoal e não comercial. Consulte [`LICENSE.md`](LICENSE.md) e [`COMMERCIAL.md`](COMMERCIAL.md).
