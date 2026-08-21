# Validação com modelo MoE real — 2026-08-20

## Veredito

O Porteiro Externo foi validado com pesos reais do **IBM Granite 3.0 1B-A400M Instruct**. O teste confirmou três coisas separadas:

1. o GGUF real gera texto em CPU;
2. a MiniMaxBrain separa os tensores MoE fundidos em blocos físicos por `(camada, especialista)` sem desquantizar nem carregar o arquivo inteiro na RAM;
3. um processo externo entrega os especialistas reais a outro processo por memória compartilhada, com SHA-256 conferido no cliente.

Ainda não foi demonstrada geração de tokens **através** do Porteiro. A geração de referência e a paginação MiniMaxBrain usam os mesmos pesos, mas são caminhos separados até que o executor tensorial publique a seleção real do router via `acquire` e consuma os blocos devolvidos.

O executável `llama.cpp` foi usado somente como instrumento independente de referência para provar que o arquivo gera texto. Ele não foi usado como fonte da arquitetura, dos contratos, do cache ou do empacotamento MiniMaxBrain.

## Modelo e identidade física

| Propriedade | Resultado observado |
|---|---:|
| Arquitetura GGUF | `granitemoe` |
| Parâmetros contados nos tensores | 1.334.628.352 |
| Camadas MoE | 24 |
| Especialistas por camada | 32 |
| Especialistas ativos por camada | 8 |
| Quantização | GGUF Q4_K_M / ftype 15 |
| Arquivo GGUF | 821.845.024 bytes |
| SHA-256 do GGUF | `242b06a85482b30fe0b8bc7d70e614e7328d1275dc236648c0e4d71db0cbdbb3` |
| Shard MiniMaxBrain | 820.109.312 bytes |
| SHA-256 do shard MiniMaxBrain | `9319f0cffb686d3587a1179e70aa5ef003f22d59dbc012b26daacc05674499e0` |

Fontes usadas: [modelo oficial IBM Granite](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct), [GGUF Q4_K_M usado no teste](https://huggingface.co/mrutkows/granite-3.0-1b-a400m-instruct-GGUF) e [release oficial do executor de referência](https://github.com/ggml-org/llama.cpp/releases/).

## Geração de referência

Executor: `llama-cli` build `b10218-de699957b`, CPU, contexto 512 e temperatura zero.

Prompt:

> Responda em portugues com uma unica frase curta: o teste real da MiniMaxBrain esta funcionando?

Resposta:

> Sim, o teste real da MiniMaxBrain está funcionando.

Medição reportada pelo executor:

- prompt: 62,5 tokens/s;
- geração: 35,9 tokens/s.

## Conversão física MiniMaxBrain

O adaptador encontrou 72 tensores MoE fundidos: `down`, `gate` e `up` em cada uma das 24 camadas. O eixo final de 32 especialistas é fisicamente contíguo no GGUF. A conversão streaming produziu:

- 170 blocos `core`, fixos: 88.725.976 bytes;
- 768 blocos de especialista: 24 camadas × 32 especialistas;
- maior bloco de especialista: 1.019.904 bytes;
- 938 blocos no total;
- 820.107.736 bytes úteis mapeados;
- contrato `mmb-raw-ggml-expert-concat-v1`;
- hashes SHA-256 individuais para todos os blocos.

Cada bloco de especialista concatena os slices GGML codificados na ordem documentada `down`, `gate`, `up`. O arquivo [`model.mmb-layout.json`](mmb-granite-pageable/model.mmb-layout.json) preserva forma, tipo, comprimento e offsets de cada segmento para o futuro backend tensorial.

## Teste do orçamento por especialistas

Configuração: [`gate.experts.json`](mmb-granite-pageable/gate.experts.json), limite de 16 especialistas residentes, 32 MiB de KV e 32 MiB de scratch.

O benchmark aplicou rotas determinísticas sintéticas sobre os **bytes reais**: 2 tokens × 24 camadas × 8 especialistas.

| Métrica | Resultado |
|---|---:|
| Orçamento total | 172.153.304 bytes |
| Capacidade física dos pesos | 105.044.440 bytes |
| Pesos residentes ao final | 105.044.440 bytes |
| Especialistas residentes | 16 |
| Aquisições de blocos de especialista | 384 |
| Bytes ativos entregues | 365.690.880 |
| Bytes lidos, incluindo core | 454.416.856 |
| Expulsões controladas | 368 |
| Prefetches descartados | 0 |
| Tempo total, incluindo core | 2,793 s |
| Acquire médio por camada | 44,047 ms |
| Acquire p95 | 59,098 ms |

O peso residente foi 105.044.440 / 820.107.736 = **12,8%** do modelo MiniMaxBrain codificado. A memória total reservada, incluindo KV e scratch, foi aproximadamente 164,18 MiB.

## Teste do orçamento por RAM

Configuração: [`gate.ram.json`](mmb-granite-pageable/gate.ram.json), teto fixo de 192 MiB, com as mesmas reservas de KV e scratch.

| Métrica | Resultado |
|---|---:|
| Capacidade física dos pesos | 134.217.728 bytes |
| Especialistas residentes ao final | 46 |
| Aquisições em um token | 192 |
| Bytes lidos, incluindo core | 275.651.032 |
| Expulsões controladas | 150 |
| Ultrapassagens do orçamento | 0 |
| Acquire médio por camada | 47,236 ms |

Isso valida os dois controles solicitados: o operador pode escolher quantidade máxima de especialistas ou um teto absoluto de RAM.

## Teste em processo externo

O serviço foi iniciado com [`gate.shared.json`](mmb-granite-pageable/gate.shared.json) na porta loopback 55323. Um segundo processo:

- negociou `mmb-external-gate-ipc-v1`;
- enviou prefetch de oito especialistas reais;
- adquiriu as rotas da camada 0;
- mapeou oito segmentos de memória compartilhada de 1.019.904 bytes;
- recalculou SHA-256 sobre cada segmento;
- confirmou os oito hashes contra o mapa;
- liberou o lease;
- observou zero expulsões e zero prefetches descartados.

O serviço de teste foi encerrado após a validação.

## O que está e não está provado

Provado agora:

- o modelo real é executável;
- a topologia MoE e os ranges físicos são lidos sem dependências Python externas;
- os especialistas quantizados são separáveis em blocos independentes;
- SSD → RAM, prefetch, lease, integridade, LRU e limites físicos funcionam sobre pesos reais;
- a separação de processos e a entrega por memória compartilhada funcionam.

Ainda pendente:

- ligar os logits/índices do router real ao `acquire_routes`;
- executar os kernels de `down/gate/up` diretamente sobre o payload devolvido;
- medir qualidade e tokens/s com a geração passando pelo porteiro;
- validar a escala de 1–2 TB em hardware com SSD e banda apropriados.

Fazer toda a otimização externamente continua possível no sentido de manter o gate fora do processo do modelo, mas não como caixa-preta: o executor precisa expor a seleção real dos especialistas e aceitar os pesos fornecidos. Sem esse hook, um processo externo não consegue saber quais matrizes são matematicamente obrigatórias.

## Reprodução

```powershell
python -B mmb.py gguf-inspect --gguf real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf
python -B mmb.py gguf-pack-moe --gguf real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf --output real_model_test\mmb-granite-pageable
python -B tools\mmb_real_moe_smoke.py --config real_model_test\mmb-granite-pageable\gate.experts.json --tokens 2
python -B tools\mmb_real_moe_smoke.py --config real_model_test\mmb-granite-pageable\gate.ram.json --tokens 1
```

O comando de conversão recusa sobrescrever um diretório já existente. Para repetir a conversão, use outro diretório de saída.

## Regressão atual

- 22 testes focados do MiniMaxBrain passaram após a limpeza do repositório;
- 14 cobrem mapa, orçamento, cache, integridade, packer, IPC e integração do gate;
- 3 cobrem o leitor/conversor GGUF;
- 5 cobrem `ModelMemory`, revisões, paginação e perfil físico de rotas.

## Comparação A/B

A comparação entre modelo integral, paginação apenas sob demanda e prefetch exato está em [`COMPARISON.md`](COMPARISON.md). O resultado central foi uma troca clara: o modelo integral venceu em velocidade; o Porteiro reduziu fortemente a memória; e o conselho perfeito acelerou a fase de paginação entre 25,20% e 34,43%, ao custo de 25,11% a 32,46% mais bytes lidos.
