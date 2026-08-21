# Gate Externo MiniMaxBrain — arquitetura física v1

## Estado desta entrega

Esta revisão implementa primeiro o **gate externo independente**, conforme a ordem definida pelo projeto. Ele já possui:

- mapa físico estrito de blocos e shards;
- empacotamento streaming em ranges alinhados;
- leitura de somente o range solicitado;
- verificação SHA-256 fail-closed;
- orçamento por bytes de RAM ou por quantidade máxima de especialistas residentes;
- cache LRU consciente de blocos especulativos, blocos fixos e leases ativos;
- fila assíncrona de prefetch e promoção de miss para admissão obrigatória;
- IPC local versionado;
- entrega zero-JSON dos pesos por memória compartilhada;
- cliente para o executor tensorial e para o futuro porteiro interno;
- `ModelMemory` estrutural, revisionada e limitada por páginas;
- telemetria física separada da memória estrutural;
- CLI de inspeção, empacotamento, validação, smoke, estimativa e serviço.

O gate interno e os kernels tensoriais ainda não são implementados. O executável atual é o runtime físico que um backend de inferência usa; ele não gera tokens sozinho.

## Origem arquitetural

Esta arquitetura deriva da própria MiniMaxBrain e de primeiros princípios de inferência:

| Princípio do MMB | Aplicação física |
|---|---|
| autoridade mecânica explícita | Admission Controller decide somente residência física |
| grafo grande sem materialização automática | modelo grande sem residência automática |
| fronteiras limitadas e alcançáveis | mapa e cursores mantêm blocos/nós frios alcançáveis |
| revisão fixada à evidência | mapa, memória, configuração e IPC possuem identidades estritas |
| conselho separado de obrigação | `prefetch` pode falhar; `acquire` preserva a matemática |
| telemetria não é significado | hits/misses informam custo, não escolhem a resposta |

Nenhum desenho, algoritmo ou contrato do projeto Warp foi usado. Ele foi deliberadamente excluído desta arquitetura.

## A divisão interno/externo

```text
                       PROCESSO DO MODELO

 prompt -> core residente -> roteador real -> seleção obrigatória
                                      |              |
                         futuro gate interno          |
                         (conselheiro compacto)        |
                                      | prefetch      | acquire
                                      v               v
========================== IPC MMB v1 =========================
                                      |
                       PROCESSO DO PORTEIRO EXTERNO

        contrato -> Admission Controller -> leases
                            |                  |
                            v                  v
                     fila de I/O -> cache RAM compartilhada
                            |
                            v
                      shards no SSD
```

O futuro porteiro interno conhece a topologia reduzida, estado oculto, logits e horizonte de execução. O externo conhece apenas identidade física:

- `block_id`;
- `layer` e `expert` como endereço opcional, nunca como significado;
- shard, offset e comprimento;
- hash e alinhamento;
- estado físico: ausente, carregando, residente, leased ou fixo.

Essa fronteira permite construir e testar o externo antes de existir qualquer predictor neural.

## Duas classes de pedido

O contrato distingue autoridade de conselho.

### `prefetch`: conselho falível

Pode vir do porteiro interno, de um histórico mecânico ou de um perfil previamente medido. O externo pode:

- reordenar por prioridade;
- coalescer duplicatas;
- descartar quando a fila estiver cheia;
- carregar e depois descartar sem que o bloco seja usado;
- rejeitar IDs desconhecidos.

Um erro de prefetch só afeta latência e bytes lidos.

### `acquire`: necessidade obrigatória

Vem da seleção real do executor tensorial. O externo deve:

1. validar o ID contra a revisão exata do mapa;
2. promover um prefetch em andamento ou fazer uma leitura síncrona;
3. obter um lease antes que outro bloco possa expulsá-lo;
4. devolver o descritor de memória compartilhada;
5. manter o bloco residente até `release` ou expiração do lease;
6. falhar explicitamente se o orçamento físico não puder satisfazer o conjunto.

O externo nunca substitui silenciosamente o especialista pedido por outro mais barato.

## Autoridade semântica e `top-k`

Há três números diferentes:

- **especialistas totais**: capacidade armazenada no SSD;
- **especialistas residentes**: blocos presentes no cache naquele instante;
- **especialistas ativos**: especialistas escolhidos pelo roteador para o cálculo.

`resident_experts=4` limita residência, não muda um modelo treinado com `top-k=8` para `top-k=4`. Reduzir ativação altera a função matemática do modelo. A futura integração só poderá expor `top-k` configurável se o contrato do backend e o manifesto semântico interno declararem que esse valor foi treinado e suportado.

## Invariantes físicos

1. O modelo inteiro nunca é lido para descobrir um bloco.
2. Todo range reside dentro do diretório do modelo e dentro do tamanho do shard.
3. Ranges no mesmo shard não podem se sobrepor.
4. RAM comprometida é `residente + reservado em I/O`; ambos contam no orçamento.
5. Bloco fixo ou leased não pode ser expulso.
6. Alocação acontece depois de uma reserva; não existe overcommit escondido.
7. Um bloco corrompido não entra no cache.
8. Conselho especulativo não pode mudar o resultado semântico.
9. Contratos com campos ou versões desconhecidas falham fechados.
10. Um cliente separado recebe apenas um handle compartilhado, nunca pesos codificados em JSON.

## Orçamento real de 12 GB

O teste de capacidade é:

```text
R >= C + K + S + W
```

onde:

- `R`: RAM disponível ao porteiro;
- `C`: blocos compartilhados/core que precisam ficar fixos;
- `K`: KV cache da janela e do batch configurados;
- `S`: scratch, buffers de desquantização e backend;
- `W`: maior conjunto simultâneo obrigatório de especialistas.

O v1 valida conservadoramente `C + maior_bloco` porque o mapa externo ainda não contém o contrato semântico de `top-k` por camada. O backend deve pedir em uma única aquisição todos os blocos que precisam coexistir. Se o conjunto não couber, a admissão falha em vez de travar ou ultrapassar a RAM.

Modo por especialistas:

```text
R = K + S + C + N * maior_bloco_especialista
```

O maior bloco, e não a média, é usado para que `N` slots sejam uma promessa física verdadeira.

## É possível fazer tudo externamente?

### Sim, para capacidade e correção, com uma condição

É possível manter 1 TB no SSD e executar com 12 GB se o modelo for **esparso e fisicamente particionável** e se o executor tensorial enviar ao externo a seleção real de blocos antes do cálculo. O porteiro interno não é obrigatório para correção: um miss apenas bloqueia enquanto o externo carrega o especialista obrigatório.

Esse hook no executor é indispensável. Um programa completamente externo observando apenas prompt e texto gerado não sabe qual expert uma camada selecionou e não tem como fornecer o peso correto. Portanto:

```text
externo ao modelo em pesos e processo      possível
externo ao grafo, mas com contrato acquire possível
caixa-preta sem hook de blocos              impossível
```

### Não, para transformar qualquer modelo em um modelo de 12 GB

Um transformer denso precisa de todas as matrizes obrigatórias em sequência. Paginar essas matrizes pode fazê-lo caber no sentido de memória virtual, mas exige reler aproximadamente o modelo inteiro por token e tende a ser impraticável.

Também não funciona quando `C + K + S + W > 12 GB`. Se, por exemplo, a parte não-especialista quantizada tiver 20 GB, nenhum cache de experts corrige o déficit.

### O limite de velocidade continua sendo I/O

Para `D` bytes frios por token e banda sustentável `B`:

```text
tempo_IO/token >= D / B
tokens/s <= B / D
```

Isso é um teto otimista: não inclui latência aleatória, compute, cópia para GPU, desquantização, KV nem contenção.

Exemplo puramente físico: 2T em 4 bits ocupa aproximadamente 1 TB antes de metadados. Se 900 blocos dividissem quase todos os pesos igualmente, cada bloco teria cerca de 1,1 GB. Em SSD sustentando 7 GB/s:

- 1 bloco frio/token: teto de I/O próximo de 6 tokens/s;
- 10 blocos frios/token: teto próximo de 0,6 token/s;
- 60 blocos frios/token: teto próximo de 0,1 token/s.

Um MoE comum pode selecionar um ou mais experts **em cada camada MoE**, não apenas um expert global por token. O número que importa é `cold_blocks_per_token`, medido no modelo real.

## Otimização externa sem gate interno

O externo v1 já consegue ganhos grandes quando existe localidade:

- core fixo é carregado uma vez;
- experts quentes sobrevivem entre tokens;
- prefetches duplicados são coalescidos;
- blocos especulativos são expulsos antes de blocos usados;
- especulação nunca expulsa um bloco aquecido por admissão obrigatória;
- leitura é contígua e alinhável por bloco;
- I/O ocorre em paralelo com o compute quando o backend envia horizonte;
- o executor mapeia o mesmo payload compartilhado sem cópia JSON;
- hashes são amortizados com `first_load` quando desejado;
- orçamento por bytes impede paginação acidental do próprio processo.

Sem predictor interno, as fontes permitidas de prefetch são mecânicas: rota anterior da mesma sessão, perfis observados para um prefixo exato e hints do próprio backend. Elas melhoram casos repetitivos, mas o primeiro caminho inesperado continua sendo miss.

## O papel futuro do porteiro interno

O interno não será dono da RAM. Ele será um advisor dentro do grafo:

```text
entrada reduzida:
  token/posição
  layer atual
  hidden-state comprimido
  logits do router
  mapa estrutural reduzido
  estado físico resumido

saída:
  block_id
  prioridade
  earliest_step
  deadline_step
  confiança
  revisão do mapa
```

O externo não precisa entender esses dados internos. Recebe somente a lista ordenada de `block_id`. A seleção real ainda chega como `acquire` separado. Assim, predictor errado não muda tokens.

## Componentes implementados

```text
model_map.py   mapa físico, confinamento, ranges e identidade
model_memory.py grafo reduzido, revisões, histórico e perfil de rotas
packer.py      empacotamento streaming alinhado
gguf.py        leitura estrita do diretório e dos ranges GGUF
gguf_moe.py    separação streaming GraniteMoE em blocos por especialista
config.py      orçamento current-only e teste de viabilidade
storage.py     leitura por range, integridade e arenas
cache.py       reserva, LRU, pin, leases e descarte
scheduler.py   prefetch assíncrono, coalescência e promoção
external.py    autoridade de admissão e sessões de lease
protocol.py    wire estrito v1
server.py      serviço TCP local JSON-lines
client.py      cliente e mapeamento dos handles compartilhados
telemetry.py   eventos físicos fora da memória estrutural
cli.py         superfície operacional
```

## Fluxo de uma etapa

1. O backend inicia o cálculo usando core já fixo.
2. Um advisor opcional envia `prefetch` ordenado.
3. O roteador real produz a seleção daquela camada.
4. O backend envia `acquire` com todos os blocos que precisam coexistir.
5. Hit retorna imediatamente; prefetch em curso é aguardado; miss vira leitura obrigatória.
6. O backend mapeia o handle, copia para GPU ou executa na CPU.
7. Após o kernel terminar, o backend envia `release`.
8. O externo pode expulsar o bloco se outro pedido precisar do orçamento.

## Formato físico

O packer recebe blocos que um conversor específico do modelo já separou. Ele escreve shards grandes contendo ranges alinhados:

```text
model/
  model.mmb-map.json
  shards/
    model-00000.mmbw
    model-00001.mmbw
```

O packer não tenta inferir onde termina um expert dentro de formatos arbitrários. Essa é responsabilidade do conversor do backend, pois layout tensorial é parte do contrato daquela arquitetura.

Veja `docs/model-format.md`.

## Operação

```powershell
python mmb.py inspect --model-map model/model.mmb-map.json
python mmb.py check --config mmb.json
python mmb.py smoke --config mmb.json --blocks 1
python mmb.py serve --config mmb.json
```

Estimativa transparente do caso hipotético:

```powershell
python mmb.py estimate --parameters 2T --bits 4 --expert-blocks 900 `
  --cold-blocks-per-token 1 --storage-bandwidth 7GB
```

Empacotamento:

```powershell
python mmb.py pack --plan pack-plan.json --output model
```

## Limites do runtime Python v1

O v1 prova contratos, correção, orçamento, IPC e I/O real. Para throughput de produção em centenas de MB por bloco, a mesma arquitetura deve receber um data plane nativo:

- `O_DIRECT`/I/O assíncrono alinhado em Linux;
- IOCP/overlapped I/O no Windows;
- arenas NUMA/pinned;
- backend CUDA/ROCm/CPU vetorizado;
- caminho direto SSD->GPU onde o hardware suportar;
- fila de múltiplas sessões com fairness e batching por expert;
- telemetria de banda sustentável e desgaste do SSD.

Essas trocas ficam atrás dos contratos `FileRangeStore`, `PayloadArena` e IPC. Elas não exigem mover autoridade semântica para o externo.

## Falhas e política

| Falha | Resultado |
|---|---|
| hash incorreto | bloco rejeitado; inferência não continua com peso corrompido |
| bloco desconhecido em prefetch | hint rejeitado; execução continua |
| bloco desconhecido em acquire | erro obrigatório |
| RAM insuficiente | erro explícito com blocos leased preservados |
| cliente morre | lease é reconciliado pelo timeout na próxima aquisição/consulta física |
| fila cheia | prefetch novo é descartado |
| versão/campo desconhecido | contrato rejeitado |
| predictor interno erra | miss/bytes extras, nunca troca de expert |

## Segurança

- paths de shards e fontes do packer ficam confinados ao diretório do contrato;
- serviço usa loopback por padrão;
- bind não-loopback exige token de pelo menos 16 caracteres;
- token é comparado em tempo constante;
- tamanho máximo de requisição é configurável;
- hashes e offsets vêm do mapa versionado;
- o servidor não oferece escrita nos pesos nem operação remota de desligamento;
- blocos compartilhados permanecem apenas enquanto residentes no cache; eviction/close destrói o segmento.

## Próximos marcos

1. Medir o externo contra um MoE pequeno real e registrar `cold_bytes/token`, hit rate e stall.
2. Criar o conversor específico da primeira arquitetura escolhida.
3. Implementar backend tensorial que chama `acquire/release` por camada.
4. Portar storage/arena para data plane nativo sem mudar o IPC.
5. Só então treinar o porteiro interno compacto e comparar ganho adicional sobre o externo puro.
