# ModelMemory — memória estrutural do MiniMaxBrain

## Objetivo

`ModelMemory` mantém em SQLite uma representação reduzida da topologia do modelo e do comportamento físico das rotas. Ela serve a dois consumidores:

- o gate externo aprende localidade, hits, misses e custo por expert;
- o futuro gate interno consulta a vizinhança necessária sem materializar toda a estrutura da LLM.

Ela não é memória conversacional, banco vetorial, mecanismo de prompt ou autoridade semântica.

## Princípios preservados

1. **Crescimento não implica materialização.** O grafo pode crescer no SSD; toda leitura pública possui limite explícito.
2. **Identidade é revisionada.** Cada banco separa observações pela identidade física e pela revisão exata do mapa.
3. **Autoridade mecânica e conselho são separados.** Estatísticas podem ordenar prefetch, mas nunca substituem o `acquire` obrigatório do router real.
4. **Ativação é explícita.** Não existe expansão recursiva ou projeção automática do grafo.
5. **Histórico é auditável.** Criações, revisões e observações geram eventos append-only.

## Schema v1

Identidade: `mmb-model-memory-v1`.

O arquivo contém cinco conjuntos principais:

| Tabela | Papel |
|---|---|
| `models` | modelo, arquitetura, caminho e `map_revision` exatos |
| `nodes` | modelo, layer, expert, block, tensor ou route signature |
| `edges` | relações direcionadas revisionadas |
| `route_stats` | requests, hits, misses, prefetch waits, bytes e latência |
| `events` | histórico append-only fixado à revisão do mapa |

Ao registrar um `PhysicalModelMap`, o MMB cria automaticamente:

```text
model:<id>
  ├─ contains -> block:<core-id>
  └─ contains -> layer:<n>
                    └─ contains -> expert:<layer>:<expert>
```

Tensores internos e assinaturas compactas podem ser adicionados posteriormente com `put_node` e `put_edge`.

## Concorrência e revisões

Nós e relações começam na revisão 1. Uma alteração exige `expected_revision`; divergência falha sem sobrescrever o estado atual.

```python
from minimaxbrain import ModelMemory, load_model_map

model_map = load_model_map("model/model.mmb-map.json")
with ModelMemory("state/model-memory.sqlite3", model_map) as memory:
    node = memory.put_node(
        "tensor:router:0",
        "tensor",
        {"shape": [1024, 32], "encoding": "q4"},
    )
    memory.put_node(
        "tensor:router:0",
        "tensor",
        {"shape": [1024, 32], "encoding": "q4", "verified": True},
        expected_revision=node["revision"],
    )
```

Abrir o mesmo banco com outra revisão do mapa cria um domínio físico separado. Estatísticas de uma versão nunca são aplicadas silenciosamente a outra.

## Perfil físico de rotas

Quando `model_memory.enabled=true`, `ExternalGate.acquire` registra automaticamente experts observados:

```text
state = hit | miss | prefetch_wait
requested_bytes += tamanho do bloco
admitted_bytes += tamanho do bloco somente em miss obrigatório
total_latency_ms += custo observado da admissão
```

Consulta:

```python
profile = memory.route_profile(layer=8, limit=32)
```

O resultado é ordenado por quantidade de pedidos. Ele é um sinal mecânico para cache/prefetch, não uma afirmação sobre significado ou qualidade da saída.

## Paginação do grafo

`neighbors` retorna no máximo 256 relações por chamada:

```python
page = memory.neighbors("layer:8", limit=64)
while page["next_cursor"] is not None:
    page = memory.neighbors(
        "layer:8",
        limit=64,
        cursor=page["next_cursor"],
    )
```

Não há expansão implícita para vizinhos de vizinhos. O consumidor decide qual fronteira abrir e quanto do grafo cabe em seu próprio orçamento.

O histórico segue a mesma regra com `history(cursor=..., limit=...)`.

## Configuração

```json
{
  "model_memory": {
    "enabled": true,
    "path": "state/model-memory.sqlite3"
  }
}
```

O caminho é relativo ao arquivo de configuração e não pode escapar desse diretório. SQLite usa foreign keys, WAL e `synchronous=FULL`.

## Política de falha

ModelMemory é um conselheiro persistente. Se o registro de uma observação falhar depois de o peso obrigatório já ter sido admitido, o gate emite telemetria de erro e preserva a execução. Uma falha de memória nunca transforma um expert correto em outro expert.

Inconsistência de schema, identidade física conflitante, endpoint de relação inexistente ou conflito de revisão falham explicitamente com `MODEL_MEMORY_INVALID`.

## Limite atual

O mapa externo conhece blocos físicos, não a semântica completa de cada tensor. A topologia fina do gate interno deverá ser construída por um adaptador de arquitetura e gravada como nós `tensor`/`route_signature`. O schema genérico já suporta isso; a política de redução e treinamento ainda não foi implementada.
