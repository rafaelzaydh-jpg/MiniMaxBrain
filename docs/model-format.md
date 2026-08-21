# Formato físico do Gate Externo

## Separação de contratos

Há dois documentos diferentes:

- **pack plan**: entrada descartável produzida por um conversor específico do modelo;
- **physical model map**: contrato runtime imutável usado pelo gate.

O gate externo não interpreta tensor names, hidden size ou função dos experts. O backend sabe como usar os bytes; o mapa diz somente onde eles estão.

## Pack plan v1

Schema: `mmb-pack-plan-v1`.

```json
{
  "schema_version": "mmb-pack-plan-v1",
  "model": {
    "id": "meu-moe-q4",
    "architecture": "minha-arquitetura-moe",
    "parameter_count": "2T",
    "quantization": {"name": "q4", "bits_per_weight": 4},
    "backend_contract": "meu-backend-v1",
    "map_revision": "weights-2026-08-20"
  },
  "alignment": 4096,
  "shard_size": "64GiB",
  "blocks": [
    {"id": "core.embeddings", "kind": "core", "source": "converted/core.embeddings.bin"},
    {
      "id": "layer.0.expert.0", "kind": "expert",
      "source": "converted/layer.0.expert.0.bin", "layer": 0, "expert": 0
    }
  ]
}
```

Regras:

- fontes são relativas ao diretório do plan e não podem escapar dele;
- cada fonte contém um bloco completo no layout esperado pelo backend;
- IDs e pares `(layer, expert)` são únicos;
- core não possui `layer`/`expert`;
- arquivos vazios são rejeitados;
- output existente nunca é sobrescrito;
- escrita usa diretório parcial e promoção ao fim;
- cópia e SHA-256 são streaming.

Comando:

```powershell
python mmb.py pack --plan pack-plan.json --output model
```

## Adaptador GGUF GraniteMoE

O adaptador direto lê o diretório GGUF sem dependências externas e aceita somente o layout explicitamente suportado `granitemoe`, com tensores `blk.N.ffn_{down,gate,up}_exps.weight` e eixo final de especialistas.

```powershell
python mmb.py gguf-inspect --gguf model.gguf
python mmb.py gguf-pack-moe --gguf model.gguf --output model-mmb
```

O conversor:

- valida arquitetura, número de camadas, especialistas totais e `top-k` treinado;
- exige os três tensores MoE em todas as camadas;
- fatia bytes GGML codificados sem desquantizar;
- cria um bloco por `(layer, expert)` na ordem `down`, `gate`, `up`;
- cria blocos core individuais e fixos;
- calcula SHA-256 do GGUF, do mapa e de cada bloco por streaming;
- produz `model.mmb-layout.json` com o contrato tensorial reduzido;
- recusa tipos, nomes ou formas que não possa provar fisicamente.

O contrato produzido é `mmb-raw-ggml-expert-concat-v1`. Outros layouts MoE exigem adaptadores explícitos; o gate continua sem interpretar nomes de tensor.

## Physical model map v1

Schema: `mmb-physical-model-map-v1`.

```json
{
  "schema_version": "mmb-physical-model-map-v1",
  "model": {
    "id": "meu-moe-q4",
    "architecture": "minha-arquitetura-moe",
    "parameter_count": 2000000000000,
    "quantization": {"name": "q4", "bits_per_weight": 4.0},
    "backend_contract": "meu-backend-v1",
    "map_revision": "weights-2026-08-20"
  },
  "blocks": [
    {
      "id": "layer.0.expert.0", "kind": "expert",
      "shard": "shards/model-00000.mmbw", "offset": 4096, "length": 536870912,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "alignment": 4096, "layer": 0, "expert": 0
    }
  ]
}
```

Validações runtime:

- versão exata e ausência de campos desconhecidos;
- paths confinados e shard existente;
- range dentro do arquivo e offset alinhado;
- SHA-256 sintaticamente válido;
- ranges não sobrepostos;
- IDs e rotas únicos.

## Configuração externa v1

Use `mmb.example.json` como base. Exatamente um modo de orçamento deve ser materializado.

Por RAM:

```json
"memory": {
  "ram_budget": "12GiB", "resident_experts": null,
  "kv_cache": "2GiB", "scratch": "1GiB",
  "transport": "shared_memory", "lease_timeout_seconds": 120
}
```

Por slots residentes:

```json
"memory": {
  "ram_budget": null, "resident_experts": 4,
  "kv_cache": "2GiB", "scratch": "1GiB",
  "transport": "shared_memory", "lease_timeout_seconds": 120
}
```

No modo por slots, o runtime calcula RAM usando o maior expert do mapa.

Memória estrutural opcional:

```json
"model_memory": {
  "enabled": true,
  "path": "state/model-memory.sqlite3"
}
```

O banco fica vinculado à identidade e à revisão exata do mapa. O caminho é relativo ao arquivo de configuração e não pode escapar desse diretório.

## IPC v1

Schema envelope: `mmb-external-gate-ipc-v1`. O transporte de controle é uma mensagem JSON por conexão TCP, terminada por newline.

Prefetch:

```json
{
  "protocol": "mmb-external-gate-ipc-v1", "op": "prefetch", "api_token": null,
  "request_id": "seq-1-token-2", "map_revision": "weights-2026-08-20",
  "items": [{"block_id": "layer.8.expert.12", "priority": 10}]
}
```

Acquire obrigatório:

```json
{
  "protocol": "mmb-external-gate-ipc-v1", "op": "acquire", "api_token": null,
  "request_id": "seq-1-token-2-layer-8", "map_revision": "weights-2026-08-20",
  "block_ids": ["layer.8.expert.12"]
}
```

Resposta inclui `lease_id` e, para cada bloco, `name`, `offset` e `length` de memória compartilhada. O cliente mapeia o handle e chama `release` depois que o kernel termina.
