# Formato físico MMB 0.3

O conversor oficial aceita um GGUF MoE suportado e produz um bundle autocontido para execução paged:

```text
bundle/
  gate.json
  model.mmb-map.json
  model.mmb-layout.json
  model.mmb-meta.gguf
  model-00000.mmbw
  model.verified.json   # quando seal está habilitado
```

## `model.mmb-map.json`

É a autoridade dos ranges físicos. Cada bloco contém:

- `id`;
- `kind` (`core` ou `expert`);
- `shard`;
- `offset`;
- `length`;
- `alignment`;
- `sha256`.

Experts também possuem `layer` e `expert`.

Ranges não podem escapar do diretório do modelo, sobrepor outro bloco ou ultrapassar o shard.

## `model.mmb-layout.json`

O schema atual é `mmb-gguf-moe-layout-v2` e o contrato do backend é `mmb-raw-ggml-expert-segments-v2`.

O layout registra:

- arquitetura e identidade do GGUF de origem;
- topologia MoE (`layer_count`, `expert_count`, `active_experts_per_token`);
- core tensors;
- para cada layer, segmentos `down`, `gate` e `up`;
- `ggml_type` e shapes;
- shape fused de origem;
- eixo do expert;
- stride codificado por expert;
- metadata-only GGUF associado.

Cada bloco físico `(layer, expert)` contém:

```text
down | gate | up
```

Os segmentos precisam cobrir o bloco de forma contígua e manter o encoding GGML original.

## `model.mmb-meta.gguf`

Preserva metadados e descritores necessários para reconstruir o modelo no `llama.cpp`, sem duplicar o payload completo dos pesos.

Inclui, quando presentes na origem:

- arquitetura;
- tokenizer;
- special tokens;
- chat template;
- parâmetros do modelo;
- nomes, shapes e tipos dos tensors.

O layout registra tamanho e SHA-256 do metadata file.

## `.mmbw`

Os shards contêm os bytes GGML copiados sem dequantização ou re-quantização.

Core tensors são armazenados como blocos próprios.

Os routed expert weights são separados por `(layer, expert)` e organizados em segmentos `down/gate/up`.

## `gate.json`

A versão atual ainda usa o schema de configuração herdado `mmb-external-gate-config-v1` por compatibilidade.

No runtime direto ele fornece principalmente:

- caminho do mapa;
- budget de memória/cache;
- política de integridade;
- configuração HTTP.

O nome/schema será simplificado em uma versão futura; isso não altera o formato MMBW.

## Validação

`mmb check` valida de forma fail-closed:

- mapa físico;
- paths e ranges;
- topologia;
- cardinalidade de rotas;
- correspondência layout/mapa;
- cobertura de segmentos;
- core tensors;
- metadata GGUF e SHA-256;
- seal quando configurado.

O backend C++ repete as invariantes críticas ao abrir o modelo.

## Selo

`mmb seal` gera `model.verified.json`.

O selo detecta alteração de dados registrados no bundle. Ele fornece integridade; não é uma assinatura de procedência do modelo.

## Execução atual

O formato é consumido diretamente por:

```text
model.mmb-meta.gguf
      +
core MMBW
      ↓
MMBLlamaRuntime / llama.cpp
      ↓
router real
      ↓
GGML_OP_MUL_MAT_ID
      ↓
MMBPager
      ↓
expert MMBW
```

Os routed expert weights usam `MMB_MOE_PLACEHOLDER`, sem backing físico completo.

Quando o router seleciona experts, o provider MMB entrega ao kernel GGML os segmentos adquiridos do `.mmbw`.

Uma execução paged só é registrada depois do compute real; apenas carregar um expert para a cache não ativa `paged_experts_used`.

## GGUF original

Depois que o bundle está convertido, o GGUF original não é necessário para o chat direto.

Ele continua necessário para:

- gerar um bundle novo;
- repetir o aceite A/B contra a baseline original.
