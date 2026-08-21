# Arquitetura do runtime MMB 0.3

O caminho suportado de inferência da versão 0.3 é o runtime nativo integrado ao `llama.cpp`/GGML.

```text
Chat / Web / API
      ↓
MMBRuntime (Python)
      ↓
mmb_backend
      ↓
MMBLlamaRuntime
      ↓
model.mmb-meta.gguf + core MMBW
      ↓
grafo llama.cpp
      ↓
router real
      ↓
GGML_OP_MUL_MAT_ID
      ↓
provider MMB
      ↓
MMBKernelLease
      ↓
MMBPager / MMBCache
      ↓
expert block .mmbw
      ↓
kernel GGML
```

## Componentes

### `MMBModel`

Abre e valida o bundle físico. Resolve:

- core tensors;
- `(layer, expert)`;
- shard, offset e comprimento;
- segmentos `down`, `gate` e `up`;
- tipos GGML e shapes.

### `MMBPager`

Responsável por:

- deduplicar IDs para residência física;
- carregar misses do `.mmbw`;
- verificar SHA-256 quando configurado;
- manter cache LRU dentro do budget;
- contabilizar hits, misses, bytes e latências;
- impedir eviction de blocos leased.

### `MMBKernelLease`

Mantém os expert blocks válidos durante todo o compute.

O fluxo é:

```text
router IDs
  -> provider.begin()
  -> pager.acquire()
  -> provider.get()
  -> GGML compute
  -> barreira dos workers
  -> commit_after_compute()
  -> provider.end()
  -> release
```

`paged_experts_used` e as invocações paged só são registrados após compute confirmado.

### `MMB_MOE_PLACEHOLDER`

Os routed expert weights existem logicamente no grafo, mas não carregam o banco completo de pesos em memória física.

O placeholder:

- preserva shape/stride/endereço lógico esperado pelo planner;
- não é host memory comum;
- não pode ser usado como fallback de payload;
- força o `MUL_MAT_ID` routed a obter os bytes através do provider MMB.

### `MMBLlamaRuntime`

Mantém o modelo/contexto llama persistente e expõe a ABI de chat:

```text
mmb_runtime_open
mmb_runtime_chat
mmb_runtime_get_stats
mmb_runtime_close
```

O GGUF original não é backing de runtime no caminho MMB.

## Invariantes

1. somente o router neural determina os IDs semanticamente usados;
2. deduplicação do pager não altera a lista lógica do kernel;
3. expert com lease ativo não pode ser evictado;
4. shape/type/encoded length precisam coincidir;
5. hash inválido aborta a leitura quando a verificação está ativa;
6. placeholder não pode fornecer silenciosamente os pesos;
7. falta do runtime nativo é erro explícito;
8. não existe fallback para `llama-server` ou geração sintética.

## Integração GGML

A integração é deliberadamente estreita. O provider só intercepta os routed weights:

```text
blk.*.ffn_down_exps.weight
blk.*.ffn_gate_exps.weight
blk.*.ffn_up_exps.weight
```

Outros `MUL_MAT_ID` continuam no caminho GGML normal.

## Capability

Um build completo anuncia:

```text
MMB_CAP_PAGER
MMB_CAP_PAGED_MOE_KERNEL
MMB_CAP_NATIVE_RUNTIME
```

Um build `--no-llama` anuncia somente a capability de pager.

## Implementação Python antiga

`ExternalGate`/cache Python ainda pode existir no source como código legado/de referência, mas **não é o executor neural suportado**. O caminho funcional do produto é o pager C++ conectado ao GGML.

Uma futura limpeza pode remover essa implementação de referência depois que todos os consumidores forem migrados.

## Limitação atual

O provider GGML é process-global. Nesta versão somente um `MMBLlamaRuntime` deve permanecer ativo por processo.
