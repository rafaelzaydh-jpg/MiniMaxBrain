# MiniMaxBrain 0.3

MiniMaxBrain é um runtime experimental para modelos **Mixture-of-Experts (MoE)** que mantém o banco lógico de experts em `.mmbw` e traz para RAM apenas o working set selecionado pelo router real do modelo.

A versão 0.3 executa o caminho paginado diretamente sobre a revisão pinada do `llama.cpp`/GGML. O GGUF original é necessário para conversão e para testes A/B de correção, mas **não é necessário para o chat normal depois que o bundle MMB foi criado**.

```text
Chat / Web / API
      ↓
MMBRuntime (Python)
      ↓
mmb_backend.dll
      ↓
MMBLlamaRuntime
      ↓
metadata + core MMBW
      ↓
router real do llama.cpp
      ↓
GGML_OP_MUL_MAT_ID
      ↓
MMBPager
      ↓
experts do .mmbw
      ↓
kernel GGML
```

## Estado atual

| Capacidade | Estado |
|---|---|
| Conversão GGUF MoE → MMBW | funcional |
| Metadata-only GGUF | funcional |
| Core carregado de MMBW | funcional |
| Expert weights sem backing físico completo | funcional |
| Router real → expert IDs | funcional |
| `MUL_MAT_ID` → MMBPager | funcional |
| Cache LRU + leases | funcional |
| SHA-256 por bloco | funcional |
| Runtime nativo persistente | funcional |
| Chat terminal direto MMB | funcional |
| Web/API direto MMB | funcional |
| GGUF necessário depois da conversão | não |
| `llama-server` necessário | não |

O aceite real já demonstrou em `qwen35moe`:

```text
token_parity=true
paged_kernel_used=true
router_requests>0
bytes_read>0
mmb_source=metadata+core_mmbw+expert_mmbw
```

Consulte [`REAL_TESTS.md`](REAL_TESTS.md) para o protocolo completo.

## Uso rápido no Windows

Esta distribuição mantém também os arquivos de desenvolvimento. Na primeira máquina é necessário compilar o backend nativo uma vez.

Requisitos de desenvolvimento:

- Python 3.11+;
- CMake 3.20+;
- MSVC C++ x64/x86 (Visual Studio Build Tools/Visual Studio com C++);
- Windows x64 para o fluxo `.bat` atual.

Abra:

```text
starter.bat
```

Fluxo recomendado para um bundle que já existe:

```text
[1] Construir/testar backend nativo
[3] Preparar bundle MMBW já convertido
[6] Chat direto MMB
```

Depois da build, o chat normal não recompila o modelo nem usa o GGUF original.

### Ambiente MSVC

Se `cl.exe`/`nmake.exe` não estiverem visíveis, use o Developer Command Prompt correspondente à sua instalação do Visual Studio ou execute `VsDevCmd.bat -arch=x64` antes da build.

O `starter.bat` da árvore atual também contém o caminho de ativação usado no ambiente Windows em que esta versão foi validada.

## Instalação Python

Para trabalhar diretamente pelo terminal:

```bat
python -m pip install -e .
```

O runtime Python usa apenas a standard library; `pytest` é dependência de desenvolvimento.

## Construir e testar o backend

```bat
python tools\build_native.py
```

Isso compila:

- o `llama.cpp` pinado;
- `mmb_core`;
- `mmb_backend`;
- testes C++.

No Windows, a DLL esperada é:

```text
native\build\Release\mmb_backend.dll
```

## Converter um GGUF MoE

Pelo assistente:

```text
conversor.bat
```

Ou pelo CLI:

```bat
python mmb.py convert ^
  --gguf "modelos\modelo.gguf" ^
  --output "modelos\modelo-mmbw" ^
  --cache-gib 1
```

A conversão copia os bytes GGML codificados. Ela não dequantiza nem re-quantiza os pesos.

O bundle principal contém:

```text
modelo-mmbw\
  gate.json
  model.mmb-map.json
  model.mmb-layout.json
  model.mmb-meta.gguf
  model-00000.mmbw
  model.verified.json   # quando seal está habilitado
```

## Usar um bundle já convertido

Não reconverta. Prepare apenas a configuração caso `gate.json` ainda não exista:

```bat
python mmb.py prepare ^
  --bundle "modelos\modelo-mmbw" ^
  --cache-gib 1
```

Esse comando não modifica os pesos `.mmbw`.

## Chat direto

Pela pasta do bundle:

```bat
python mmb.py chat ^
  --bundle "modelos\modelo-mmbw" ^
  --cache-gib 1 ^
  --ctx 2048 ^
  --tokens 128
```

Ou pela configuração:

```bat
python mmb.py chat ^
  --config "modelos\modelo-mmbw\gate.json" ^
  --ctx 2048 ^
  --tokens 128
```

## Web/API

```bat
python mmb.py web ^
  --bundle "modelos\modelo-mmbw" ^
  --ctx 2048 ^
  --port 8080 ^
  --open-browser
```

Endpoints atuais:

- `GET /health`
- `GET /api/stats`
- `GET /v1/models`
- `POST /v1/chat/completions`

O histórico completo é reconstruído pelo chat template armazenado no modelo.

## Validação rápida

Bundle:

```bat
python mmb.py check --config "modelos\modelo-mmbw\gate.json"
```

Pager isolado:

```bat
python mmb.py smoke --config "modelos\modelo-mmbw\gate.json" --blocks 4
```

Runtime direto sem abrir o GGUF:

```bat
python tools\direct_runtime_test.py ^
  --bundle "modelos\modelo-mmbw" ^
  --cache-gib 1 ^
  --ctx 1024 ^
  --tokens 16
```

Resultado esperado:

```text
inference_mode=paged_mmb
paged_kernel_used=true
router_requests>0
bytes_read>0
DIRECT_RUNTIME_OK
```

`smoke` testa armazenamento/cache. A prova de inferência paginada é o gate do runtime direto e, para correção A/B, o aceite descrito em `REAL_TESTS.md`.

## Desenvolvimento e testes

Os arquivos de desenvolvimento permanecem nesta distribuição de propósito para facilitar experimentação.

Python:

```bat
python -m pytest -q
```

Nativo:

```bat
python tools\build_native.py
```

Aceite real GGUF × MMB:

```bat
python tools\qwen36_acceptance.py ...
```

Arquitetura:

- [`docs/runtime-architecture.md`](docs/runtime-architecture.md)
- [`docs/model-format.md`](docs/model-format.md)

## ABI nativa

A interface pública fica em:

```text
native/include/mmb.h
```

Capabilities do build completo:

```text
MMB_CAP_PAGER
MMB_CAP_PAGED_MOE_KERNEL
MMB_CAP_NATIVE_RUNTIME
```

A ABI de runtime inclui:

```text
mmb_runtime_open
mmb_runtime_chat
mmb_runtime_get_stats
mmb_runtime_close
```

Python usa essa ABI estável e não espelha structs privadas do `llama.cpp`/GGML.

## Invariantes do runtime

1. o router neural determina os expert IDs;
2. os IDs reais chegam ao `MMBPager`;
3. expert blocks permanecem leased durante o kernel;
4. os expert weights vêm do `.mmbw`;
5. release ocorre somente após o compute;
6. `paged_experts_used` só muda depois de execução paged real;
7. não existe fallback silencioso para GGUF, `llama-server` ou geração sintética.

## Limitação atual

O provider GGML ainda é process-global. Por segurança, somente um `MMBLlamaRuntime` pode ficar ativo por processo nesta versão.

## Licença

Consulte [`LICENSE.md`](LICENSE.md) e [`COMMERCIAL.md`](COMMERCIAL.md).
