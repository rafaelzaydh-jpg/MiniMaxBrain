# Testes reais do MiniMaxBrain 0.3

Os testes abaixo devem ser executados na máquina que possui o GGUF real e o bundle `.mmbw` correspondente.

## 1. Build e testes locais

### Windows

Pré-requisitos:

- Python 3.11+;
- CMake 3.20+ no `PATH`;
- Visual Studio/Build Tools com ferramentas **MSVC C++ x64/x86**;
- espaço em disco para a build e para o bundle MMBW.

Use um Developer Command Prompt/PowerShell da instalação do Visual Studio quando `cl.exe` não estiver no `PATH`. O `starter.bat` também pode ativar o ambiente configurado para a máquina.


```powershell
python -m pytest -q
python tools/build_native.py --clean
```

Aceite:

- testes Python passam;
- o source pinado de `llama.cpp` compila junto com o MMB;
- `mmb_backend` compila;
- `mmb_ggml_tests` passa;
- `mmb_native_tests` passa.

O build usa a revisão registrada em `native/LLAMA_VERSION`.

## 2. Validar o bundle

```powershell
python mmb.py check ^
  --config conversor/SEU-MODELO-mmbw/gate.json
```

Aceite:

- `ok: true`;
- mapa e layout válidos;
- metadata válida;
- budget válido;
- selo válido quando `integrity=seal`.

## 3. Testar o pager nativo isoladamente

```powershell
python mmb.py smoke ^
  --config conversor/SEU-MODELO-mmbw/gate.json ^
  --blocks 4
```

Verifique:

- leitura de experts sem erro;
- segmentos `down/gate/up` com tamanho maior que zero;
- misses na primeira leitura;
- `resident_bytes` dentro do budget.

Esse teste prova armazenamento/cache. Ele não prova inferência.

## 4. Prova matemática do hook GGML

O build nativo executa `mmb_ggml_tests`.

Nesse teste, o tensor original usado por `GGML_OP_MUL_MAT_ID` é zerado de propósito. O resultado só pode bater com o baseline se o kernel consumir a matriz de expert entregue pelo pager a partir do `.mmbw`.

Aceite:

```text
ctest: 100% passed
```

Esse é o teste isolado do contrato:

```text
router expert IDs
→ provider MMB
→ MMBKernelLease
→ .mmbw
→ GGML_OP_MUL_MAT_ID
→ release
```

## 5. Aceite real GGUF x MMBW no mesmo processo

O runner nativo executa duas gerações greedy na mesma build pinada do
`llama.cpp`:

```text
A: GGUF original (baseline)
B: model.mmb-meta.gguf + core .mmbw + experts .mmbw
```

No caminho B, o GGUF original **não é fonte de dados dos tensors**. Os
tensors routed `ffn_*_exps` recebem apenas um endereço virtual placeholder
sem páginas físicas comprometidas; o `GGML_OP_MUL_MAT_ID` precisa obter os
bytes pelo provider MMB ou falhar.

Para Qwen3.6-35B-A3B, prefira o fluxo específico da seção 5.1.

Fluxo genérico:

```powershell
python tools/native_moe_acceptance.py ^
  --build ^
  --gguf conversor\SEU-MODELO.gguf ^
  --bundle conversor\SEU-MODELO-mmbw ^
  --tokens 32 ^
  --cache-bytes 4294967296
```

Aceite obrigatório:

```text
mmb_source=metadata+core_mmbw+expert_mmbw
token_parity=true
paged_kernel_used=true
paged_kernel_invocations > 0
router_requests > 0
bytes_read > 0
baseline_tokens == mmb_tokens
```

Se `paged_kernel_used=false`, o modelo não entrou no caminho MMB esperado.
Se houver erro de tamanho/shape/type, não relaxe o kernel: revise o layout
produzido pelo conversor.

### 5.1 Qwen3.6-35B-A3B

Use o **GGUF base**, não o arquivo MTP separado. O preflight específico exige:

```text
architecture=qwen35moe
layers=40
routed experts=256
active experts=8
120 routed tensors (40 x down/gate/up)
```

Primeira execução, convertendo e compilando:

```powershell
python tools\qwen36_acceptance.py ^
  --gguf "D:\modelos\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --bundle "D:\modelos\Qwen3.6-35B-A3B-Q4_K_M-mmbw" ^
  --convert ^
  --build ^
  --clean ^
  --cache-gib 1 ^
  --tokens 16 ^
  --ctx 1024
```

O conversor cria outra cópia física dos pesos no formato MMBW. Mantenha
espaço livre pelo menos igual ao tamanho do GGUF mais uma margem de 2 GiB.

Nas execuções seguintes, reutilize bundle e build:

```powershell
python tools\qwen36_acceptance.py ^
  --gguf "D:\modelos\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --bundle "D:\modelos\Qwen3.6-35B-A3B-Q4_K_M-mmbw" ^
  --cache-gib 1 ^
  --tokens 32 ^
  --ctx 2048
```

Não use `--no-verify` no primeiro aceite real.

## 6. Runtime direto MMB (sem GGUF)

Depois que o aceite A/B passou, o caminho normal do produto deve ser testado
sem `--gguf` e sem `llama-server`.

Para um bundle já convertido, não reconverta. Prepare apenas `gate.json`:

```powershell
python mmb.py prepare ^
  --bundle "D:\modelos\SEU-MODELO-mmbw" ^
  --cache-gib 1
```

Teste automático do runtime direto:

```powershell
python tools\direct_runtime_test.py ^
  --bundle "D:\modelos\SEU-MODELO-mmbw" ^
  --cache-gib 1 ^
  --tokens 16 ^
  --ctx 1024
```

Aceite obrigatório:

```text
inference_mode=paged_mmb
paged_kernel_used=true
router_requests > 0
bytes_read > 0
DIRECT_RUNTIME_OK
```

O teste não abre o GGUF original. Se ele passar, o caminho exercitado é:

```text
Python
→ mmb_backend.dll
→ MMBLlamaRuntime
→ metadata + core MMBW
→ router real
→ MMBPager
→ expert MMBW
→ GGML MUL_MAT_ID
```

### 6.1 Chat terminal

```powershell
python mmb.py chat ^
  --bundle "D:\modelos\SEU-MODELO-mmbw" ^
  --cache-gib 1 ^
  --ctx 2048 ^
  --tokens 128
```

### 6.2 Web/API

```powershell
python mmb.py web ^
  --bundle "D:\modelos\SEU-MODELO-mmbw" ^
  --ctx 2048 ^
  --port 8080 ^
  --open-browser
```

O GGUF original não deve ser necessário para esses comandos.

## 7. Fail-closed

Execute o chat com um bundle ausente ou com uma biblioteca nativa sem
`MMB_CAP_NATIVE_RUNTIME`.

Aceite:

```text
BACKEND_UNAVAILABLE
```

Nenhum token deve ser inventado e não existe fallback silencioso para o GGUF.

## 8. Roteamento e origem dos pesos

Durante o aceite nativo, a igualdade de tokens é necessária, mas não suficiente.

O backend precisa manter:

```text
router IDs reais
→ MMBPager.acquire()
→ segmentos do expert no .mmbw
→ ponteiros válidos durante todo o kernel
→ release após o compute
```

O pager pode deduplicar experts apenas para residência/cache. Ele não pode alterar os IDs usados semanticamente pelo kernel.

Aceite desta fase:

```text
router_requests > 0
paged_kernel_invocations > 0
MMBW bytes_read > 0
```

## 9. Gate de origem dos pesos

O loader MMB atual já remove o backing físico dos routed experts no caminho B.

O aceite deve provar simultaneamente:

```text
mmb_source=metadata+core_mmbw+expert_mmbw
MMBW bytes_read > 0
paged_kernel_invocations > 0
token_parity=true
```

A baseline ainda abre o GGUF original apenas para produzir os tokens de
referência da execução A. Ela é um modelo separado e é liberada antes da
validação final.

Após o aceite real de produção, o build completo anuncia:

```text
MMB_CAP_PAGER
MMB_CAP_PAGED_MOE_KERNEL
MMB_CAP_NATIVE_RUNTIME
```

O build `--no-llama` continua anunciando somente `MMB_CAP_PAGER`.

## 10. Estabilidade de RAM

Depois que o primeiro aceite Qwen3.6 passar, gerar ao menos 256 tokens e registrar:

- RSS do processo;
- `resident_bytes`;
- `peak_resident_bytes`;
- hits;
- misses;
- evictions;
- bytes lidos.

Aceite:

- expert cache respeita o budget;
- leases não vazam;
- RSS estabiliza;
- geração greedy continua idêntica ao baseline.
