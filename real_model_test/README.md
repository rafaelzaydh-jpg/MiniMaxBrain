# MiniMaxBrain External Gate — comparação com modelo MoE real

Validação experimental do **Porteiro Externo da MiniMaxBrain** usando pesos reais do IBM Granite 3.0 1B-A400M Instruct.

Este diretório documenta três cenários distintos:

1. inferência convencional com o modelo integral, sem Porteiro;
2. paginação MiniMaxBrain somente sob demanda;
3. paginação MiniMaxBrain com prefetch exato da próxima camada, representando o melhor caso de ajuda do futuro Porteiro Interno.

> [!IMPORTANT]
> A inferência convencional realmente gera tokens. O caminho MiniMaxBrain atual mede paginação, integridade, cache, leases e entrega dos pesos reais, mas ainda não executa os kernels tensoriais sobre esses blocos. Por isso, os resultados de paginação não são apresentados como tokens/s de inferência.

## Resultado em uma frase

Neste modelo de 782,12 MiB, **sem Porteiro é mais rápido**, pois o modelo cabe inteiro na memória; **com Porteiro usa muito menos RAM**, e o prefetch reduz a espera de paginação em até 34,43%, mas aumenta o volume de I/O.

## Resumo comparativo

### Modelo integral versus paginação MiniMaxBrain

| Métrica | Modelo integral, sem Porteiro | Harness de paginação MiniMaxBrain |
|---|---:|---:|
| Geração real | **25,00 ± 7,43 tokens/s** | ainda não integrada |
| Pico de working set observado | **1.024,12 MiB** | **127,15 MiB** |
| Pico de memória privada | 766,78 MiB | 123,60 MiB |
| Redução observada de working set | — | **87,58%** |
| Razão de working set | — | **8,05× menor** |

O pico MiniMaxBrain acima mede o processo de paginação dos pesos. As reservas configuradas de KV e scratch não foram fisicamente alocadas porque o backend tensorial consumidor ainda não está conectado.

### Porteiro sem e com ajuda — 16 especialistas residentes

Sete rodadas alternadas por modo, sempre com a mesma sequência de dois tokens sintéticos, 24 camadas e oito especialistas reais por camada.

| Métrica | Somente sob demanda | Prefetch exato da próxima camada | Efeito |
|---|---:|---:|---:|
| Tempo mediano da rota física | 1,960 s | **1,466 s** | **−25,20%** |
| Rotas completas de token/s | 1,020 | **1,364** | **1,337×** |
| Acquire médio por camada | 40,358 ms | **29,229 ms** | −27,58% |
| Bytes lidos medianos | 454.416.856 | 601.909.720 | **+32,46%** |
| Expulsões medianas | 368 | 526 | +42,93% |
| Prefetches descartados | 0 | 0 | igual |

### Porteiro sem e com ajuda — teto de 192 MiB

Cinco rodadas alternadas por modo.

| Métrica | Somente sob demanda | Prefetch exato da próxima camada | Efeito |
|---|---:|---:|---:|
| Tempo mediano da rota física | 1,915 s | **1,256 s** | **−34,43%** |
| Rotas completas de token/s | 1,044 | **1,592** | **1,525×** |
| Acquire médio por camada | 39,335 ms | **25,042 ms** | −36,34% |
| Bytes lidos medianos | 454.416.856 | 568.510.936 | **+25,11%** |
| Expulsões medianas | 338 | 461 | +36,39% |
| Prefetches descartados | 0 | 0 | igual |

O prefetch melhorou a latência, mas trouxe amplificação de leitura. Esse resultado é importante: prever corretamente o próximo especialista não basta; o advisor precisa considerar também custo em bytes, deadline, confiança, espaço livre e risco de expulsar um peso útil.

## Modelo testado

| Propriedade | Valor observado |
|---|---:|
| Modelo | IBM Granite 3.0 1B-A400M Instruct |
| Arquitetura GGUF | `granitemoe` |
| Parâmetros contados | 1.334.628.352 |
| Camadas MoE | 24 |
| Especialistas por camada | 32 |
| Especialistas ativos por camada | 8 |
| Quantização | GGUF Q4_K_M / ftype 15 |
| GGUF | 821.845.024 bytes |
| Shard MiniMaxBrain | 820.109.312 bytes |

Fontes:

- [modelo oficial IBM Granite](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct);
- [GGUF Q4_K_M usado no teste](https://huggingface.co/mrutkows/granite-3.0-1b-a400m-instruct-GGUF);
- [releases oficiais do executor usado como referência](https://github.com/ggml-org/llama.cpp/releases/).

O executável `llama.cpp` foi usado somente como instrumento independente de referência para confirmar que o GGUF gera texto. Ele não definiu a arquitetura, o cache, o mapa físico, o protocolo ou as decisões do Porteiro MiniMaxBrain. Projetos citados na conversa original, incluindo Warp, foram deliberadamente excluídos da criação da arquitetura.

## O que o teste executa

```text
                       MESMOS PESOS GRANITEMOE
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
        modelo integral                     conversor MiniMaxBrain
        geração em CPU                768 blocos de especialistas
                 |                                 |
                 |                         mapa físico + hashes
                 |                                 |
                 |                     Porteiro Externo independente
                 |                      /                         \
                 |             acquire sob demanda       prefetch + acquire
                 |                      \                         /
                 |                       cache limitado por RAM
                 |                                 |
                 +---------- comparação -----------+
```

O conversor encontrou 72 tensores MoE fundidos — `down`, `gate` e `up` em cada uma das 24 camadas — e criou:

- 170 blocos core fixos, totalizando 88.725.976 bytes;
- 768 blocos de especialista, um para cada `(camada, especialista)`;
- 938 blocos no total;
- SHA-256 individual para cada bloco;
- um layout reduzido para o futuro backend tensorial.

Nenhum tensor é desquantizado durante a conversão. Cada bloco preserva os bytes GGML codificados e concatena os segmentos na ordem documentada `down`, `gate`, `up`.

## O que está comprovado

- O modelo GGUF real gera texto em CPU.
- A topologia MoE é identificada sem dependências Python externas.
- Tensores fundidos podem ser separados fisicamente por especialista.
- O Porteiro lê apenas os ranges solicitados.
- Orçamentos por quantidade de especialistas e por RAM são respeitados.
- Prefetch, cache, expulsão, lease e verificação SHA-256 funcionam sobre pesos reais.
- O serviço independente entrega os blocos por memória compartilhada a outro processo.
- O cliente externo recalculou e confirmou o SHA-256 de oito especialistas mapeados.

## O que ainda não está comprovado

- Geração de tokens usando diretamente os blocos entregues pelo Porteiro.
- Qualidade do texto após a integração tensorial.
- Tokens/s ponta a ponta com paginação ativa.
- Desempenho de um modelo de 1–2 TB em 12 GB de RAM.
- Localidade suficiente para impedir que SSD → RAM se torne o gargalo dominante.

Para completar a inferência integrada, o executor precisa:

1. obter os índices reais produzidos pelo router da camada;
2. enviar esses índices a `acquire_routes`;
3. mapear os blocos devolvidos;
4. executar `down`, `gate` e `up` sobre o payload;
5. chamar `release` após o kernel terminar.

O Porteiro pode permanecer em outro processo, mas não pode operar como caixa-preta observando apenas prompt e texto: a seleção matemática real dos especialistas precisa atravessar o contrato.

## Estrutura do diretório

```text
real_model_test/
  README.md                         esta documentação
  REPORT.md                         validação completa do modelo real
  COMPARISON.md                     metodologia e números detalhados do A/B
  .gitignore                        impede commit acidental dos binários grandes
  downloads/                        GGUF e ZIP do runner; não versionados
  runner/                           executáveis locais de referência; não versionados
  mmb-granite-pageable/
    model.mmb-map.json             mapa físico e hashes por bloco
    model.mmb-layout.json          layout tensorial reduzido
    gate.experts.json           limite por especialistas residentes
    gate.ram.json               teto por RAM
    gate.shared.json            serviço externo com memória compartilhada
    model-00000.mmbw                 shard gerado; não versionado
```

Código relacionado:

- [`../minimaxbrain/gguf.py`](../minimaxbrain/gguf.py): leitura estrita do diretório GGUF;
- [`../minimaxbrain/gguf_moe.py`](../minimaxbrain/gguf_moe.py): separação streaming dos especialistas;
- [`../tools/mmb_real_moe_smoke.py`](../tools/mmb_real_moe_smoke.py): benchmark físico;
- [`../tools/mmb_ab_compare.py`](../tools/mmb_ab_compare.py): comparação A/B alternada;
- [`../tools/mmb_real_ipc_smoke.py`](../tools/mmb_real_ipc_smoke.py): teste entre processos.

## Pré-requisitos

- Windows x64 para reproduzir exatamente os comandos desta execução;
- Python 3.12 ou compatível;
- espaço livre para o GGUF e para o shard convertido;
- um runner CPU compatível para a linha de base;
- o repositório MiniMaxBrain como diretório de trabalho.

O leitor GGUF, o conversor e o Porteiro não dependem de NumPy, PyTorch ou Transformers.

## Preparação dos artefatos

Crie os diretórios locais:

```powershell
New-Item -ItemType Directory -Force real_model_test\downloads | Out-Null
New-Item -ItemType Directory -Force real_model_test\runner | Out-Null
```

Baixe o arquivo `granite-3.0-1b-a400m-instruct-Q4_K_M.gguf` para `real_model_test\downloads` a partir do repositório GGUF indicado acima.

Baixe e extraia uma release CPU x64 do executor de referência para `real_model_test\runner`. A execução documentada utilizou o build `b10218-de699957b`.

Verifique o modelo usado neste relatório:

```powershell
Get-FileHash `
  real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf `
  -Algorithm SHA256
```

SHA-256 esperado:

```text
242b06a85482b30fe0b8bc7d70e614e7328d1275dc236648c0e4d71db0cbdbb3
```

## Inspeção e conversão

Inspecione a topologia física:

```powershell
python -B mmb.py gguf-inspect `
  --gguf real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf
```

Converta para blocos MiniMaxBrain:

```powershell
python -B mmb.py gguf-pack-moe `
  --gguf real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf `
  --output real_model_test\mmb-granite-pageable-novo
```

O conversor recusa sobrescrever um diretório existente. Use outro nome de saída ao repetir a operação.

## Validar as configurações

```powershell
python -B mmb.py check `
  --config real_model_test\mmb-granite-pageable\gate.experts.json

python -B mmb.py check `
  --config real_model_test\mmb-granite-pageable\gate.ram.json
```

## Reproduzir a comparação A/B

Limite de 16 especialistas:

```powershell
python -B tools\mmb_ab_compare.py `
  --config real_model_test\mmb-granite-pageable\gate.experts.json `
  --tokens 2 `
  --rounds 7
```

Teto de 192 MiB:

```powershell
python -B tools\mmb_ab_compare.py `
  --config real_model_test\mmb-granite-pageable\gate.ram.json `
  --tokens 2 `
  --rounds 5
```

Cada rodada alterna a ordem dos modos para reduzir viés. O benchmark compara exatamente os mesmos blocos e as mesmas rotas.

## Reproduzir a linha de base sem Porteiro

```powershell
real_model_test\runner\llama-bench.exe `
  -m real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf `
  -ngl 0 `
  -p 0 `
  -n 32 `
  -r 3
```

Geração curta:

```powershell
real_model_test\runner\llama-cli.exe `
  -m real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf `
  -ngl 0 `
  -c 512 `
  -n 32 `
  --temp 0 `
  --seed 42 `
  -p "Responda em portugues com uma frase curta: o teste real da MiniMaxBrain esta funcionando?"
```

## Executar como processo externo

Terminal 1:

```powershell
python -B mmb.py serve `
  --config real_model_test\mmb-granite-pageable\gate.shared.json
```

Terminal 2:

```powershell
python -B tools\mmb_real_ipc_smoke.py `
  --config real_model_test\mmb-granite-pageable\gate.shared.json
```

O segundo processo adquire oito especialistas da camada 0, mapeia os segmentos compartilhados, recalcula os hashes e libera o lease.

## Como interpretar os números

Para o caminho paginado, cada rota sintética representa os especialistas necessários ao longo das 24 camadas para um token. Ela mede:

- admissão obrigatória;
- espera por prefetch;
- leitura e cópia de ranges;
- verificação de integridade;
- residência e expulsão;
- criação e liberação de leases.

Ela não mede atenção, multiplicações tensoriais, sampling ou qualidade. Somar diretamente a taxa de rotas ao resultado do executor integral produziria uma estimativa enganosa.

O cache de arquivos do Windows não foi limpo entre as rodadas. A ordem foi alternada, mas os valores continuam específicos desta máquina e do estado do sistema em 2026-08-20.

## Decisões arquiteturais derivadas do teste

1. **Bypass quando couber:** se o modelo inteiro couber no orçamento, o modo integral deve ser preferido.
2. **Paginação como viabilidade:** quando o modelo não couber, o primeiro objetivo do Porteiro é tornar a execução possível.
3. **Prefetch não é gratuito:** a redução de latência precisa compensar bytes adicionais e churn.
4. **Conselho com custo:** o Porteiro Interno deve emitir prioridade, deadline, confiança e custo físico estimado.
5. **Seleção obrigatória separada:** erro de previsão nunca pode substituir o especialista escolhido pelo router real.
6. **Medição antes de promessa:** modelos de 1–2 TB exigem medir bytes frios por token e banda sustentável do SSD.

## Testes de regressão

- 22 testes focados do MiniMaxBrain aprovados após a limpeza do repositório;
- 14 testes do gate/mapa/IPC, 3 do GGUF e 5 da memória estrutural;
- o runtime usa somente a biblioteca padrão do Python; `pytest` é dependência de desenvolvimento.

## GitHub e arquivos grandes

O GitHub bloqueia arquivos acima de 100 MiB em repositórios Git comuns. O GGUF e o shard MiniMaxBrain excedem esse limite; por isso, o `.gitignore` deste diretório exclui:

- `downloads/`;
- `runner/`;
- `mmb-granite-pageable/*.mmbw`.

Versione o código, os mapas JSON, as configurações e os relatórios. Distribua modelos e shards por suas fontes originais, por releases apropriadas ou por Git LFS, respeitando limites e licenças. Consulte a [documentação oficial do GitHub sobre arquivos grandes](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) e [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## Documentação adicional

- [`REPORT.md`](REPORT.md): validação física completa e hashes;
- [`COMPARISON.md`](COMPARISON.md): metodologia detalhada e todas as ressalvas do A/B;
- [`../docs/external-gate-architecture.md`](../docs/external-gate-architecture.md): arquitetura do Porteiro Externo;
- [`../docs/model-format.md`](../docs/model-format.md): mapa físico, configuração e contratos.

## Licenças

O código MiniMaxBrain segue os termos de [`../LICENSE.md`](../LICENSE.md) e [`../COMMERCIAL.md`](../COMMERCIAL.md). O modelo IBM Granite, a quantização comunitária e o executor de referência mantêm suas próprias licenças e condições de distribuição. Os pesos e binários não são incorporados ao código MiniMaxBrain por este README.

---

**Conclusão:** neste teste, o Porteiro não supera um modelo pequeno que já cabe em RAM. Ele demonstra redução física de memória e controle de residência, e o prefetch demonstra ganho mensurável na fase de paginação. O próximo marco é conectar o router e os kernels reais para obter a primeira comparação ponta a ponta de tokens/s usando o Porteiro.
