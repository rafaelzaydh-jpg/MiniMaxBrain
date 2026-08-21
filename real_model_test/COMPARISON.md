# Comparação real: sem ajuda, com Porteiro e modelo integral

Data: 2026-08-20.

## Escopo honesto

Existem duas comparações diferentes:

1. **inferência normal sem Porteiro**: o executor mantém/mapeia o modelo GGUF integral e realmente gera tokens;
2. **paginação física MiniMaxBrain**: o Porteiro entrega os pesos reais selecionados, mas o backend tensorial ainda não calcula tokens sobre eles.

Portanto, os números de paginação abaixo não são chamados de tokens/s de inferência. Eles medem o trajeto físico completo de um token sintético — 24 camadas × 8 especialistas reais — e mostram quanto o Porteiro ajuda ou atrapalha a fase de I/O.

## Linha de base: modelo integral, sem Porteiro

O `llama-bench` executou o GraniteMoE GGUF integral em CPU, três repetições de geração de 32 tokens:

| Métrica | Resultado |
|---|---:|
| Geração real | **25,00 ± 7,43 tokens/s** |
| Pico de working set observado | **1.024,12 MiB** |
| Pico de memória privada | 766,78 MiB |
| Modelo codificado | 782,12 MiB |

Como o modelo cabe confortavelmente nesta máquina, este é o caminho correto para velocidade.

## A/B físico com limite de 16 especialistas

Configuração total: 172.153.304 bytes, dos quais 105.044.440 bytes são a capacidade de pesos. Cada execução usa a mesma sequência determinística de 2 tokens, 24 camadas e top-8. Foram feitas sete rodadas por modo, em ordem alternada.

| Métrica | Sob demanda, sem ajuda | Porteiro com próximo layer exato | Efeito |
|---|---:|---:|---:|
| Tempo mediano da rota de 2 tokens | 1,960 s | **1,466 s** | **−25,20%** |
| Rotas completas de token/s | 1,020 | **1,364** | **1,337×** |
| Acquire médio mediano por camada | 40,358 ms | **29,229 ms** | −27,58% |
| Bytes lidos medianos | 454.416.856 | 601.909.720 | **+32,46%** |
| Expulsões medianas | 368 | 526 | +42,93% |
| Prefetches descartados | 0 | 0 | igual |

O conselho perfeito reduziu latência, mas comprou essa redução com mais leituras e churn. Com somente 16 slots, pesos especulativos próximos competem com pesos recém-usados.

Uma medição isolada do processo de paginação sob demanda mostrou:

| Métrica | Resultado |
|---|---:|
| Pico de working set | **127,15 MiB** |
| Pico de memória privada | 123,60 MiB |
| Redução frente ao processo integral | **87,58%** |
| Razão de working set | **8,05× menor** |

Essa medição aloca os pesos do gate, mas não materializa as reservas configuradas de 32 MiB de KV e 32 MiB de scratch, pois ainda não há backend consumidor. Não representa a RAM final de inferência integrada.

## A/B físico com teto de 192 MiB

Foram feitas cinco rodadas alternadas por modo.

| Métrica | Sob demanda, sem ajuda | Porteiro com próximo layer exato | Efeito |
|---|---:|---:|---:|
| Tempo mediano da rota de 2 tokens | 1,915 s | **1,256 s** | **−34,43%** |
| Rotas completas de token/s | 1,044 | **1,592** | **1,525×** |
| Acquire médio mediano por camada | 39,335 ms | **25,042 ms** | −36,34% |
| Bytes lidos medianos | 454.416.856 | 568.510.936 | **+25,11%** |
| Expulsões medianas | 338 | 461 | +36,39% |
| Prefetches descartados | 0 | 0 | igual |

Mais RAM melhorou o ganho de latência e reduziu a amplificação de I/O, mas não a eliminou.

## Conclusão

Para este modelo de 782 MiB:

- **sem Porteiro vence em velocidade**: 25 tokens/s reais, pois o modelo cabe inteiro;
- **com Porteiro vence em memória física**: o harness de pesos caiu de aproximadamente 1.024 para 127 MiB;
- o melhor prefetch testado acelerou a paginação em 34,43%, mas essa paginação ainda é muito mais lenta que manter o pequeno modelo residente;
- o Porteiro deve possuir um modo `full-resident/bypass` quando o modelo cabe no orçamento;
- para modelos que não cabem, o Porteiro deixa de ser uma aceleração garantida e passa primeiro a ser um mecanismo de **viabilidade**;
- o futuro Porteiro Interno precisa considerar simultaneamente deadline, custo em bytes, chance de uso e espaço livre. Prever apenas “qual é o próximo expert” pode aumentar I/O demais.

Não é correto concluir destes números que um modelo de 1 TB já rodará bem em 12 GB. Está provado que a memória pode ser limitada e que o prefetch reduz parte da espera. Ainda falta provar que a localidade de um modelo grande é suficiente para não tornar SSD → RAM o gargalo dominante.

## Reproduzir

```powershell
python -B tools\mmb_ab_compare.py `
  --config real_model_test\mmb-granite-pageable\gate.experts.json `
  --tokens 2 --rounds 7

python -B tools\mmb_ab_compare.py `
  --config real_model_test\mmb-granite-pageable\gate.ram.json `
  --tokens 2 --rounds 5

real_model_test\runner\llama-bench.exe `
  -m real_model_test\downloads\granite-3.0-1b-a400m-instruct-Q4_K_M.gguf `
  -ngl 0 -p 0 -n 32 -r 3
```

O cache de arquivos do Windows não foi limpo entre rodadas. A ordem A/B foi alternada para reduzir viés, mas os valores devem ser entendidos como resultados desta máquina e deste estado do sistema.
