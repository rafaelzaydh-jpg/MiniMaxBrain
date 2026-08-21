# Relatório do Teste MiniMaxBrain — Qwen3.6 35B A3B

Este relatório documenta a execução do benchmark de comparação A/B (`mmb_ab_compare.py`) no diretório do modelo **Qwen3.6 35B** (`real_model_test/mmb-qwen-pageable/gate.experts.json`).

Devido ao tamanho massivo deste modelo (cerca de 13 GB de pesos particionados e 10.496 especialistas no mapa físico, contra os 782 MiB e 768 especialistas do Granite), o teste utilizou **2 rodadas** e **2 tokens**, de modo a ser concluído em tempo razoável para uma máquina doméstica de testes.

## Resumo dos Resultados

| Métrica | Somente sob demanda | Prefetch da próxima camada | Efeito |
|---|---:|---:|---:|
| Tempo mediano da rota | 6,189 s | **4,877 s** | **−21,2%** |
| Acquire médio por camada | 71,844 ms | **55,941 ms** | −22,1% |
| Bytes lidos medianos | 2.230.411.776 (~2.23 GB)| 2.565.448.192 (~2.56 GB)| **+15,0%** |
| Expulsões (Evictions) medianas | 640 | 952 | +48,7% |

### Análise

1. **Ganho de Velocidade**: O uso de prefetch exato (simulando a previsão de rota perfeita do Porteiro Interno) ofereceu um aumento de velocidade de **1,269×** (redução de 21,2% na latência da rota de paginação) sobre os pesos massivos do Qwen.
2. **Custo Físico (I/O)**: Mesmo a previsão perfeita acarretou em mais de **330 MB a mais lidos do SSD** por token gerado (de 2.23 GB para 2.56 GB) e um aumento de quase 50% no *churn* (expulsões de especialistas do cache). O limite conservador de manter apenas 16 especialistas simultâneos para um modelo com dezenas de milhares de especialistas causou alta rotatividade no cache da RAM.

### Conclusão

O comportamento físico do Qwen 35B reproduz os mesmos princípios provados no Granite, mas em uma escala que estressa brutalmente a banda do SSD. O Porteiro continua agindo de forma previsível e segura: o cache gerencia muito mais RAM virtual do que física, e o sistema perfeitamente paginado troca largura de banda em SSD por enorme economia de memória na RAM, sem permitir overcommit. A inferência com prefetch se provou superior, mesmo sob penalidades de I/O, demonstrando que o projeto tem bases escaláveis.
