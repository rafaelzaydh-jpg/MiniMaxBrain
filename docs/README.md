# Documentação do MiniMaxBrain

Comece pelo [`README.md`](../README.md), que descreve o estado atual, os limites e a comparação real.

## Contratos atuais

- [`external-gate-architecture.md`](external-gate-architecture.md) — divisão entre gate externo, executor e futuro gate interno;
- [`model-format.md`](model-format.md) — pack plan, mapa físico, configuração e IPC;
- [`memory-kernel.md`](memory-kernel.md) — grafo estrutural persistente e perfil de rotas.

## Evidência experimental

- [`../real_model_test/README.md`](../real_model_test/README.md) — execução, conversão, paginação e IPC com modelo real;
- [`../real_model_test/COMPARISON.md`](../real_model_test/COMPARISON.md) — A/B sem prefetch, com ajuda exata e modelo integral;
- [`../real_model_test/REPORT.md`](../real_model_test/REPORT.md) — relatório técnico da validação.

Documentos descrevem somente funcionalidades presentes no repositório. Funcionalidades futuras são marcadas explicitamente como não implementadas.
