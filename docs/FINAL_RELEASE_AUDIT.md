# MiniMaxBrain 0.3 — Auditoria Final de Release

## Base
- Fonte: pacote final 0.3 fornecido pelo projeto.
- Arquitetura neural/pager/GGML preservada.
- Alterações desta finalização ficaram concentradas em Web UI, bootstrap/distribuição Windows e testes de contrato de release.

## UI final
- Shell limitado a `100dvh`; `html/body` não crescem com o histórico.
- Um único scroll vertical principal: histórico da conversa.
- Header e composer permanecem dentro da janela.
- Composer cresce até 144 px e depois usa scroll próprio.
- Streaming acompanha o fim somente enquanto o usuário está próximo do fim.
- Ao rolar para cima, auto-scroll é suspenso e aparece `↓ Mais recente`.
- Scrollbar permanece visível no Windows/Chrome.
- Métrica de tokens/s permanece visível; detalhes técnicos ficam sob disclosure.
- Parser SSE trabalha por eventos completos e tolera chunks parciais/múltiplos.

## Distribuição Windows
- Backend pré-compilado: `runtime/windows-x64/mmb_backend.dll`
- Tamanho: 3,741,696 bytes
- SHA-256: `e5685b77adf7093159e5745127e22319cbeaf7e7957ddfd8dfe40385409bae13`
- O loader prioriza o backend de release antes de diretórios de build.
- `native/build` não faz parte do release.
- `starter.bat` não executa `pip install` nem exige compilação no fluxo normal.
- Se Python >=3.11 não existir, o starter pode preparar o CPython 3.11.9 embeddable oficial.
- Se o runtime MSVC necessário não estiver presente, o starter oferece instalar o Visual C++ v14 x64 oficial.
- Ferramentas de compilação permanecem disponíveis somente no submenu de desenvolvedor.

## Gates executados neste ambiente
```text
32 passed
1 skipped
```

Skip esperado:
- `tests/test_native_pager.py`: backend Windows não pode ser carregado neste ambiente Linux.

Validações adicionais realizadas:
- compilação sintática dos módulos Python alterados;
- `node --check` no JavaScript embutido;
- teste headless de layout com histórico longo:
  - viewport/documento permaneceram em 800 px;
  - histórico interno cresceu para ~6364 px;
  - portanto o documento não virou página infinita.

## Limite da validação
O `mmb_backend.dll` pré-compilado foi promovido a partir da build funcional presente no pacote Windows fornecido. Ele não foi recompilado nem executado neste ambiente Linux. O gate final de release no Windows continua sendo:
1. extrair em uma máquina Windows;
2. abrir `starter.bat`;
3. iniciar Web Chat;
4. confirmar `ready / paged_mmb`;
5. gerar texto real e observar router/cache/tokens/s.

## Inventário
- Arquivos: 3391
- Diretórios: 380
- Tamanho extraído aproximado: 159.71 MiB
- Caches Python/pytest no release: 0
- Diretórios `native/build*` no release: 0
