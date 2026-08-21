# MiniMaxBrain — Web UI Redesign

A interface web foi redesenhada em `minimaxbrain/server_http.py` sem alterar o contrato HTTP existente.

## Direção visual
- Produto/utility: conversa permanece como centro de gravidade.
- Superfícies em azul-marinho/perolado, evitando o cinza neutro dominante.
- Accent azul com teal reservado para estado saudável.
- Material translúcido somente na barra funcional, composer e painel transitório de runtime.
- Resposta do assistente sem “card” pesado; mensagem do usuário recebe superfície discreta.
- Tipografia via system stack, hierarquia por tamanho/peso/espaço e não apenas por cor.
- Light/dark adaptativos com escolha persistida no navegador.

## Interação
- Streaming SSE preservado.
- Enter envia; Shift+Enter cria nova linha.
- Botão de envio vira ação de parar durante a geração.
- Novo chat limpa apenas o estado local da conversa.
- Respostas podem ser copiadas.
- Runtime técnico fica oculto até ser solicitado.
- Autenticação bearer existente foi preservada.

## Acessibilidade e responsividade
- Estados de foco visíveis.
- `aria-label`, `aria-live` e semântica de navegação/chat.
- Layout móvel sem encolher alvos de toque.
- Respeita `prefers-reduced-motion` e `prefers-reduced-transparency`.

## Validação
- `python -m py_compile minimaxbrain/server_http.py`
- `node --check` no JavaScript extraído
- `pytest -q tests/test_runtime.py` → 5 passed
