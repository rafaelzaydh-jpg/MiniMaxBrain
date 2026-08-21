# MiniMaxBrain — Web UI 0.3

A interface Web é uma camada fina sobre o runtime local. Ela não contém lógica de inferência e não altera o contrato do pager, cache, leases ou GGML.

## Direção final

A tela principal é deliberadamente simples:

- conversa como conteúdo principal;
- cabeçalho compacto com MiniMaxBrain, modelo e estado textual;
- sem indicadores luminosos repetidos, gradientes ou decoração contínua;
- velocidade de geração (`tok/s`) sempre visível;
- métricas técnicas em `Detalhes`, fechadas por padrão;
- mensagem do assistente sem card pesado;
- mensagem do usuário com superfície discreta;
- system font stack, light/dark mode e layout responsivo.

## Regra estrutural de scroll

A janela inteira não rola.

```text
100dvh
├── header       fixo no layout
├── chat-region  minmax(0, 1fr)
│   └── chat-scroll  ← único scroll vertical do histórico
└── composer     fixo no layout
```

`html`, `body` e `.app` têm altura limitada ao viewport e `overflow:hidden`. O conteúdo das mensagens cresce dentro de `#chatScroll`, não aumenta a altura da página.

O textarea cresce até 144 px; depois passa a ter scroll próprio.

## Auto-follow durante streaming

O histórico acompanha a resposta somente enquanto o usuário permanece próximo do final.

```text
perto do fim → pinnedToBottom=true → novos chunks acompanham
usuário sobe → pinnedToBottom=false → streaming não muda a posição
                                     → mostra “↓ Mais recente”
```

Uma nova mensagem enviada pelo usuário volta explicitamente ao final.

Isso permite ler conteúdo antigo enquanto o modelo continua gerando.

## Streaming SSE

O cliente processa eventos SSE por delimitador de evento (`\n\n` / CRLF equivalente), não por chunk de rede.

Um `reader.read()` pode conter:

- parte de um evento;
- um evento inteiro;
- vários eventos juntos.

O buffer só executa `JSON.parse()` depois de encontrar um evento SSE completo. `[DONE]` encerra o stream.

## Tokens por segundo

`MMBRuntime.stream_chat()` publica:

```text
tokens_generated
tokens_per_second
ttft_ms
generation_elapsed_ms
decode_elapsed_ms
```

`tokens_per_second` exclui TTFT do denominador:

```text
(tokens_generated - 1) / decode_elapsed
```

A UI mostra `≈ x.xx tok/s` porque a medição ocorre sobre callbacks emitidos pelo runtime.

## Acessibilidade

- foco visível;
- labels semânticos;
- status textual, não apenas por cor;
- `aria-live` para throughput;
- controles confortáveis;
- scrollbar visível;
- teclado: Enter envia, Shift+Enter quebra linha;
- `prefers-reduced-motion`;
- layout móvel.

## Referências de comportamento

A implementação foi revisada contra padrões de chats open source, sem copiar branding ou assets:

- Open WebUI — canvas de conversa e estado de auto-scroll:
  https://github.com/open-webui/open-webui
- NextChat — estado explícito de auto-scroll e scroll-to-bottom:
  https://github.com/ChatGPTNextWeb/NextChat
- `use-stick-to-bottom` — semântica de “grudar no fim” para chats com streaming:
  https://github.com/stackblitz-labs/use-stick-to-bottom

Referência primária de design:
https://developer.apple.com/design/human-interface-guidelines/
