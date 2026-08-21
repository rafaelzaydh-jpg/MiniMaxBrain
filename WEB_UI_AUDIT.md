# MiniMaxBrain 0.3 — Auditoria final Web e distribuição

## Resultado

**Status: aprovado para o source/release 0.3, sujeito ao teste final no Windows com o Qwen real.**

A auditoria preservou o runtime nativo e os contratos HTTP/GGML. Nenhuma alteração foi feita em pager, cache, leases, MMBW, bridge GGML ou llama.cpp.

## P0 — página “infinita”

### Sintoma

Conforme mensagens eram adicionadas, a página inteira aumentava de altura. Header/composer desciam junto com o documento.

### Causa

O shell usava `min-height:100dvh`. Em CSS Grid, o conteúdo podia aumentar a altura intrínseca do documento em vez de permanecer confinado ao `1fr`.

### Correção

```text
html/body: height:100%; overflow:hidden
.app:      height:100dvh; min-height:0; overflow:hidden
grid:      auto minmax(0,1fr) auto
chat:      único overflow-y:auto
```

Validação headless em viewport 1280×800 com 35 mensagens longas:

```text
window.innerHeight              = 800
document.documentElement.scrollHeight = 800
document.body.scrollHeight      = 800
chat clientHeight               = 639
chat scrollHeight               = 6364
composer bottom                 = 800
```

Ou seja: o documento permanece em 800 px e somente o histórico passa a rolar.

## P0 — auto-scroll durante streaming

O streaming não força mais o usuário ao final quando ele sobe para ler.

- distância do fim <= 140 px: auto-follow ligado;
- usuário se afasta: auto-follow desligado;
- aparece `↓ Mais recente`;
- clicar no controle restaura o acompanhamento;
- enviar uma nova mensagem força o posicionamento no fim.

## P0 — streaming SSE

O parser permanece orientado ao delimitador SSE completo (`\n\n` / CRLF). Ele suporta fragmentação e múltiplos eventos no mesmo chunk e evita concatenar JSONs.

## P1 — composer

O textarea:

- começa em 44 px;
- cresce até 144 px;
- depois usa scroll interno;
- não aumenta indefinidamente o shell da aplicação.

## P1 — scrollbar

A área de conversa usa o scroll nativo do browser, com thumb neutro e visível. Não existe outro scroll vertical no documento principal.

## P1 — distribuição para usuário

O backend funcional compilado foi promovido para:

```text
runtime/windows-x64/mmb_backend.dll
```

`minimaxbrain.native` procura esse artefato antes dos diretórios de build.

`starter.bat`:

- não executa mais `pip install -e .`;
- usa o backend precompilado;
- usa Python 3.11+ já instalado quando houver;
- se Python não existir, prepara CPython 3.11.9 embeddable oficial;
- se o backend não carregar por ausência do VC++ runtime, oferece instalar o redistribuível oficial Microsoft;
- mantém compilação em um submenu de desenvolvedor.

A build nativa deixa de ser requisito de uso.

## Segurança do bootstrap

Python portátil:

- origem: `python.org`;
- ZIP x64 oficial;
- SHA-256 verificado antes da extração;
- runtime Python permanece isolado e sem `pip/site-packages`.

VC++ runtime:

- origem: permalink oficial Microsoft;
- assinatura Authenticode é validada;
- só então o instalador é executado.

## Testes

```text
python compile: server_http.py PASS
python compile: runtime.py PASS
python compile: native.py PASS
JavaScript: node --check PASS
pytest: 32 passed, 1 skipped
layout headless: documento fixo / scroll interno PASS
```

O skip ocorre porque o ambiente Linux de auditoria não carrega a DLL Windows. O binário incluído é exatamente o backend compilado presente no ZIP funcional fornecido pelo projeto.

## Referências funcionais consultadas

- Apple HIG — Scroll views:
  https://developer.apple.com/design/human-interface-guidelines/scroll-views
- Apple HIG — Layout:
  https://developer.apple.com/design/human-interface-guidelines/layout
- Open WebUI:
  https://github.com/open-webui/open-webui
- NextChat:
  https://github.com/ChatGPTNextWeb/NextChat
- use-stick-to-bottom:
  https://github.com/stackblitz-labs/use-stick-to-bottom
