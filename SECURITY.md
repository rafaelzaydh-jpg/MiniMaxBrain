# Segurança do MiniMaxBrain

## Relato responsável

Relate vulnerabilidades ao proprietário do repositório por canal privado. Inclua versão, plataforma, configuração relevante, impacto e reprodução mínima.

Não envie pesos de modelos, tokens de API ou dados privados em relatórios públicos.

## Fronteiras atuais

- Web/API usa loopback por padrão.
- Bind fora de `127.0.0.1`, `localhost` ou `::1` exige `server.api_token`.
- Endpoints protegidos exigem `Authorization: Bearer <token>`.
- O corpo HTTP possui limite configurável.
- Caminhos do mapa/shards não podem escapar do diretório do bundle.
- Ranges inválidos ou sobrepostos são rejeitados.
- Políticas `integrity=always` e `first_load` verificam SHA-256 por bloco.
- `integrity=seal` verifica o conjunto registrado no startup.
- O runtime direto exige `MMB_CAP_PAGED_MOE_KERNEL` e `MMB_CAP_NATIVE_RUNTIME`.
- Expert placeholders não são tratados como payload válido.
- Não existe fallback silencioso para GGUF, `llama-server` ou geração sintética.

## Integridade do bundle

O runtime valida contratos de mapa/layout, shapes, tipos e comprimentos codificados antes de entregar expert bytes ao kernel.

Leases impedem eviction enquanto um kernel ainda usa os ponteiros de um expert.

O seal atual detecta corrupção/modificação, mas não autentica o autor/origem do bundle. Assinaturas digitais de distribuição são uma extensão separada.

## Limitações

- O provider GGML é process-global; somente um runtime MMB deve ficar ativo por processo nesta versão.
- O backend validado atualmente é o caminho CPU da build testada; novos backends exigem aceite próprio.
- MiniMaxBrain não substitui controle de acesso do sistema operacional, criptografia de disco, sandbox ou verificação da licença dos modelos.
