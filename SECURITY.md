# Segurança do MiniMaxBrain

## Relato responsável

Relate vulnerabilidades ao proprietário do repositório por um canal privado. Não publique exploração funcional, tokens, hashes sensíveis ou pesos de terceiros antes de haver oportunidade razoável de correção.

Inclua versão/commit, plataforma, configuração relevante, impacto e reprodução mínima.

## Fronteiras atuais

- O serviço usa loopback por padrão.
- Bind fora de `127.0.0.1`, `localhost` ou `::1` exige um token de pelo menos 16 caracteres.
- O protocolo de controle é JSON-lines e possui limite configurável de requisição.
- O token é comparado em tempo constante.
- Paths de configuração, banco, shards e fontes são confinados aos diretórios dos respectivos contratos.
- Ranges fora do arquivo, sobrepostos ou desalinhados são rejeitados.
- Integridade pode ser `always`, `first_load` ou `none`; use `always` para material não confiável.
- O servidor não oferece escrita nos pesos nem desligamento remoto.

## Memória compartilhada

No modo `shared_memory`, clientes locais autorizados recebem o nome e o range do segmento. Trate usuários locais com acesso ao mesmo host como parte da fronteira de confiança. O MMB remove o segmento quando o bloco é expulso ou o gate é fechado, mas um processo já autorizado pode ter copiado os bytes.

## ModelMemory

O banco contém topologia física, caminhos, hashes e padrões de uso. Mantenha-o fora do Git e aplique as permissões adequadas do sistema operacional. O caminho configurado não pode escapar do diretório da configuração.

## Limitações

MiniMaxBrain ainda é um runtime Python experimental. Ele não substitui isolamento de processo, criptografia de disco, controle de acesso do host, sandbox do executor tensorial ou verificação de licença dos modelos utilizados.
