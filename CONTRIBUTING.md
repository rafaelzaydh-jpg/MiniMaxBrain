# Contribuindo

MiniMaxBrain é atualmente desenvolvido e mantido por seu autor e não aceita pull requests externos neste momento. Issues com relatórios reproduzíveis continuam úteis quando o canal estiver habilitado.

Ao relatar um problema, inclua:

- sistema operacional e versão do Python;
- comando executado e mensagem completa;
- schema e `map_revision` do modelo, sem enviar pesos protegidos;
- orçamento de RAM, transporte e modo de integridade;
- passos mínimos de reprodução.

Para validar uma alteração privada permitida pela licença:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q minimaxbrain tests tools mmb.py
python mmb.py --help
```

Não inclua modelos, chaves, tokens ou bancos locais de `ModelMemory` em relatórios públicos.
