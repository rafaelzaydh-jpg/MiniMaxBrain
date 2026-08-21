# Contribuindo

MiniMaxBrain é atualmente desenvolvido e mantido por seu autor e não aceita pull requests externos neste momento. Relatórios reproduzíveis continuam úteis quando o canal estiver habilitado.

Ao relatar um problema, inclua:

- sistema operacional;
- versão do Python;
- versão/capabilities de `mmb_backend`;
- comando executado e mensagem completa;
- schema/map revision do bundle, sem enviar os pesos;
- cache/budget configurado;
- modo de integridade;
- passos mínimos de reprodução.

Para validar uma alteração local:

```bat
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools\build_native.py
python -m compileall -q minimaxbrain tests tools mmb.py
python mmb.py --help
```

Mudanças no loader, pager, placeholder ou hook GGML devem também passar pelo aceite descrito em `REAL_TESTS.md`.

Não inclua modelos, chaves, tokens ou outros dados locais sensíveis em relatórios públicos.
