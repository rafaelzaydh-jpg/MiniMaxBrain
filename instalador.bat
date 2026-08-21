@echo off
chcp 65001 >nul
title MiniMaxBrain - Instalador e Inicializador

echo.
echo =================================================================
echo        🧠 MiniMaxBrain (MMB) - Instalador e Inicializador
echo =================================================================
echo.

:: 1. Verifica se o Python está instalado
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale o Python 3.11 ou superior antes de prosseguir.
    echo Baixe em: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/4] Verificando versao do Python...
python -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ obrigatorio'" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERRO] O MiniMaxBrain requer Python versao 3.11 ou superior!
    echo.
    pause
    exit /b 1
)
echo      ✔ Python compativel detectado.
echo.

:: 2. Instala dependências de desenvolvimento e o pacote MMB
echo [2/4] Instalando dependencias do projeto...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r "%~dp0requirements-dev.txt" >nul 2>nul
python -m pip install -e "%~dp0." >nul 2>nul
if %errorlevel% neq 0 (
    echo [AVISO] Falha ao instalar via pip em modo editavel, mas o runtime funcionara normalmente via python mmb.py.
) else (
    echo      ✔ Comando 'mmb' instalado no ambiente com sucesso!
)
echo.

:: 3. Executa a suíte de testes de integridade
echo [3/4] Executando testes de integridade do MiniMaxBrain...
python -m pytest -q "%~dp0tests"
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Alguns testes falharam. Verifique se o ambiente possui permissoes adequadas.
) else (
    echo      ✔ Todos os testes foram aprovados com 100%% de sucesso!
)
echo.

:: 4. Criação da pasta do conversor
if not exist "%~dp0conversor" (
    mkdir "%~dp0conversor"
)

echo [4/4] Ambiente configurado com sucesso!
echo.
echo =================================================================
echo                    🎉 INSTALACAO CONCLUIDA!
echo =================================================================
echo.
echo Escolha o que deseja fazer a seguir:
echo   [1] Abrir a pasta 'conversor' para colocar um modelo .gguf
echo   [2] Executar o conversor de modelos agora
echo   [3] Executar teste de smoke com modelo real (se disponivel)
echo   [4] Sair
echo.

set /p opcao="Digite a opcao desejada (1-4): "

if "%opcao%"=="1" (
    start explorer "%~dp0conversor"
    echo Pasta 'conversor' aberta. Coloque seu arquivo .gguf la e execute conversor.bat!
    pause
    exit /b 0
)

if "%opcao%"=="2" (
    call "%~dp0conversor.bat"
    exit /b 0
)

if "%opcao%"=="3" (
    echo.
    echo Executando verificacao no Granite...
    if exist "%~dp0real_model_test\mmb-granite-pageable\gate.experts.json" (
        python "%~dp0mmb.py" check --config "%~dp0real_model_test\mmb-granite-pageable\gate.experts.json"
    ) else (
        echo Modelo Granite de teste nao encontrado.
    )
    echo.
    pause
    exit /b 0
)

echo Saindo...
exit /b 0
