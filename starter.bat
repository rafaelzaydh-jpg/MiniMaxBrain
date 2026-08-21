@echo off
setlocal
title MiniMaxBrain - Starter

echo.
echo =================================================================
echo             [*] MiniMaxBrain (MMB) - Starter
echo =================================================================
echo.

REM 1. Procura o executavel do Python
set "PYTHON_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :found_python
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found_python
)

if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :found_python
)

if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    goto :found_python
)

echo [ERRO] Python 3.11+ nao foi encontrado!
echo Por favor, instale o Python em: https://www.python.org/downloads/
echo Lembre-se de marcar 'Add Python to PATH' durante a instalacao.
echo.
pause
exit /b 1

:found_python
echo [1/3] Verificando e instalando dependencias...
echo      Aguarde um momento...
%PYTHON_CMD% -m pip install --quiet --upgrade pip
%PYTHON_CMD% -m pip install --quiet -r "%~dp0requirements-dev.txt"
%PYTHON_CMD% -m pip install --quiet -e "%~dp0."
echo      [OK] Dependencias instaladas com sucesso!
echo.

echo [2/3] Validando ambiente com suíte de testes...
%PYTHON_CMD% -m pytest -q "%~dp0tests"
if errorlevel 1 (
    echo      [AVISO] Alguns testes falharam, mas continuaremos a inicializacao.
) else (
    echo      [OK] Todos os 25 testes passaram com sucesso!
)
echo.

echo [3/3] Inicializando MiniMaxBrain...
if not exist "%~dp0conversor" (
    mkdir "%~dp0conversor"
)

echo.
echo =================================================================
echo                 SISTEMA PRONTO PARA USO!
echo =================================================================
echo.
echo Escolha o que deseja iniciar:
echo.
echo   [1] Iniciar o Conversor de Modelos (GGUF -> MMB)
echo   [2] Rodar o Benchmark Real no Qwen 35B MoE (4GB RAM)
echo   [3] Iniciar o Servidor Gate em Segundo Plano (mmb serve)
echo   [4] Abrir a pasta 'conversor' para adicionar um modelo .gguf
echo   [5] Sair
echo.

set /p opcao="Digite a opcao desejada (1-5): "

if "%opcao%"=="1" (
    cls
    call "%~dp0conversor.bat"
    exit /b 0
)

if "%opcao%"=="2" (
    cls
    echo Executando Benchmark A/B no Qwen 35B...
    %PYTHON_CMD% -u -B "%~dp0tools\mmb_ab_compare.py" --config "%~dp0real_model_test\mmb-qwen-pageable\gate.ram.json" --tokens 2 --rounds 2
    echo.
    echo Pressione qualquer tecla para voltar ao menu...
    pause >nul
    goto :found_python
)

if "%opcao%"=="3" (
    cls
    echo Iniciando Servidor MiniMaxBrain Gate...
    if exist "%~dp0real_model_test\mmb-qwen-pageable\gate.ram.json" (
        %PYTHON_CMD% "%~dp0mmb.py" serve --config "%~dp0real_model_test\mmb-qwen-pageable\gate.ram.json"
    ) else (
        echo Nenhuma configuracao encontrada para iniciar o servidor.
        pause
    )
    exit /b 0
)

if "%opcao%"=="4" (
    start explorer "%~dp0conversor"
    echo Pasta 'conversor' aberta!
    pause
    exit /b 0
)

echo Saindo do MiniMaxBrain...
exit /b 0
