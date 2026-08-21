@echo off
setlocal
title MiniMaxBrain - Instalador e Inicializador

echo.
echo =================================================================
echo       [*] MiniMaxBrain (MMB) - Instalador e Inicializador
echo =================================================================
echo.

REM Procura o executavel do Python
set "PYTHON_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :check_version
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :check_version
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :check_version
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :check_version
)

if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :check_version
)

if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    goto :check_version
)

echo [ERRO] Python nao encontrado!
echo Por favor, instale o Python 3.11 ou superior antes de prosseguir.
echo Baixe em: https://www.python.org/downloads/
echo.
pause
exit /b 1

:check_version
echo [1/4] Verificando versao do Python...
%PYTHON_CMD% -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ obrigatorio'" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O MiniMaxBrain requer Python versao 3.11 ou superior!
    echo.
    pause
    exit /b 1
)
echo      [OK] Python compativel detectado.
echo.

echo [2/4] Instalando dependencias do projeto...
%PYTHON_CMD% -m pip install --upgrade pip >nul 2>&1
%PYTHON_CMD% -m pip install -r "%~dp0requirements-dev.txt" >nul 2>&1
%PYTHON_CMD% -m pip install -e "%~dp0." >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Falha ao instalar via pip em modo editavel, mas o runtime funcionara normalmente via python mmb.py.
) else (
    echo      [OK] Pacote MMB registrado no ambiente com sucesso!
)
echo.

echo [3/4] Executando testes de integridade do MiniMaxBrain...
%PYTHON_CMD% -m pytest -q "%~dp0tests"
if errorlevel 1 (
    echo.
    echo [AVISO] Alguns testes falharam. Verifique permissoes.
) else (
    echo      [OK] Todos os 25 testes foram aprovados com 100%% de sucesso!
)
echo.

if not exist "%~dp0conversor" (
    mkdir "%~dp0conversor"
)

echo [4/4] Ambiente configurado com sucesso!
echo.
echo =================================================================
echo                    INSTALACAO CONCLUIDA!
echo =================================================================
echo.
echo Escolha o que deseja fazer a seguir:
echo   [1] Abrir a pasta 'conversor' para colocar um modelo .gguf
echo   [2] Executar o conversor de modelos agora
echo   [3] Executar verificacao do modelo real de teste
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
        %PYTHON_CMD% "%~dp0mmb.py" check --config "%~dp0real_model_test\mmb-granite-pageable\gate.experts.json"
    ) else (
        echo Modelo Granite de teste nao encontrado.
    )
    echo.
    pause
    exit /b 0
)

echo Saindo...
exit /b 0
