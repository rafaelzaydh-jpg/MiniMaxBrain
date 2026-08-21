@echo off
setlocal
title MiniMaxBrain - Conversor de Modelos GGUF

echo.
echo =================================================================
echo       [*] MiniMaxBrain (MMB) - Conversor Automatico GGUF
echo =================================================================
echo.

REM Procura o executavel do Python
set "PYTHON_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :run_wizard
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :run_wizard
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :run_wizard
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :run_wizard
)

if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :run_wizard
)

if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    goto :run_wizard
)

echo [ERRO] Python 3.11+ nao foi encontrado no sistema!
echo Por favor, instale o Python ou marque a opcao 'Add Python to PATH' durante a instalacao.
echo Baixe em: https://www.python.org/downloads/
echo.
pause
exit /b 1

:run_wizard
if not exist "%~dp0conversor" (
    mkdir "%~dp0conversor"
)

%PYTHON_CMD% "%~dp0tools\conversor_wizard.py"

echo.
echo Pressione qualquer tecla para fechar...
pause >nul
