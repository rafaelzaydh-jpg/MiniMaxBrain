@echo off
setlocal EnableExtensions
title MiniMaxBrain - Conversor de Modelos GGUF
cd /d "%~dp0"
set "ROOT=%CD%"

echo.
echo =================================================================
echo       MiniMaxBrain 0.3 - Conversor GGUF para MMBW
echo =================================================================
echo.

set "PYTHON_EXE="
set "PYTHON_ARGS="

if defined MMB_PYTHON_EXE (
    set "PYTHON_EXE=%MMB_PYTHON_EXE%"
    set "PYTHON_ARGS=%MMB_PYTHON_ARGS%"
    goto :run_wizard
)

if exist "%ROOT%\runtime\python\python.exe" (
    set "PYTHON_EXE=%ROOT%\runtime\python\python.exe"
    goto :run_wizard
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto :run_wizard
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto :run_wizard
)

echo [*] Python nao encontrado. Preparando runtime portatil oficial...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\bootstrap_portable_python.ps1" -ProjectRoot "%ROOT%"
if errorlevel 1 (
    echo [ERRO] Nao foi possivel preparar Python.
    pause
    exit /b 1
)
set "PYTHON_EXE=%ROOT%\runtime\python\python.exe"

:run_wizard
if not exist "%ROOT%\conversor" mkdir "%ROOT%\conversor"
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\tools\conversor_wizard.py"

echo.
pause
