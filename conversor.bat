@echo off
chcp 65001 >nul
title MiniMaxBrain - Conversor de Modelos GGUF

echo.
echo =================================================================
echo        🧠 MiniMaxBrain (MMB) - Conversor Automatico GGUF
echo =================================================================
echo.

:: Verifica se o Python está disponível
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado no sistema!
    echo Por favor, instale o Python 3.11+ ou adicione-o ao PATH do Windows.
    echo Baixe em: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Verifica se a pasta conversor existe, se não cria
if not exist "%~dp0conversor" (
    mkdir "%~dp0conversor"
)

:: Executa o assistente Python
python "%~dp0tools\conversor_wizard.py"

echo.
echo Pressione qualquer tecla para fechar...
pause >nul
