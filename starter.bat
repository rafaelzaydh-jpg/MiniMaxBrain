@echo off
setlocal EnableExtensions
title MiniMaxBrain 0.3 - Direct MMB Runtime

set "PYTHON_CMD="
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    echo [ERRO] Python 3.11+ nao foi encontrado.
    pause
    exit /b 1
)

cd /d "%~dp0"

%PYTHON_CMD% -m pip install --quiet -e .
if errorlevel 1 (
    echo [ERRO] Falha ao instalar o pacote local.
    pause
    exit /b 1
)

if not exist "%~dp0conversor" mkdir "%~dp0conversor"
if not exist "%~dp0modelos" mkdir "%~dp0modelos"

:menu
call :find_bundle
echo.
echo ================================================================
echo             MiniMaxBrain 0.3 - MMB Direto
echo ================================================================
echo [1] Construir/testar backend nativo
echo [2] Converter GGUF para MMBW
echo [3] Preparar bundle MMBW ja convertido
echo [4] Validar bundle
echo [5] Testar pager MMBW
echo [6] Chat direto MMB
echo [7] Web/API direto MMB
echo [8] Abrir pasta modelos
echo [9] Sair
echo ================================================================
if defined CONFIG_PATH (
    echo Bundle atual: %BUNDLE_PATH%
    echo Config:       %CONFIG_PATH%
) else if defined BUNDLE_PATH (
    echo Bundle detectado sem gate.json: %BUNDLE_PATH%
) else (
    echo Bundle atual: nenhum bundle *-mmbw detectado
)
echo.
set "opcao="
set /p opcao="Opcao (1-9): "

if "%opcao%"=="1" goto :native
if "%opcao%"=="2" goto :convert
if "%opcao%"=="3" goto :prepare
if "%opcao%"=="4" goto :check
if "%opcao%"=="5" goto :smoke
if "%opcao%"=="6" goto :chat
if "%opcao%"=="7" goto :web
if "%opcao%"=="8" goto :folder
if "%opcao%"=="9" goto :end
echo Opcao invalida.
goto :menu

:find_bundle
set "BUNDLE_PATH="
set "CONFIG_PATH="
for /d %%d in ("%~dp0conversor\*-mmbw") do (
    if exist "%%d\model.mmb-map.json" (
        set "BUNDLE_PATH=%%d"
        if exist "%%d\gate.json" set "CONFIG_PATH=%%d\gate.json"
    )
)
for /d %%d in ("%~dp0modelos\*-mmbw") do (
    if exist "%%d\model.mmb-map.json" (
        set "BUNDLE_PATH=%%d"
        if exist "%%d\gate.json" set "CONFIG_PATH=%%d\gate.json"
    )
)
exit /b 0

:require_config
if defined CONFIG_PATH exit /b 0
if defined BUNDLE_PATH (
    echo [*] Criando gate.json para o bundle existente...
    %PYTHON_CMD% "%~dp0mmb.py" prepare --bundle "%BUNDLE_PATH%" --cache-gib 1
    if errorlevel 1 exit /b 1
    set "CONFIG_PATH=%BUNDLE_PATH%\gate.json"
    exit /b 0
)
echo [ERRO] Nenhum bundle MMBW foi encontrado em conversor\ ou modelos\.
echo Use a opcao [2] para converter ou [3] para informar um bundle existente.
pause
exit /b 1

:native
cls
echo [*] Ativando ambiente MSVC x64...

set "MMB_VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
if exist "%MMB_VSDEVCMD%" (
    call "%MMB_VSDEVCMD%" -arch=x64
) else (
    echo [ERRO] VsDevCmd.bat nao encontrado em:
    echo        %MMB_VSDEVCMD%
    echo.
    echo Abra um Developer Command Prompt do Visual Studio 2026 e execute novamente.
    pause
    goto :menu
)

where cl >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERRO] O ambiente Visual Studio foi chamado, mas cl.exe ainda nao apareceu.
    pause
    goto :menu
)

where nmake >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERRO] O ambiente Visual Studio foi chamado, mas nmake.exe ainda nao apareceu.
    pause
    goto :menu
)

echo [OK] MSVC ativado.
where cl
where nmake
echo.

%PYTHON_CMD% "%~dp0tools\build_native.py"
if errorlevel 1 (
    echo.
    echo [ERRO] O backend nativo nao foi construido.
) else (
    echo.
    echo [OK] Backend direto MMB compilado e testado.
)
echo.
pause
goto :menu

:convert
cls
call "%~dp0conversor.bat"
goto :menu

:prepare
cls
set "BUNDLE_INPUT="
set /p BUNDLE_INPUT="Cole o caminho completo da pasta *-mmbw: "
if not defined BUNDLE_INPUT goto :menu
%PYTHON_CMD% "%~dp0mmb.py" prepare --bundle "%BUNDLE_INPUT%" --cache-gib 1
echo.
pause
goto :menu

:check
cls
call :require_config
if errorlevel 1 goto :menu
%PYTHON_CMD% "%~dp0mmb.py" check --config "%CONFIG_PATH%"
echo.
pause
goto :menu

:smoke
cls
call :require_config
if errorlevel 1 goto :menu
%PYTHON_CMD% "%~dp0mmb.py" smoke --config "%CONFIG_PATH%" --blocks 4
echo.
pause
goto :menu

:chat
cls
call :require_config
if errorlevel 1 goto :menu
%PYTHON_CMD% "%~dp0mmb.py" chat --config "%CONFIG_PATH%" --tokens 128 --ctx 2048
echo.
pause
goto :menu

:web
cls
call :require_config
if errorlevel 1 goto :menu
%PYTHON_CMD% "%~dp0mmb.py" web --config "%CONFIG_PATH%" --port 8080 --open-browser --ctx 2048
echo.
pause
goto :menu

:folder
start "" explorer "%~dp0modelos"
goto :menu

:end
exit /b 0
