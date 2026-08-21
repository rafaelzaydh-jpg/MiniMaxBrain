@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MiniMaxBrain 0.3 - Direct MMB Runtime
cd /d "%~dp0"

set "ROOT=%CD%"
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "MMB_BACKEND_LIBRARY="

call :resolve_python
if errorlevel 1 (
    echo.
    echo [ERRO] Nao foi possivel preparar o runtime Python.
    echo        Verifique sua conexao ou instale Python 3.11+.
    pause
    exit /b 1
)

call :resolve_backend
if errorlevel 1 (
    echo.
    echo [AVISO] O backend nativo nao esta pronto.
    echo         A release de usuario deve conter:
    echo         runtime\windows-x64\mmb_backend.dll
    echo.
    echo Desenvolvedores podem recompilar em [D].
    pause
)

if not exist "%ROOT%\conversor" mkdir "%ROOT%\conversor"
if not exist "%ROOT%\modelos" mkdir "%ROOT%\modelos"

:menu
call :find_bundle
cls
echo ================================================================
echo             MiniMaxBrain 0.3 - MMB Direto
echo ================================================================
echo [1] Verificar runtime
echo [2] Converter GGUF para MMBW
echo [3] Preparar bundle MMBW ja convertido
echo [4] Validar bundle
echo [5] Testar pager MMBW
echo [6] Chat direto MMB
echo [7] Web Chat / API
echo [8] Abrir pasta modelos
echo [D] Ferramentas de desenvolvedor
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
if exist "%ROOT%\runtime\windows-x64\mmb_backend.dll" (
    echo Backend:      precompilado Windows x64
) else (
    echo Backend:      build local/fallback
)
echo.
set "opcao="
set /p opcao="Opcao: "

if "%opcao%"=="1" goto :runtime_check
if "%opcao%"=="2" goto :convert
if "%opcao%"=="3" goto :prepare
if "%opcao%"=="4" goto :check
if "%opcao%"=="5" goto :smoke
if "%opcao%"=="6" goto :chat
if "%opcao%"=="7" goto :web
if "%opcao%"=="8" goto :folder
if /i "%opcao%"=="D" goto :developer
if "%opcao%"=="9" goto :end
echo Opcao invalida.
timeout /t 1 >nul
goto :menu

:resolve_python
if exist "%ROOT%\runtime\python\python.exe" (
    set "PYTHON_EXE=%ROOT%\runtime\python\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    set "PYTHON_ARGS="
    exit /b 0
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
)

echo [*] Python 3.11+ nao encontrado. Preparando runtime portatil...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\bootstrap_portable_python.ps1" -ProjectRoot "%ROOT%"
if errorlevel 1 exit /b 1
if not exist "%ROOT%\runtime\python\python.exe" exit /b 1
set "PYTHON_EXE=%ROOT%\runtime\python\python.exe"
set "PYTHON_ARGS="
exit /b 0

:resolve_backend
if exist "%ROOT%\runtime\windows-x64\mmb_backend.dll" (
    set "MMB_BACKEND_LIBRARY=%ROOT%\runtime\windows-x64\mmb_backend.dll"
) else if exist "%ROOT%\native\build\Release\mmb_backend.dll" (
    set "MMB_BACKEND_LIBRARY=%ROOT%\native\build\Release\mmb_backend.dll"
) else (
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "from minimaxbrain.native import NativeLibrary; lib=NativeLibrary(); print('[OK] Backend nativo:', lib.path)" >"%TEMP%\mmb_backend_check.txt" 2>&1
if not errorlevel 1 (
    type "%TEMP%\mmb_backend_check.txt"
    del "%TEMP%\mmb_backend_check.txt" >nul 2>&1
    exit /b 0
)

echo.
type "%TEMP%\mmb_backend_check.txt"
del "%TEMP%\mmb_backend_check.txt" >nul 2>&1
echo.
echo [INFO] A DLL existe, mas o Windows nao conseguiu carrega-la.
echo        Em um PC novo normalmente falta o Microsoft Visual C++ Runtime.
choice /C SN /N /M "Instalar agora o runtime oficial da Microsoft? [S/N]: "
if errorlevel 2 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\bootstrap_vc_runtime.ps1" -ProjectRoot "%ROOT%"
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" %PYTHON_ARGS% -c "from minimaxbrain.native import NativeLibrary; lib=NativeLibrary(); print('[OK] Backend nativo:', lib.path)"
exit /b %errorlevel%

:find_bundle
set "BUNDLE_PATH="
set "CONFIG_PATH="
for /d %%d in ("%ROOT%\conversor\*-mmbw") do (
    if exist "%%d\model.mmb-map.json" (
        set "BUNDLE_PATH=%%d"
        if exist "%%d\gate.json" set "CONFIG_PATH=%%d\gate.json"
    )
)
for /d %%d in ("%ROOT%\modelos\*-mmbw") do (
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
    "%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" prepare --bundle "%BUNDLE_PATH%" --cache-gib 1
    if errorlevel 1 exit /b 1
    set "CONFIG_PATH=%BUNDLE_PATH%\gate.json"
    exit /b 0
)
echo [ERRO] Nenhum bundle MMBW foi encontrado em conversor\ ou modelos\.
echo Use [2] para converter ou [3] para informar um bundle existente.
pause
exit /b 1

:runtime_check
cls
echo ================================================================
echo MiniMaxBrain - Diagnostico
echo ================================================================
echo Python:
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print(sys.executable); print(sys.version)"
echo.
echo Backend:
call :resolve_backend
echo.
if defined BUNDLE_PATH (
    echo Bundle: %BUNDLE_PATH%
) else (
    echo Bundle: nenhum detectado
)
echo ================================================================
pause
goto :menu

:convert
cls
set "MMB_PYTHON_EXE=%PYTHON_EXE%"
set "MMB_PYTHON_ARGS=%PYTHON_ARGS%"
call "%ROOT%\conversor.bat"
goto :menu

:prepare
cls
set "BUNDLE_INPUT="
set /p BUNDLE_INPUT="Cole o caminho completo da pasta *-mmbw: "
if not defined BUNDLE_INPUT goto :menu
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" prepare --bundle "%BUNDLE_INPUT%" --cache-gib 1
echo.
pause
goto :menu

:check
cls
call :require_config
if errorlevel 1 goto :menu
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" check --config "%CONFIG_PATH%"
echo.
pause
goto :menu

:smoke
cls
call :require_config
if errorlevel 1 goto :menu
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" smoke --config "%CONFIG_PATH%" --blocks 4
echo.
pause
goto :menu

:chat
cls
call :require_config
if errorlevel 1 goto :menu
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" chat --config "%CONFIG_PATH%" --tokens 128 --ctx 2048
echo.
pause
goto :menu

:web
cls
call :require_config
if errorlevel 1 goto :menu
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\mmb.py" web --config "%CONFIG_PATH%" --port 8080 --open-browser --ctx 2048
echo.
pause
goto :menu

:folder
start "" explorer "%ROOT%\modelos"
goto :menu

:developer
cls
echo ================================================================
echo MiniMaxBrain - Ferramentas de desenvolvedor
echo ================================================================
echo O usuario normal NAO precisa desta etapa.
echo.
echo [1] Compilar/testar backend nativo
echo [2] Rodar testes Python
echo [3] Voltar
echo.
set "devop="
set /p devop="Opcao: "
if "%devop%"=="1" goto :developer_build
if "%devop%"=="2" goto :developer_pytest
goto :menu

:developer_build
set "MMB_VSDEVCMD=C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
if exist "%MMB_VSDEVCMD%" (
    call "%MMB_VSDEVCMD%" -arch=x64
) else (
    echo [ERRO] VsDevCmd.bat nao encontrado em:
    echo        %MMB_VSDEVCMD%
    echo Abra um Developer Command Prompt do Visual Studio e execute novamente.
    pause
    goto :menu
)
"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\tools\build_native.py"
if not errorlevel 1 (
    "%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%\tools\promote_windows_runtime.py"
)
pause
goto :menu

:developer_pytest
"%PYTHON_EXE%" %PYTHON_ARGS% -m pytest -q
pause
goto :menu

:end
exit /b 0
