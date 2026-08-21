param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$RuntimeDir = Join-Path $ProjectRoot "runtime\python"

if (Test-Path (Join-Path $RuntimeDir "python.exe")) {
    Write-Host "[OK] Python portatil ja esta pronto."
    exit 0
}

$Version = "3.11.9"
$Url = "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip"
$ExpectedSha256 = "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b"
$TempZip = Join-Path $env:TEMP "minimaxbrain-python-$Version-embed-amd64.zip"
$TempDir = Join-Path $env:TEMP "minimaxbrain-python-$Version"

Write-Host "[*] Python 3.11+ nao foi encontrado."
Write-Host "[*] Baixando CPython $Version embeddable oficial (python.org)..."

if (Test-Path $TempZip) { Remove-Item -Force $TempZip }
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }

Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $TempZip

$Actual = (Get-FileHash -Algorithm SHA256 $TempZip).Hash.ToLowerInvariant()
if ($Actual -ne $ExpectedSha256) {
    Remove-Item -Force $TempZip -ErrorAction SilentlyContinue
    throw "Checksum SHA-256 do Python portatil nao confere. Esperado=$ExpectedSha256 Obtido=$Actual"
}

New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
Expand-Archive -Path $TempZip -DestinationPath $TempDir -Force

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RuntimeDir) | Out-Null
if (Test-Path $RuntimeDir) { Remove-Item -Recurse -Force $RuntimeDir }
Move-Item -Path $TempDir -Destination $RuntimeDir

$Pth = Get-ChildItem -Path $RuntimeDir -Filter "python*._pth" | Select-Object -First 1
if ($null -eq $Pth) {
    throw "Arquivo python*._pth nao foi encontrado no pacote embeddable."
}

$Lines = @(Get-Content $Pth.FullName)
if ($Lines -notcontains "..\..") {
    $Lines += "..\.."
}
# No third-party packages are required; keep isolated mode and do not enable site.
Set-Content -Path $Pth.FullName -Value $Lines -Encoding ASCII

Remove-Item -Force $TempZip -ErrorAction SilentlyContinue

$PythonExe = Join-Path $RuntimeDir "python.exe"
& $PythonExe -c "import sys; import minimaxbrain; print('[OK] Python portatil:', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) {
    throw "Python portatil foi extraido, mas nao conseguiu importar MiniMaxBrain."
}

Write-Host "[OK] Runtime Python portatil preparado."
exit 0
