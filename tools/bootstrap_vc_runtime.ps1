param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Url = "https://aka.ms/vc14/vc_redist.x64.exe"
$Installer = Join-Path $env:TEMP "minimaxbrain-vc_redist.x64.exe"

Write-Host "[*] Baixando Microsoft Visual C++ v14 Redistributable x64..."
Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Installer

$Signature = Get-AuthenticodeSignature $Installer
if ($Signature.Status -ne "Valid") {
    Remove-Item -Force $Installer -ErrorAction SilentlyContinue
    throw "O instalador VC++ baixado nao possui assinatura Authenticode valida."
}
if ($null -eq $Signature.SignerCertificate -or
    $Signature.SignerCertificate.Subject -notmatch "Microsoft") {
    Remove-Item -Force $Installer -ErrorAction SilentlyContinue
    throw "O instalador VC++ nao foi assinado pela Microsoft."
}

Write-Host "[*] Instalando o runtime C++ oficial. O Windows pode solicitar permissao."
$Process = Start-Process -FilePath $Installer `
    -ArgumentList "/install","/quiet","/norestart" `
    -Verb RunAs -Wait -PassThru

Remove-Item -Force $Installer -ErrorAction SilentlyContinue

if ($Process.ExitCode -notin @(0, 1638, 3010)) {
    throw "Instalacao do VC++ Redistributable falhou com codigo $($Process.ExitCode)."
}

if ($Process.ExitCode -eq 3010) {
    Write-Host "[INFO] O runtime foi instalado; o Windows recomenda reiniciar."
} else {
    Write-Host "[OK] Microsoft Visual C++ Runtime pronto."
}
exit 0
