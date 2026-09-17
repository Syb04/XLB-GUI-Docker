param(
    [switch]$Gpu,
    [switch]$LocalData,
    [string]$Distro = 'Ubuntu-24.04',
    [ValidateRange(1024,65535)][int]$Port = 8766
)
$ErrorActionPreference = 'Stop'
$composeArgs = @('compose', '-f', 'compose.yaml')
if ($Gpu) { $composeArgs += @('-f', 'compose.gpu.yaml') }
if ($LocalData) { $composeArgs += @('-f', 'compose.local.yaml') }
$composeArgs += @('up', '--build', '-d', '--wait')
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $previousPort = $env:XLB_PORT
    try {
        $env:XLB_PORT = "$Port"
        Push-Location $PSScriptRoot
        try { & docker @composeArgs } finally { Pop-Location }
    } finally { $env:XLB_PORT = $previousPort }
} else {
    # --exec avoids passing Windows backslashes through the Linux shell.
    $pathOutput = wsl.exe -d $Distro --exec wslpath -a $PSScriptRoot
    if ($LASTEXITCODE -ne 0 -or -not $pathOutput) { throw 'WSL path conversion failed' }
    $linuxPath = ($pathOutput -join "`n").Trim()
    wsl.exe -d $Distro -u root --cd $linuxPath -- env "XLB_PORT=$Port" docker @composeArgs
}
if ($LASTEXITCODE -ne 0) { throw 'Docker startup failed; check the output above.' }
Write-Host "XLB Workbench: http://127.0.0.1:$Port"
if ($Gpu) { Write-Host 'Select cuda:0 in the study settings to run on the GPU.' }
