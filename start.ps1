param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$PythonPath = '/home/tk/src/XLB/.venv/bin/python',
    [int]$Port = 8766
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$linuxPath = (wsl.exe -d $Distro -- wslpath -a $projectPath).Trim()
if ($LASTEXITCODE -ne 0) { throw 'WSL path conversion failed' }
Write-Host "XLB Workbench: http://127.0.0.1:$Port"
wsl.exe -d $Distro --cd $linuxPath -- $PythonPath -m workbench.server --port $Port
exit $LASTEXITCODE
