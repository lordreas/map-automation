param([switch]$Headless, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectionPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectionPython)) {
    throw 'Create map_projection/.venv and install requirements first; see map_projection/README.md.'
}
Set-Location -LiteralPath $projectRoot
$projectionArgs = @('-m', 'map_projection')
if ($Headless) { $projectionArgs += '--headless' }
if (-not $NoBrowser) { $projectionArgs += '--open-browser' }
& $projectionPython @projectionArgs
