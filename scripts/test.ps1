$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$condaExe = $env:LAYERED1D_CONDA_EXE
if ([string]::IsNullOrWhiteSpace($condaExe)) {
    $condaExe = "I:\programming\anaconda3\Scripts\conda.exe"
}

$condaEnv = $env:LAYERED1D_CONDA_ENV
if ([string]::IsNullOrWhiteSpace($condaEnv)) {
    $condaEnv = "multilayer_model"
}

if (-not (Test-Path -LiteralPath $condaExe)) {
    throw "Conda executable was not found: $condaExe"
}

& $condaExe run --no-capture-output -n $condaEnv python -m unittest discover -s tests -v
