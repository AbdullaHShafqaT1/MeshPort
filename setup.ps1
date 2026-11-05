# MeshPort local client. No local GPU, inference engines, or model weights.
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path .venv)) {
        py -3.10 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Install Python 3.10 before running setup." }
    }
    $meshportPython = ".\.venv\Scripts\python.exe"
    & $meshportPython -c "import sys; assert sys.version_info[:2] == (3,10), 'MeshPort FBX requires Python 3.10'"
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 is required." }
    & $meshportPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Client dependency installation failed." }
    & $meshportPython -m pip install --target .fbx-tools -r requirements-fbx.txt
    if ($LASTEXITCODE -ne 0) { throw "FBX dependency installation failed." }
    Write-Host "MeshPort is ready. Run .\run.ps1 and open http://127.0.0.1:8000"
} finally { Pop-Location }
