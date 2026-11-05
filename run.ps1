param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & .\.venv\Scripts\python.exe -m uvicorn app.studio:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "MeshPort exited with code $LASTEXITCODE" }
} finally { Pop-Location }
