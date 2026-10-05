# Orient — run the Server app (from the backend/ directory)
#   .\run.ps1            start the REST server on :8000
#   .\run.ps1 dashboard  start the Streamlit dashboard
#   .\run.ps1 install    install the backend dependencies
#   .\run.ps1 docs       print the OpenAPI docs URL
param([string]$Target = "server")

# Prefer the project venv (orient\.venv) so you never touch the system Python.
$Venv = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Py = if (Test-Path $Venv) { $Venv } else { "python" }
if ($Py -eq "python") {
    Write-Warning "No venv found at ..\.venv - falling back to system python."
}

switch ($Target) {
    "server"    { & $Py -m app.main }
    "dashboard" { & $Py -m streamlit run ui/dashboard.py }
    "install"   { & $Py -m pip install -r requirements.txt }
    "docs"      { Write-Output "Interactive API docs: http://127.0.0.1:8000/docs" }
    default     { Write-Output "Usage: .\run.ps1 [server|dashboard|install|docs]" }
}
