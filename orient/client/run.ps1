# Orient — run the Client app (from the client/ directory)
#   .\run.ps1 datasets   generate example dataset bundles
#   .\run.ps1 ui         start the Streamlit client UI
#   .\run.ps1 cli        run the CLI client (edit the arguments below)
#   .\run.ps1 install    install the client dependencies
param([string]$Target = "cli")

$Venv = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Py = if (Test-Path $Venv) { $Venv } else { "python" }
if ($Py -eq "python") {
    Write-Warning "No venv found at ..\.venv - falling back to system python."
}

switch ($Target) {
    "datasets" { & $Py examples/make_datasets.py }
    "ui"       { & $Py -m streamlit run ui/app.py }
    "cli"      { & $Py -m app.main --server http://127.0.0.1:8000 --client-id client-1 --dataset examples/bundles/gramacy_lee_client1 }
    "install"  { & $Py -m pip install -r requirements.txt }
    default    { Write-Output "Usage: .\run.ps1 [datasets|ui|cli|install]" }
}
