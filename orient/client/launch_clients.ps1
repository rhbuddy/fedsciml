# Orient — launch N CLI clients as background processes (from the client/ directory)
#   .\launch_clients.ps1                 # 5 clients (default) for gramacy_lee
#   .\launch_clients.ps1 -Count 3 -Problem poisson
#   .\launch_clients.ps1 -Count 5 -Stop  # kill all running clients
param(
    [int]$Count = 5,
    [string]$Problem = "gramacy_lee",
    [string]$Server = "http://127.0.0.1:8000",
    [switch]$Stop
)

$Venv = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Py = if (Test-Path $Venv) { $Venv } else { "python" }

if ($Stop) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*app.main*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output "stopped PID $($_.ProcessId)" }
    return
}

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Output "Launching $Count client(s) for '$Problem' against $Server"
for ($i = 1; $i -le $Count; $i++) {
    $bundle = Join-Path $PSScriptRoot "examples\bundles\${Problem}_client$i"
    if (-not (Test-Path $bundle)) {
        Write-Warning "Missing bundle: $bundle  (run: .\run.ps1 datasets --clients $Count)"
        continue
    }
    $out = Join-Path $logDir "client$i.log"
    $err = Join-Path $logDir "client$i.err"
    Start-Process -FilePath $Py `
        -ArgumentList "-m", "app.main", "--server", $Server, "--client-id", "client-$i", "--dataset", $bundle `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru |
        ForEach-Object { Write-Output "  client-$i  PID $($_.Id)  -> $out" }
}
Write-Output ""
Write-Output "All clients started. Check who registered:"
Write-Output "  curl.exe $Server/clients"