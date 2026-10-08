# Orient — start the whole system with ONE command, then drive it from the browser.
#
#   .\start.ps1                  # server + dashboard, then opens the dashboard
#   .\start.ps1 -WithClientUi    # also starts the client UI (port 8502)
#   .\start.ps1 -Stop            # stop everything Orient started
#
# After it runs:
#   Dashboard  -> http://localhost:8501   (start runs, watch metrics)
#   Client UI  -> http://localhost:8502   (open one tab per client)
param([switch]$WithClientUi, [switch]$Stop)

$Venv   = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Py     = if (Test-Path $Venv) { $Venv } else { "python" }
$Logs   = Join-Path $PSScriptRoot "logs"
$ApiUrl = "http://127.0.0.1:8000"

function Stop-Orient {
    $killed = 0
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -like "*app.main*" -or $_.CommandLine -like "*streamlit*"
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Output "  stopped PID $($_.ProcessId)"
        $killed++
    }
    Write-Output "Stopped $killed Orient process(es)."
}

function Wait-Port([int]$Port, [int]$Seconds = 90) {
    # Raw TCP connect. Deliberately NOT Invoke-WebRequest: in a hidden
    # (NonInteractive) PowerShell it throws and the launcher silently stalls.
    for ($i = 0; $i -lt ($Seconds * 2); $i++) {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $client.Connect("127.0.0.1", $Port)
            $client.Close()
            return $true
        } catch { }
        finally { $client.Close() }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

if ($Stop) { Stop-Orient; return }

if ($Py -eq "python") { Write-Warning "No venv at .venv - using system python." }
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

Write-Output "Starting Orient..."
Write-Output ""

# 1) Server
Start-Process -FilePath $Py -ArgumentList "-m", "app.main" `
    -WorkingDirectory (Join-Path $PSScriptRoot "backend") `
    -RedirectStandardOutput "$Logs\server.log" -RedirectStandardError "$Logs\server.err" `
    -WindowStyle Hidden | Out-Null
if (Wait-Port 8000) { Write-Output "  [ok] Server API      $ApiUrl" }
else { Write-Error "  [!!] Server failed to start - see logs\server.err"; return }

# 2) Dashboard
# NOTE: the whole ArgumentList must stay on ONE logical line. A trailing comma
# before the backtick continuation silently turns -WorkingDirectory into another
# array element, which makes Start-Process fail with "positional parameter".
Start-Process -FilePath $Py `
    -ArgumentList "-m", "streamlit", "run", "ui/dashboard.py", "--server.port", "8501", "--server.headless", "true" `
    -WorkingDirectory (Join-Path $PSScriptRoot "backend") `
    -RedirectStandardOutput "$Logs\dashboard.log" -RedirectStandardError "$Logs\dashboard.err" `
    -WindowStyle Hidden | Out-Null
if (Wait-Port 8501) { Write-Output "  [ok] Dashboard       http://localhost:8501" }
else { Write-Warning "  [!!] Dashboard did not respond - see logs\dashboard.err" }

# 3) Optional client UI
if ($WithClientUi) {
    Start-Process -FilePath $Py `
        -ArgumentList "-m", "streamlit", "run", "ui/app.py", "--server.port", "8502", "--server.headless", "true" `
        -WorkingDirectory (Join-Path $PSScriptRoot "client") `
        -RedirectStandardOutput "$Logs\clientui.log" -RedirectStandardError "$Logs\clientui.err" `
        -WindowStyle Hidden | Out-Null
    if (Wait-Port 8502) { Write-Output "  [ok] Client UI       http://localhost:8502" }
    else { Write-Warning "  [!!] Client UI did not respond - see logs\clientui.err" }
}

Write-Output ""
Write-Output "=============================================="
Write-Output " 1. Open the DASHBOARD  http://localhost:8501"
Write-Output " 2. Pick problem + aggregator, press 'Start run'"
Write-Output " 3. Open the CLIENT UI  http://localhost:8502"
Write-Output "    (open one browser TAB per client, and give"
Write-Output "     each tab a different Client ID)"
Write-Output " 4. In each client tab: Load dataset ->"
Write-Output "    Connect & register -> Start federated training"
Write-Output "=============================================="
Write-Output ""
Write-Output "Stop everything later with:  .\start.ps1 -Stop"

Start-Process "http://localhost:8501"