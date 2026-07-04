# start-ordered.ps1
# Native Windows PowerShell script to start A2A agents and frontend in order.

# Keep track of background processes to clean them up on exit
$processes = @()

function CleanUp {
    Write-Host "`n🛑 Stopping all services..." -ForegroundColor Red
    foreach ($p in $processes) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
    exit
}

# Trap Ctrl+C / SIGINT
$null = [System.Console]::TreatControlCAsInput
$ExecutionContext.InvokeCommand.NewScriptBlock({
    [System.Management.Automation.PSSecurityException]
}) | Out-Null

trap { CleanUp }

# Helper function to wait for a port to be ready
function WaitForPort($port, $name) {
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "⏳ Waiting for $name on port $port..." -ForegroundColor Yellow
    while ($true) {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            break
        }
        Start-Sleep -Seconds 2
    }
    Write-Host "✅ $name is ready!" -ForegroundColor Green
}

Write-Host "🚀 Starting BeeAI Ecosystem in order (Windows Native)..." -ForegroundColor Cyan

# 1. Start Priority Agent (Multimodal)
Write-Host "Step 1: Starting Priority Agent (10003)..." -ForegroundColor Cyan
$p = Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev:agent:multimodal" -PassThru -WindowStyle Minimized
$processes += $p

# Wait for priority agent
WaitForPort 10003 "Multimodal Agent"

# 2. Start Remaining Agents (Images & Medical)
Write-Host "`nStep 2: Starting Remaining Agents (10001 & 10002)..." -ForegroundColor Cyan
$p1 = Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev:agent:images" -PassThru -WindowStyle Minimized
$p2 = Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev:agent:medical" -PassThru -WindowStyle Minimized
$processes += $p1
$processes += $p2

# Wait for remaining agents
WaitForPort 10001 "Images Agent"
WaitForPort 10002 "Medical Agent"

# 3. Start Orchestrator
Write-Host "`nStep 3: Starting Orchestrator (Backend)..." -ForegroundColor Cyan
$p3 = Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev:backend" -PassThru -WindowStyle Minimized
$processes += $p3
WaitForPort 12000 "Orchestrator"

# 4. Start Frontend
Write-Host "`nStep 4: Starting Frontend..." -ForegroundColor Cyan
Write-Host "--------------------------------------------------" -ForegroundColor Cyan
Write-Host "Frontend will be available at http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop everything." -ForegroundColor Yellow
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

# Run frontend in the foreground
npm run dev:frontend

# Cleanup on exit
CleanUp
