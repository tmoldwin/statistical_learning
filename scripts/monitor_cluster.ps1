# Auto-poll cluster sweep status. Agent or user can leave this running.
param(
    [string]$RemoteHost = "loginserver.elsc.huji.ac.il",
    [string]$RemoteRepo = "code/statistical_learning",
    [string]$LogGlob = "sixteen_word_lengths_ns_h500_*",
    [int]$IntervalSec = 60,
    [switch]$Once
)

$ErrorActionPreference = "Continue"
$MobaBash = Join-Path $env:APPDATA "MobaXterm\slash\mx86_64b\bin\bash.exe"
if (-not (Test-Path $MobaBash)) { throw "MobaXterm bash not found at $MobaBash" }

function Invoke-MobaSsh([string]$Command) {
    $cmd = ($Command -replace "`r?`n", " " -replace "\s+", " ").Trim()
    $escaped = $cmd -replace "'", "'\\''"
    & $MobaBash -lc "ssh $RemoteHost -p 22 -o ConnectTimeout=15 -o LogLevel=ERROR -C -j '$escaped'" 2>$null
}

function Show-ClusterStatus {
    $remote = "cd ~/$RemoteRepo && git pull -q 2>/dev/null; python3 scripts/cluster_status.py --log-glob $LogGlob"
    $raw = Invoke-MobaSsh $remote
    foreach ($line in @($raw)) {
        if ($line -and $line -notmatch '^(From |Already |Updating |Fast-forward|\s*\* )') {
            Write-Host $line
        }
    }
}

if (-not $Once) {
    Write-Host ""
    Write-Host "Cluster monitor - $RemoteHost - every ${IntervalSec}s (Ctrl+C to stop)" -ForegroundColor Cyan
    Write-Host ""
}

while ($true) {
    if (-not $Once) {
        Write-Host "--- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---" -ForegroundColor DarkGray
    }
    Show-ClusterStatus
    Write-Host ""
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
}
