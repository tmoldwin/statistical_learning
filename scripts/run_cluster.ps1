# Launch cluster.py on the ELSC login server from Windows (MobaXterm credentials).
#
# Examples:
#   .\scripts\run_cluster.ps1 -Action plan -Preset sixteen_word_lengths_ns_h500
#   .\scripts\run_cluster.ps1 -Action submit -Preset sixteen_word_lengths_ns_h500
#   .\scripts\run_cluster.ps1 -Action all -Preset sixteen_word_lengths_ns_h500 -Partition ss.cpu -Time 04:00:00
#   .\scripts\run_cluster.ps1 -Action submit -Tasks sixteen_word_ns_h500 -Seeds 1,2,3,5,7
#   .\scripts\run_cluster.ps1 -Action all -Preset sixteen_word_lengths_ns_h500 -SyncCheckpoints -Pull

param(
    [ValidateSet("plan", "submit", "wait", "plot", "all")]
    [string]$Action = "all",

    [string]$Preset = "",
    [string]$Name = "",
    [string]$Tasks = "",
    [string]$Seeds = "",
    [string]$ModelType = "rnn",
    [string]$Kinds = "trajectory_geometry,closed_loop_trajectories,learning_curves",

    [string]$Partition = "ss.cpu",
    [string]$Time = "04:00:00",
    [string]$Mem = "8G",

    [string]$RemoteHost = "loginserver.elsc.huji.ac.il",
    [string]$RemoteRepo = "code/statistical_learning",
    [string]$RemoteUser = "toviah.moldwin",

    [switch]$Push,
    [switch]$SyncCheckpoints,
    [switch]$Pull,
    [switch]$Smoke,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MobaBash = Join-Path $env:APPDATA "MobaXterm\slash\mx86_64b\bin\bash.exe"
if (-not (Test-Path $MobaBash)) { throw "MobaXterm bash not found at $MobaBash" }

function Invoke-MobaSsh([string]$Command) {
    $cmd = ($Command -replace "`r?`n", " " -replace "\s+", " ").Trim()
    & $MobaBash -lc "ssh $RemoteHost -p 22 -C -j '$cmd'"
    if ($LASTEXITCODE -ne 0) { throw "SSH failed: $cmd" }
}

function Invoke-MobaScp([string]$Source, [string]$Dest) {
    & $MobaBash -lc "scp -C -j '$Source' '$Dest'"
    if ($LASTEXITCODE -ne 0) { throw "SCP failed: $Source -> $Dest" }
}

function Build-ClusterArgs() {
    $parts = @("python3", "scripts/cluster.py", $Action)
    if ($Preset) { $parts += @("--preset", $Preset) }
    if ($Name) { $parts += @("--name", $Name) }
    if ($Tasks) { $parts += @("--tasks") + ($Tasks -split ",") }
    if ($Seeds) { $parts += @("--seeds") + ($Seeds -split "," | ForEach-Object { $_.Trim() }) }
    if ($ModelType) { $parts += @("--model-type", $ModelType) }
    if ($Kinds -and ($Action -in @("plot", "all"))) {
        $parts += @("--kinds") + ($Kinds -split "," | ForEach-Object { $_.Trim() })
    }
    if ($Smoke) { $parts += "--smoke" }
    if ($Force) { $parts += "--force" }
    if ($DryRun) { $parts += "--dry-run" }
  $parts += @("--partition", $Partition, "--time", $Time, "--mem", $Mem)
    return ($parts -join " ")
}

Set-Location $RepoRoot

if ($Push) {
    Write-Host ">> git push" -ForegroundColor Cyan
    git add -A
    $dirty = git status --porcelain
    if ($dirty) { git commit -m "cluster sweep updates" }
    git push origin HEAD
}

Write-Host ">> remote setup" -ForegroundColor Cyan
Invoke-MobaSsh "mkdir -p ~/code && if [ ! -d ~/$RemoteRepo/.git ]; then git clone https://github.com/tmoldwin/statistical_learning.git ~/$RemoteRepo; fi"
Invoke-MobaSsh "cd ~/$RemoteRepo && git pull && python3 -m pip install --user -q -r requirements.txt || true"

if ($SyncCheckpoints) {
    Write-Host ">> sync local checkpoints" -ForegroundColor Cyan
    $pattern = if ($Tasks) { ($Tasks -split "," | ForEach-Object { "$($_.Trim())" }) } else { @("*") }
    foreach ($task in $pattern) {
        $glob = if ($task -eq "*") { "$RepoRoot\experiments\*\rnn\model_seed*.npz" } else { "$RepoRoot\experiments\$task\rnn\model_seed*.npz" }
        Get-ChildItem $glob -ErrorAction SilentlyContinue | ForEach-Object {
            $t = $_.Directory.Parent.Name
            Invoke-MobaSsh "mkdir -p ~/$RemoteRepo/experiments/$t/rnn"
            Write-Host "  $($_.Name) -> $t"
            Invoke-MobaScp $_.FullName "${RemoteUser}@${RemoteHost}:$RemoteRepo/experiments/$t/rnn/"
        }
    }
}

$clusterCmd = Build-ClusterArgs
Write-Host ">> $clusterCmd" -ForegroundColor Cyan
Invoke-MobaSsh "cd ~/$RemoteRepo && $clusterCmd"

if ($Pull) {
    Write-Host ">> pull results" -ForegroundColor Cyan
    $cmpName = if ($Preset) { $Preset } elseif ($Name) { $Name } else { $null }
    if ($cmpName) {
        $localCmp = Join-Path $RepoRoot "experiments\comparisons\$cmpName"
        New-Item -ItemType Directory -Force -Path $localCmp | Out-Null
        Invoke-MobaScp "-r ${RemoteUser}@${RemoteHost}:$RemoteRepo/experiments/comparisons/$cmpName/*" $localCmp
    }
    if ($Tasks) {
        foreach ($task in ($Tasks -split ",")) {
            $task = $task.Trim()
            $localRnn = Join-Path $RepoRoot "experiments\$task\rnn"
            New-Item -ItemType Directory -Force -Path $localRnn | Out-Null
            Invoke-MobaScp "${RemoteUser}@${RemoteHost}:$RemoteRepo/experiments/$task/rnn/model_seed*.npz" $localRnn
        }
    }
}

Write-Host ">> done" -ForegroundColor Green
