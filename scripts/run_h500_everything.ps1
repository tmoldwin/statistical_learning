# End-to-end: push code, sync checkpoints, submit SLURM jobs, wait, plot, pull results.
# Uses MobaXterm's bash/ssh (saved credentials via -j flag).
#
# Usage:
#   .\scripts\run_h500_everything.ps1
#   .\scripts\run_h500_everything.ps1 -SkipPush -SkipWait

param(
    [string]$Preset = "sixteen_word_lengths_ns_h500",
    [string]$RemoteHost = "loginserver.elsc.huji.ac.il",
    [string]$RemoteRepo = "code/statistical_learning",
    [switch]$SkipPush,
    [switch]$SkipWait,
    [switch]$SkipPull
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MobaBash = Join-Path $env:APPDATA "MobaXterm\slash\mx86_64b\bin\bash.exe"

if (-not (Test-Path $MobaBash)) {
    throw "MobaXterm bash not found at $MobaBash"
}

function Invoke-MobaSsh([string]$Command) {
    $escaped = $Command -replace '"', '\"'
    & $MobaBash -lc "ssh $RemoteHost -p 22 -C -j `"$escaped`""
    if ($LASTEXITCODE -ne 0) { throw "SSH failed (exit $LASTEXITCODE): $Command" }
}

function Invoke-MobaScp([string[]]$ScpArgs) {
    $argStr = ($ScpArgs | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " "
    & $MobaBash -lc "scp -C -j $argStr"
    if ($LASTEXITCODE -ne 0) { throw "SCP failed (exit $LASTEXITCODE)" }
}

Set-Location $RepoRoot
Write-Host "=== 1/6 Commit and push ===" -ForegroundColor Cyan
if (-not $SkipPush) {
    git add experiment.py viz/compare/spec.py scripts/run_comparison_cluster.sh scripts/run_h500_everything.ps1
    $status = git status --porcelain
    if ($status) {
        git commit -m @"
Add 500-unit sixteen-word comparison and cluster launcher.

Enables h500 task presets and SLURM submission for the full seed sweep.
"@
        git push origin HEAD
    } else {
        Write-Host "Nothing to commit; pushing anyway..."
        git push origin HEAD
    }
} else {
    Write-Host "Skipped."
}

Write-Host "=== 2/6 Ensure remote repo ===" -ForegroundColor Cyan
$setupRemote = @"
set -euo pipefail
mkdir -p ~/code
if [ ! -d ~/$RemoteRepo/.git ]; then
  git clone https://github.com/tmoldwin/statistical_learning.git ~/$RemoteRepo
fi
cd ~/$RemoteRepo
git pull origin main || git pull origin master || git pull
chmod +x scripts/run_comparison_cluster.sh
python3 -m pip install --user -q -r requirements.txt numpy matplotlib seaborn scipy scikit-learn 2>/dev/null || true
"@
Invoke-MobaSsh $setupRemote

Write-Host "=== 3/6 Sync local checkpoints ===" -ForegroundColor Cyan
$localCkpts = Get-ChildItem "$RepoRoot\experiments\*_h500\rnn\model_seed*.npz" -ErrorAction SilentlyContinue
foreach ($ckpt in $localCkpts) {
    $task = $ckpt.Directory.Parent.Parent.Name
    $remoteDir = "toviah.moldwin@${RemoteHost}:$RemoteRepo/experiments/$task/rnn/"
    Invoke-MobaSsh "mkdir -p ~/$RemoteRepo/experiments/$task/rnn"
    Write-Host "  uploading $($ckpt.Name) -> $task"
    Invoke-MobaScp @($ckpt.FullName, $remoteDir)
}
if (-not $localCkpts) { Write-Host "  (no local checkpoints to sync)" }

Write-Host "=== 4/6 Submit SLURM jobs ===" -ForegroundColor Cyan
Invoke-MobaSsh "cd ~/$RemoteRepo && PARTITION=ss.cpu TIME=04:00:00 ./scripts/run_comparison_cluster.sh $Preset"

if (-not $SkipWait) {
    Write-Host "=== 5/6 Wait for jobs to finish ===" -ForegroundColor Cyan
    $waitRemote = @"
set -euo pipefail
cd ~/$RemoteRepo
echo 'Queued/running jobs:'
squeue -u \$(whoami) || true
while squeue -u \$(whoami) -h 2>/dev/null | grep -q .; do
  n=\$(squeue -u \$(whoami) -h 2>/dev/null | wc -l)
  echo \"\$(date -Iseconds)  \$n jobs remaining...\"
  sleep 60
done
echo 'All jobs finished.'
./scripts/run_comparison_cluster.sh $Preset --plot
"@
    Invoke-MobaSsh $waitRemote
} else {
    Write-Host "=== 5/6 Wait skipped (check squeue manually) ===" -ForegroundColor Yellow
}

if (-not $SkipPull) {
    Write-Host "=== 6/6 Pull results to local machine ===" -ForegroundColor Cyan
    $localCmp = Join-Path $RepoRoot "experiments\comparisons\$Preset"
    New-Item -ItemType Directory -Force -Path $localCmp | Out-Null
    $remoteCmp = "toviah.moldwin@${RemoteHost}:$RemoteRepo/experiments/comparisons/$Preset/"
    Invoke-MobaScp @("-r", "${remoteCmp}*", $localCmp)
    $remoteExps = "toviah.moldwin@${RemoteHost}:$RemoteRepo/experiments/*_h500"
    $localExps = Join-Path $RepoRoot "experiments"
    Invoke-MobaScp @("-r", $remoteExps, $localExps)
}

Write-Host "=== Done ===" -ForegroundColor Green
if ($SkipWait) {
    Write-Host "Monitor: ssh $RemoteHost 'squeue -u `$USER'"
    Write-Host "When finished: ssh ... 'cd ~/$RemoteRepo && ./scripts/run_comparison_cluster.sh $Preset --plot'"
}
