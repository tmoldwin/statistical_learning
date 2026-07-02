# Poll cluster job status for a sweep. Run in a loop or once.
param(
    [string]$RemoteHost = "loginserver.elsc.huji.ac.il",
    [string]$RemoteRepo = "code/statistical_learning",
    [string]$LogGlob = "sixteen_word_lengths_ns_h500_*",
    [int]$IntervalSec = 120,
    [switch]$Once
)

$MobaBash = Join-Path $env:APPDATA "MobaXterm\slash\mx86_64b\bin\bash.exe"
if (-not (Test-Path $MobaBash)) { throw "MobaXterm bash not found" }

function Get-ClusterStatus {
    $cmd = @"
echo === `$(date -Iseconds) ===
RUNNING=`$(squeue -u toviah.moldwin -h | wc -l)
PENDING=`$(squeue -u toviah.moldwin -t PENDING -h 2>/dev/null | wc -l)
SEED_CKPTS=`$(find ~/$RemoteRepo/experiments -name 'model_seed*.npz' 2>/dev/null | wc -l)
DONE=`$(grep -l 'saved trained model' ~/$RemoteRepo/cluster_logs/$LogGlob/*.out 2>/dev/null | wc -l)
FAILED=`$(grep -l Traceback ~/$RemoteRepo/cluster_logs/$LogGlob/*.out 2>/dev/null | wc -l)
PROGRESS=`$(grep -h '^iter ' ~/$RemoteRepo/cluster_logs/$LogGlob/*.out 2>/dev/null | tail -1)
echo "running=$RUNNING pending=$PENDING checkpoints=$SEED_CKPTS done=$DONE failed=$FAILED"
echo "latest: $PROGRESS"
"@
    & $MobaBash -lc "ssh $RemoteHost -p 22 -C -j '$($cmd -replace "'", "'\\''")'" 2>$null | Select-String -Pattern '===|running=|latest:'
}

do {
    Get-ClusterStatus
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
} while ($true)
