# Pure-GPU profile on the phone over the GpuBench gprof.ps1 case list
# (per-dispatch timestamps, min over reps). Logs land in bench-out/android/gp-<tag>-*.log.
param([string]$tag = "x", [string[]]$cases = @(
    "det medium 960 960", "det medium 640 960", "rec medium 8 480", "rec medium 1 320",
    "det small 960 960", "rec small 8 480", "det tiny 960 960", "rec tiny 8 480"),
    [int]$reps = 10, [switch]$all, [string]$Env = "")
$here = $PSScriptRoot
$out = (Resolve-Path "$here\..\..").Path + "\bench-out\android"
$envs = "SIMD_OCR_GPU_PROF=1" + $(if ($all) { ";SIMD_OCR_GPU_PROF_ALL=1" }) + $(if ($Env) { ";$Env" })
foreach ($c in $cases) {
    $k, $m, $a, $b = $c -split " "
    $mode = if ($k -eq "det") { "--detprof" } else { "--recprof" }
    $id = "gp-$tag-$k-$m-$a-$b"
    & "$here\run.ps1" -Bench "$mode $m $a $b $reps" -Env $envs -Run $id | Out-Null
    $log = "$out\$id.log"
    $t = (Select-String -Path $log -Pattern "GPU profile \((\d+) dispatches, ([\d.]+) ms").Matches
    $w = (Select-String -Path $log -Pattern "best wall ([\d.]+)").Matches
    $th = (Select-String -Path $log -Pattern "thermal=(\d+)").Matches
    "{0,-22} gpu={1,8} ms  disp={2,4}  wall={3}  thermal={4}" -f $c, $t[0].Groups[2].Value, $t[0].Groups[1].Value,
        $(if ($w) { $w[0].Groups[1].Value } else { "?" }), $(if ($th) { $th[0].Groups[1].Value } else { "?" })
}
