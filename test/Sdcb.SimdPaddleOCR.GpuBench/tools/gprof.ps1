# Pure-GPU profile over a fixed case list (no shader rebuild). -poc picks the GpuBench build
# (default: main tree); -tag names the output files bench-out/gp-<tag>-<case>.txt.
param([string]$tag = "x", [string]$poc = "", [string[]]$cases = @(
    "det medium 960 960", "det medium 640 960", "rec medium 8 480", "rec medium 1 320",
    "det small 960 960", "rec small 8 480", "det tiny 960 960", "rec tiny 8 480"), [int]$reps = 10, [switch]$all, [string]$grep = "")
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (-not $poc) { $poc = "$root\test\Sdcb.SimdPaddleOCR.GpuBench\bin\Release\net10.0\Sdcb.SimdPaddleOCR.GpuBench.exe" }
$env:SIMD_OCR_GPU_PROF = "1"; if ($all) { $env:SIMD_OCR_GPU_PROF_ALL = "1" }
foreach ($c in $cases) {
    $k, $m, $a, $b = $c -split " "
    $mode = if ($k -eq "det") { "--detprof" } else { "--recprof" }
    $out = "$root\bench-out\gp-$tag-$k-$m-$a-$b.txt"
    & $poc $mode "$root\bench-out\models\$m-$k.onnx" $a $b $reps 2>&1 | Out-File -Encoding utf8 $out
    $t = (Select-String -Path $out -Pattern "GPU profile \((\d+) dispatches, ([\d.]+) ms").Matches
    $w = (Select-String -Path $out -Pattern "best wall ([\d.]+)").Matches
    $ks = ""
    if ($grep) {
        $sum = 0.0
        foreach ($l in Get-Content $out) { if ($l -match "^\s+(\S+)\s+([\d.]+) ms$") { $kn = $Matches[1]; $kt = [double]$Matches[2]; if ($kn -match $grep) { $sum += $kt } } }
        $ks = " {0}={1:F2}" -f $grep, $sum
    }
    "{0,-22} gpu={1,8} ms  disp={2,4}  wall={3}{4}" -f $c, $t[0].Groups[2].Value, $t[0].Groups[1].Value, ($(if ($w) { $w[0].Groups[1].Value } else { "?" })), $ks
}
Remove-Item env:SIMD_OCR_GPU_PROF, env:SIMD_OCR_GPU_PROF_ALL -ErrorAction SilentlyContinue
