param([string[]]$models = @("tiny", "small", "medium"), [int]$rounds = 1, [string]$tag = "ab", [string]$engine = "vulkan", [string[]]$sides = @("base", "new"), [int]$count = 0, [string]$wt = "wt")
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$exe = @{
    base = "$root\bench-out\$wt\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"
    new  = "$root\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"
}
$files = @()
for ($r = 1; $r -le $rounds; $r++) {
    foreach ($m in $models) {
        foreach ($s in $sides) {
            $out = "$root\bench-out\$tag-$s-$m-$engine-r$r.json"
            $a = @("--workers", "4", "--model", $m, "--engine", $engine, "--benchmark-kind", "simd", "--warmup", "1", "--input", "$root\dataset", "--out", $out)
            if ($count -gt 0) { $a += @("--count", "$count") }
            & $exe[$s] @a 2>&1 | Select-String "median|exact|CER" | Select-Object -Last 3 | ForEach-Object { "[$s $m r$r] $($_.Line.Trim())" }
            $files += $out
        }
    }
}
& $exe["new"] --summarize @files --input "$root\dataset"
