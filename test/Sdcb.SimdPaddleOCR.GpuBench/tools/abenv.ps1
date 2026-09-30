# Alternating A/B of one binary with an env var unset (A) vs set (B). Prints median per run.
param([string]$var, [string]$val, [string[]]$models = @("tiny", "small", "medium"), [int]$rounds = 3, [string]$tag = "abenv")
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$exe = "$root\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0\Sdcb.SimdPaddleOCR.Tests.exe"
$res = [ordered]@{}
for ($r = 1; $r -le $rounds; $r++) {
    foreach ($m in $models) {
        foreach ($s in "A", "B") {
            if ($s -eq "B") { Set-Item "env:$var" $val } else { Remove-Item "env:$var" -ErrorAction SilentlyContinue }
            $out = "$root\bench-out\$tag-$s-$m-r$r.json"
            $o = & $exe --workers 4 --model $m --engine vulkan --benchmark-kind simd --warmup 1 --input "$root\dataset" --out $out 2>&1
            $med = [regex]::Match(($o | Select-String "total_ms" | Select-Object -Last 1).Line, 'median=([\d.]+)').Groups[1].Value
            $acc = [regex]::Match(($o | Select-String "accuracy" | Select-Object -Last 1).Line, 'exact_lines=(\d+)/\d+.*CER=([\d.]+%)').Groups
            $k = "$m $s"
            if (-not $res[$k]) { $res[$k] = @() }
            $res[$k] += "$med($($acc[1].Value),$($acc[2].Value))"
        }
    }
}
Remove-Item "env:$var" -ErrorAction SilentlyContinue
$res.GetEnumerator() | ForEach-Object { "{0,-10} {1}" -f $_.Key, ($_.Value -join " ") }
