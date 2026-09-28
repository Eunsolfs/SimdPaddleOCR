# Side-by-side per-dispatch compare of two SIMD_OCR_GPU_PROF_ALL dumps, matched by
# "<node> <shape>" part of the tag (kernel name may differ). Prints the top -n by |delta|.
param([string]$a, [string]$b, [int]$n = 25, [string]$grep = "")
function Load($f) {
    $h = [ordered]@{}
    foreach ($l in Get-Content $f) {
        if ($l -match '^\s+\[\s*\d+\]\s+([\d.]+)\s+(\S+)\s+(n\d+)\s*(.*)$') {
            $key = "$($Matches[3]) $($Matches[4])"
            $h[$key] = @([double]$Matches[1], $Matches[2])
        }
    }
    $h
}
$ha = Load $a; $hb = Load $b
$rows = foreach ($k in ($ha.Keys + $hb.Keys | Select-Object -Unique)) {
    $x = $ha[$k]; $y = $hb[$k]
    $ta = if ($x) { $x[0] } else { 0 }; $tb = if ($y) { $y[0] } else { 0 }
    [pscustomobject]@{ key = $k; a = $ta; b = $tb; d = $tb - $ta; ka = $(if ($x) { $x[1] } else { "-" }); kb = $(if ($y) { $y[1] } else { "-" }) }
}
if ($grep) { $rows = $rows | Where-Object { $_.key -match $grep -or $_.ka -match $grep -or $_.kb -match $grep } }
"sumA={0:F2} sumB={1:F2} (matched rows only)" -f ($rows | Measure-Object a -Sum).Sum, ($rows | Measure-Object b -Sum).Sum
$rows | Sort-Object { -[math]::Abs($_.d) } | Select-Object -First $n | ForEach-Object {
    "{0,7:F3} {1,7:F3} {2,8:+0.000;-0.000}  {3,-22} {4,-22} {5}" -f $_.a, $_.b, $_.d, $_.ka, $_.kb, $_.key }
