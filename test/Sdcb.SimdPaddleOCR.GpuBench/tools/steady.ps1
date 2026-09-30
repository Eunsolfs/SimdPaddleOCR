param([string[]]$files)
foreach ($f in $files) {
    $r = (Get-Content $f -Raw | ConvertFrom-Json).rows | Where-Object { -not $_.warmup }
    $all = @($r | ForEach-Object { [double]$_.total_ms } | Sort-Object)
    $tail = @($r | Select-Object -Last 50 | ForEach-Object { [double]$_.total_ms } | Sort-Object)
    "{0,-45} all-med {1,6:F1}  last50-med {2,6:F1}  last50-mean {3,6:F1}" -f (Split-Path $f -Leaf), $all[[int]($all.Count / 2)], $tail[25], ($tail | Measure-Object -Average).Average
}
