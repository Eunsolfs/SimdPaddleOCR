param([string]$a, [string]$b, [string[]]$fields = @("hash", "detected"))
$ra = @{}; (Get-Content $a -Raw | ConvertFrom-Json).rows | ForEach-Object { $ra[$_.file] = $_ }
$rb = @{}; (Get-Content $b -Raw | ConvertFrom-Json).rows | ForEach-Object { $rb[$_.file] = $_ }
$same = 0; $diff = @()
foreach ($f in $ra.Keys) {
    if (-not $rb.ContainsKey($f)) { $diff += "$f missing"; continue }
    $ok = $true
    foreach ($fld in $fields) {
        $x = $ra[$f].$fld | ConvertTo-Json -Compress -Depth 5
        $y = $rb[$f].$fld | ConvertTo-Json -Compress -Depth 5
        if ($x -ne $y) { $ok = $false; $diff += "$f $fld" }
    }
    if ($ok) { $same++ }
}
"{0}/{1} identical ({2})" -f $same, $ra.Count, ($fields -join ",")
$diff | Select-Object -First 10
