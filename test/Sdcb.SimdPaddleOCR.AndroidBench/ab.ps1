# End-to-end A/B on the phone: for each model, backends alternate round by
# round (cpu, vulkan, cpu, vulkan, ...) with a cool-down between runs. Prints
# median ms (all / last 50 images), thermal status and accuracy per run.
#   .\ab.ps1 -tag final -models tiny,small,medium -rounds 3
param([string]$tag = "ab", [string[]]$models = @("tiny", "small", "medium"),
    [string[]]$backends = @("cpu", "vulkan"), [int]$rounds = 3, [int]$cool = 60, [string]$Env = "")
$here = $PSScriptRoot
$out = (Resolve-Path "$here\..\..").Path + "\bench-out\android"
foreach ($m in $models) {
    for ($r = 1; $r -le $rounds; $r++) {
        foreach ($b in $backends) {
            $id = "$tag-$m-$b-r$r"
            Start-Sleep -Seconds $cool
            & "$here\run.ps1" -Bench "--e2e --model $m --backend $b --out $id.json" -Env $Env -Run $id -TimeoutSec 3600 | Out-Null
            $log = "$out\$id.log"
            $st = (Select-String -Path $log -Pattern "steady: all-med ([\d.]+) last50-med ([\d.]+) ws-peak (\d+)MB thermal=(-?\d+)").Matches
            $ac = (Select-String -Path $log -Pattern "exact_lines=(\d+)/\d+ .*CER=([\d.]+)%").Matches
            $gpu = (Select-String -Path $log -Pattern "usesGpu=(\w+)").Matches
            if ($st) {
                "{0,-7} {1,-7} r{2}  med {3,7}  last50 {4,7}  ws {5,5}MB  thermal {6}  lines {7}  CER {8}%  gpu={9}" -f $m, $b, $r,
                    $st[0].Groups[1].Value, $st[0].Groups[2].Value, $st[0].Groups[3].Value, $st[0].Groups[4].Value,
                    $ac[0].Groups[1].Value, $ac[0].Groups[2].Value, $gpu[0].Groups[1].Value
            } else { "{0,-7} {1,-7} r{2}  FAILED (see $log)" -f $m, $b, $r }
        }
    }
}
