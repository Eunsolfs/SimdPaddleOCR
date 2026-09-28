# Synthetic GEMM sweep: compiles bench-out/exp/cm.comp per config and times each shape with
# GpuBench --rawbench. Config = "name|tm|tn|defs..." ; prints per-shape ms and the sum.
param([string[]]$cfgs, [string]$src = "", [int]$reps = 8, [string]$shapeSet = "mix")
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (-not $src) { $src = "$root\bench-out\exp\cm.comp" }
$poc = "$root\test\Sdcb.SimdPaddleOCR.GpuBench\bin\Release\net10.0\Sdcb.SimdPaddleOCR.GpuBench.exe"
$g = if ($env:VULKAN_SDK) { "$env:VULKAN_SDK\Bin\glslc.exe" } else { (Get-Command glslc).Source }
$sets = @{
    mix = @(@(5760, 1024, 512), @(5760, 512, 1024), @(2880, 1536, 768), @(2880, 768, 1536), @(11520, 256, 512), @(11520, 512, 256),
        @(3600, 512, 1024), @(900, 1792, 896), @(14400, 256, 512), @(57600, 128, 256), @(14400, 512, 256), @(2880, 384, 768), @(5760, 192, 384))
    small = @(@(2880, 384, 768), @(2880, 768, 384), @(5760, 192, 384), @(5760, 384, 192), @(11520, 192, 96), @(3600, 192, 384), @(900, 768, 384), @(14400, 96, 192))
    wide = @(@(57600, 128, 256), @(57600, 256, 128), @(14400, 128, 256), @(14400, 256, 256), @(230400, 64, 128), @(57600, 64, 128))
}
$shapes = $sets[$shapeSet]
"{0,-14} {1}" -f "cfg", (($shapes | ForEach-Object { "{0}x{1}x{2}" -f $_[0], $_[1], $_[2] }) -join " ")
foreach ($c in $cfgs) {
    $parts = $c -split "\|"
    $name = $parts[0]; $tm = [int]$parts[1]; $tn = [int]$parts[2]; $defs = @($parts | Select-Object -Skip 3 | Where-Object { $_ -and $_ -notlike "src=*" })
    $csrc = ($parts | Where-Object { $_ -like "src=*" } | ForEach-Object { $_.Substring(4) }) ?? $src
    if (-not $csrc) { $csrc = $src }
    $spv = "$root\bench-out\exp\sw-$name.spv"
    & $g -O --target-env=vulkan1.1 @defs $csrc -o $spv
    if (!$?) { "$name compile failed"; continue }
    $tot = 0.0; $cells = @()
    foreach ($sh in $shapes) {
        $gx = [math]::Ceiling($sh[0] / $tm); $gy = [math]::Ceiling($sh[2] / $tn)
        $r = & $poc --rawbench $spv 6 $gx $gy $reps 64 $sh[0] $sh[2] $sh[1] 0 2>&1 | Select-String "best=([\d.]+)"
        if (-not $r) { $cells += "ERR"; continue }
        $ms = [double]$r.Matches[0].Groups[1].Value; $tot += $ms
        $cells += "{0:F3}" -f $ms
    }
    "{0,-14} {1}  sum={2:F3}" -f $name, ($cells -join " "), $tot
}
