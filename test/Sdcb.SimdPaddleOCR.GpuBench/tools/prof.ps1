param([string]$tag = "x", [string]$kind = "det", [string]$model = "medium", [int]$h = 960, [int]$w = 960, [string]$grep = "", [string[]]$defs = @())
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$g = if ($env:VULKAN_SDK) { "$env:VULKAN_SDK\Bin\glslc.exe" } else { (Get-Command glslc).Source }
$d = "$root\src\Sdcb.SimdPaddleOCR\Backends\Vulkan\Shaders"
$sg = "$d\conv1x1_cm_sg32.comp"
& "$root\src\Sdcb.SimdPaddleOCR\Backends\Vulkan\tools\build-shaders.bat" | Out-Null; if (!$?) { exit 1 }
if ($defs.Count -gt 0) { & $g -O --target-env=vulkan1.1 @defs $sg -o "$d\conv1x1_cm_sg32.spv"; if (!$?) { exit 1 } }
dotnet build "$root\test\Sdcb.SimdPaddleOCR.GpuBench" -c Release 2>&1 | Select-String " error " | Select-Object -First 5
$env:SIMD_OCR_GPU_PROF = "1"; $env:SIMD_OCR_GPU_PROF_ALL = "1"
$exe = "$root\test\Sdcb.SimdPaddleOCR.GpuBench\bin\Release\net10.0\Sdcb.SimdPaddleOCR.GpuBench.exe"
$out = "$root\bench-out\prof-$kind-$model-$tag.txt"
if ($kind -eq "det") { & $exe --detprof "$root\bench-out\models\$model-det.onnx" $h $w 10 2>&1 | Out-File -Encoding utf8 $out }
else { & $exe --recprof "$root\bench-out\models\$model-rec.onnx" $h $w 10 2>&1 | Out-File -Encoding utf8 $out }
Remove-Item env:SIMD_OCR_GPU_PROF, env:SIMD_OCR_GPU_PROF_ALL -ErrorAction SilentlyContinue
Get-Content $out | Where-Object { $_ -notmatch '^\s+\[' } | Select-Object -First 8
if ($grep) { Get-Content $out | Select-String $grep | ForEach-Object Line }
