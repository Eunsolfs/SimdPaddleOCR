# Same as extract-models.ps1 but reads the model assemblies from the Tests Release output
# (models are ProjectReferences here, not NuGet packages).
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$out = "$root\bench-out\models"
$bin = "$root\test\Sdcb.SimdPaddleOCR.Tests\bin\Release\net10.0"
New-Item -ItemType Directory -Force $out | Out-Null
$map = @{ ChineseV6Tiny = "tiny"; ChineseV6Small = "small"; ChineseV6Medium = "medium"; TextLineOrientation = "" }
foreach ($m in $map.Keys) {
    $a = [System.Reflection.Assembly]::LoadFile("$bin\Sdcb.SimdPaddleOCR.Models.$m.dll")
    foreach ($n in $a.GetManifestResourceNames()) {
        $kind = ($n -split '\.')[-2]; $ext = ($n -split '\.')[-1]
        $name = if ($map[$m]) { "$($map[$m])-$kind.$ext" } else { "$kind.$ext" }
        $s = $a.GetManifestResourceStream($n)
        $f = [System.IO.File]::Create("$out\$name"); $s.CopyTo($f); $f.Close()
        "$name $($s.Length)"
    }
}
