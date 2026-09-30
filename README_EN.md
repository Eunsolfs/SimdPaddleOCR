# Sdcb.SimdPaddleOCR [![test](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml/badge.svg)](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml) [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-579060605-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/q/bPw5jAK4qk)

[中文](README.md) | **English**

Pure C# PP-OCRv6 inference library: cross-platform hand-written kernels and GEMM, very high performance, low memory requirements, and very high accuracy.
It ships a managed ONNX interpreter and does not depend on Paddle Inference, ONNX Runtime, or OpenCV native libraries.

The core API accepts interleaved pixels (BGR24 by default; RGB24 / BGRA32 / RGBA32 are also first-class). It does not decode images, so ImageSharp, SkiaSharp, or OpenCvSharp are not required.

## Quick start

Install the core package and the tiny models (tiny transitively references CLS and `ModelProvider`):

```powershell
dotnet add package Sdcb.SimdPaddleOCR
dotnet add package Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny
dotnet add package SixLabors.ImageSharp --version 3.1.11
```

Models load from embedded assembly resources. They are not extracted or written to temp files. `stride = 0` means tightly packed (`width *` bytes-per-pixel). The default is `ImagePixelFormat.Bgr24`; other layouts are swizzled in-place during resize / crop and are not converted into an intermediate BGR image.

### ImageSharp 3 (recommended)

```csharp
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
using Image<Rgba32> image = await Image.LoadAsync<Rgba32>(CreateDecoderOptions(), "sample.jpg");
Rgba32[]? packedCopy = null;
if (!image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
{
    packedCopy = new Rgba32[checked(image.Width * image.Height)];
    image.CopyPixelDataTo(packedCopy);
    memory = packedCopy;
}

PaddleOcrResult result = ocr.Run(MemoryMarshal.AsBytes(memory.Span), image.Width, image.Height,
    format: ImagePixelFormat.Rgba32);
Console.WriteLine(result.Text);

// Clone Default so PNG/JPEG decoders stay registered. Do not set this on Configuration.Default.
// Default allocator splits pixels into 4MB chunks; large images then fail DangerousTryGetSinglePixelMemory.
static DecoderOptions CreateDecoderOptions()
{
    Configuration configuration = Configuration.Default.Clone();
    configuration.PreferContiguousImageBuffers = true;
    return new DecoderOptions { Configuration = configuration };
}
```

ImageSharp's default allocator splits pixels into 4MB chunks, so `DangerousTryGetSinglePixelMemory` fails on large images. Clone a `Configuration`, set `PreferContiguousImageBuffers` for a zero-copy buffer when possible, and fall back to `CopyPixelDataTo`. Do not mutate `Configuration.Default`.

The next three samples wrap a lock / native pointer as `ReadOnlySpan<byte>` before `Run`. Do not Unlock / Dispose the source until `Run` returns.

### SkiaSharp

```csharp
using SkiaSharp;

SKBitmap bitmap = SKBitmap.Decode("sample.jpg")
    ?? throw new InvalidDataException("Failed to read image");
if (bitmap.ColorType != SKColorType.Bgra8888)
    bitmap = bitmap.Copy(SKColorType.Bgra8888)
        ?? throw new InvalidDataException("Failed to convert to BGRA");
int stride = bitmap.RowBytes;
unsafe
{
    PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)bitmap.GetPixels(), stride * bitmap.Height),
        bitmap.Width, bitmap.Height, stride, ImagePixelFormat.Bgra32);
}
```

### OpenCvSharp5

```csharp
using OpenCvSharp;

using Mat image = Cv2.ImRead("sample.jpg", ImreadModes.Color);
if (image.Empty()) throw new InvalidDataException("Failed to read image");
int stride = (int)image.Step();
unsafe
{
    PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)image.Data, stride * image.Height),
        image.Width, image.Height, stride, ImagePixelFormat.Bgr24);
}
```

### Bitmap

```csharp
using System.Drawing;
using System.Drawing.Imaging;

using Bitmap bitmap = new("sample.jpg");
Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
try
{
    unsafe
    {
        PaddleOcrResult result = ocr.Run(new ReadOnlySpan<byte>((byte*)data.Scan0, data.Stride * bitmap.Height),
            bitmap.Width, bitmap.Height, data.Stride, ImagePixelFormat.Bgra32);
    }
}
finally
{
    bitmap.UnlockBits(data);
}
```

## NuGet packages

| NuGet package                                   | Version                                                                                                                                                                    | Description                                                                                           |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `Sdcb.SimdPaddleOCR`                            | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR)                                                       | Pure-managed inference core (`net10.0;netstandard2.0`)                                                |
| `Sdcb.SimdPaddleOCR.ModelProvider`              | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.ModelProvider.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.ModelProvider)                           | Model contracts (`IPaddleOcrModelProvider` / `PaddleOcrModelBundle`), usually referenced transitively |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny`       | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny)             | PP-OCRv6 tiny DET+REC+dictionary; `ChineseV6TinyModels.Default` includes CLS                          |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Small`      | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Small.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Small)           | PP-OCRv6 small; `ChineseV6SmallModels.Default`                                                        |
| `Sdcb.SimdPaddleOCR.Models.ChineseV6Medium`     | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.ChineseV6Medium)         | PP-OCRv6 medium; `ChineseV6MediumModels.Default`                                                      |
| `Sdcb.SimdPaddleOCR.Models.TextLineOrientation` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.SimdPaddleOCR.Models.TextLineOrientation.svg)](https://www.nuget.org/packages/Sdcb.SimdPaddleOCR.Models.TextLineOrientation) | PP-LCNet text-line orientation CLS, transitively referenced by the three Chinese model packages       |

Each `IPaddleOcrModelProvider` exposes `Name`, `Kind`, `Format`, language and version metadata, plus `OpenRead()` / `OpenReadAsync()`. A full OCR set is a `PaddleOcrModelBundle` (DET, REC, dictionary, and optional CLS). The current language code is `zh`. Individual models can also be consumed by other inference implementations, for example `ChineseV6TinyModel.Detection.OpenReadAsync()`. `Model`, `PaddleOcrDetector`, `PaddleOcrClassifier`, `PaddleOcrRecognizer`, and `PaddleOcrAll` all accept Stream load entry points; after parsing they do not keep the full raw ONNX bytes.

## Local models

The core does not download models. To use local DET, CLS, REC, and dictionary files:

```csharp
using Sdcb.SimdPaddleOCR;

using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(
    detectionPath: "models/det.onnx",
    classificationPath: "models/cls.onnx",
    recognitionPath: "models/rec.onnx",
    dictionaryPath: "models/ppocr_keys.txt");
```

Do not mix the two parallelism knobs in `PaddleOcrOptions`: `DetIntraOpThreads` is in-graph convolution threads for detection
(one session, default cap 8); `LineWorkerCount` is the number of CLS/REC worker lanes
(one session per lane, an upper bound, actually `min(requested, ProcessorCount)`; `0` means
`min(ProcessorCount, 4)`). Detection thresholds, min box side, orientation classification,
dynamic recognition width, and session cache limits live in the same options object.

## Examples

All four samples share `examples/sample.jpg`. Each sample decodes the image and converts it to BGR:

- `examples/ImageSharp.AspNetCore`: ASP.NET Core + ImageSharp 3, upload UI and `POST /api/ocr` JSON API.
- `examples/SkiaSharp.Avalonia`: Avalonia desktop sample, SkiaSharp decode.
- `examples/OpenCvSharp5.Wpf`: WPF sample, OpenCvSharp5 decode.
- `examples/SystemDrawing.WinForms`: dual-target WinForms sample for .NET 10 Windows / .NET Framework 4.8, using `Bitmap`/`LockBits`; install the .NET Framework 4.8 Developer Pack before running `net48`, and set the project platform to x64.

```powershell
dotnet run --project examples/ImageSharp.AspNetCore
dotnet run --project examples/OpenCvSharp5.Wpf -- path/to/image.jpg
dotnet run --project examples/SkiaSharp.Avalonia -- path/to/image.jpg
dotnet run --project examples/SystemDrawing.WinForms --framework net10.0-windows
```

Open the web sample in a browser to upload; the API is `POST /api/ocr` (`multipart/form-data` fields `file`, `model`), docs at `/scalar`.

`Sdcb.SimdPaddleOCR` also ships 4 LINQPad scripts (`imagesharp` / `skiasharp` / `opencvsharp5` / `bitmap`): download a sample image, run the tiny model, print text. `bitmap` runs on both LINQPad 5 / .NET Framework 4.8 and .NET 10. The package is tagged `linqpad-samples`, so the free edition can use them.

## FAQ

### Why is OCR recognition much slower while debugging?

The debugger can disable JIT optimization when modules load. This prevents the runtime from optimizing OCR's compute-intensive code, making recognition significantly slower.

Clear this option, then restart the debugging session:

- Visual Studio: `Tools > Options > Debugging > General` > clear `Suppress JIT optimization on module load (Managed only)`.
- Rider: `Build, Execution, Deployment > Debugger > JIT` > clear `Disable JIT optimization on module load`.
- VS Code: open `Settings (JSON)` and add `"csharp.debug.suppressJITOptimizations": false`.

### Why is Native AOT much slower than JIT?

When publishing Native AOT on x64, the executable project **must** set:

```xml
<IlcInstructionSet>avx2</IlcInstructionSet>
```

Without it, ILC targets the SSE2 / 128-bit `Vector<T>` baseline, `Avx2.IsSupported` is folded to `false`, the AVX2 kernels are stripped, and inference is much slower. Do not set this on CPUs without AVX2. ARM64 AOT already includes NEON / `AdvSimd` in the baseline, so you usually do not need `IlcInstructionSet`.

### ImageSharp throws "pixels are not a single contiguous buffer" or `DangerousTryGetSinglePixelMemory` fails?

ImageSharp's default allocator splits pixels into 4MB chunks, so large images do not get one packed buffer. Follow the ImageSharp 3 sample above: clone `Configuration`, set `PreferContiguousImageBuffers`, and fall back to `CopyPixelDataTo`. Do not mutate `Configuration.Default` or PNG/JPEG decoders disappear. The full pattern is in `examples/ImageSharp.AspNetCore/Program.cs`.

## Support

|                     | Notes                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Target frameworks   | Core `net10.0;netstandard2.0`; `ModelProvider` and all model packages are `netstandard2.0`                                                                               |
| Recommended runtime | .NET 10: full x86 SIMD and NativeAOT (`IsAotCompatible`)                                                                                                                 |
| Compatible runtime  | `netstandard2.0` can run on .NET Framework 4.8 and similar; AVX / AVX-512 / VNNI sources are excluded at compile time, falling back to `System.Numerics.Vector` / scalar |
| CI architectures    | Windows x64 / x86 / ARM64, Linux x64 / ARM64, macOS x64 / ARM64                                                                                                          |
| SIMD                | .NET 10 probes AVX → AVX2 → AVX-512 / VNNI at runtime; Vector/scalar when those ISAs are missing or on ARM                                                               |
| Input               | Interleaved pixels (BGR24 by default; RGB24 / BGRA32 / RGBA32 also accepted); no image path, file, or image-library API                                                  |
| Device              | CPU by default; `net10.0` also has an optional Vulkan backend (`OcrBackend`) loading the system loader directly: `vulkan-1.dll` on Windows, `libvulkan.so.1` on Linux, `libvulkan.so` on Android; `netstandard2.0` is CPU only |
| Android             | Currently run through the dev host `test/Sdcb.SimdPaddleOCR.AndroidBench` (`net10.0-android`, references the `net10.0` library, driven over adb) on a Snapdragon 8 Gen 3, CPU and Vulkan; see [`docs/vulkan-8gen3.md`](docs/vulkan-8gen3.md). Desktop Vulkan routing and shaders are unchanged |
| NativeAOT           | Keep the core assembly and the model assemblies you use when publishing trimmed                                                                                          |

## License and third-party components

Source code and documentation written in this repository are released under [Apache License 2.0](LICENSE).
Apache-2.0 includes an explicit patent grant, which is a better fit for a public library and NuGet packages.

Model assets and third-party code are not relicensed by this project:

- PP-OCRv6 DET/REC, TextLineOrientation CLS, and dictionaries come from the PaddleOCR ecosystem and are marked
  Apache-2.0 in their source materials; keep origin and license notices when publishing model packages.
- Sample dependencies follow their upstream licenses; in particular ImageSharp 3.x uses the Six Labors Split License,
  not a plain MIT license.

Full third-party attribution is in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). PaddleOCR, PP-OCR, and related names belong to their
respective owners. This project is not official and does not imply endorsement.

## Performance

**1.4.2** vs **1.3.0**: graph-level NHWC now covers ns2 / x64 scalar / net10 AdvSIMD, and preprocess writes NHWC directly.
**Memory dropped sharply**: tiny-4w working-set peak is about **300 MB** lower (win-x64 817→**515 MB**, linux-arm64 840→**572 MB**); Δ WS fell from ~400 MB to ~100–160 MB.
CI tiny is **767/1032**, CER **2.36%** on the full 100 (cls 1020/1020; 1.3 skip-first was 757/1022, 3.53% — different ruler, not a tiny gain). Local exact_lines stay 742 / 950 / 1004; CER is **2.78% → 2.37%**, **0.60% → 0.41%**, **0.67% → 0.14%** (left 4:1 CLS, inverted long lines rotate before REC).

Median wall time per image on GitHub-hosted runners, PP-OCRv6 tiny, first image excluded as warmup:

| Path                                     |     1.3 |    1.4.2 |       vs 1.3 |          WS peak |            Δ WS |
| ---------------------------------------- | ------: | -------: | -----------: | ---------------: | --------------: |
| linux-arm64 N2 `tiny-4w` (net10 AdvSIMD) |     241 |  **180** |    **0.75×** | 840 → **572 MB** | 418 → **162 MB** |
| linux-arm64 `tiny-4w-ns2`                |     374 |  **295** |    **0.79×** |                  |                 |
| linux-arm64 `tiny-4w-scalar`             |     984 |  **856** |    **0.87×** |                  |                 |
| win-x64 7763 `tiny-4w` (AVX2)            |     167 |     ~184 | flat (noise) | 817 → **515 MB** | 398 → **107 MB** |
| win-x64 7763 `tiny-4w-ns2`               | **343** |  **228** |    **0.66×** |                  |                 |
| win-x64 7763 `tiny-4w-noavx`             |     481 |  **380** |    **0.79×** |                  |                 |
| win-x64 7763 `tiny-4w-scalar`            |    1368 | **1220** |    **0.89×** |                  |                 |

Local Ryzen 7 5800X, 4 workers, repo `dataset/` 100 images (n=99), this library mean (1.3 NuGet → 1.4.2): tiny **86.0 → 63.1 ms** (0.73×), small **222 → 200 ms** (0.90×), medium **628 → 585 ms** (0.93×). Same machine ns2: tiny **203 → 96.5 ms** (0.48×), small **432 → 303 ms** (0.70×), medium **1606 → 874 ms** (0.54×).
Same-machine C engine and per-ISA ratios: [`docs/perf.md`](docs/perf.md).

## Reproducing performance

The [GitHub Actions `test` workflow](https://github.com/sdcb/SimdPaddleOCR/actions/workflows/test.yml)
runs unit tests and benches tiny / small / medium on Windows / Linux / macOS across architectures
(including disabling AVX-512 / AVX2 / AVX / all hardware acceleration, and the `netstandard2.0` build). Summary output goes to the job summary and the `perf-report` artifact.

## WeChat group

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png?0922)

If the WeChat QR code has expired, join the QQ group [C#/.NET Computer Vision 579060605](https://qm.qq.com/q/bPw5jAK4qk).
