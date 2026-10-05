param(
    [Parameter(Mandatory = $true)]
    [string]$VideoDir,
    [Parameter(Mandatory = $true)]
    [string]$FramesDir,
    [double]$SampleFps = 1.0,
    [switch]$RecurseVideos,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$Sam3Root = "C:\My_Project\SAM_3"
$Extractor = ".\guideline_line\extract_video_frames.py"

if (-not (Test-Path -LiteralPath $Sam3Root)) {
    throw "Missing SAM_3 root: $Sam3Root"
}

Push-Location $Sam3Root
try {
    $extractArgs = @(
        $Extractor,
        "--video-dir", ([System.IO.Path]::GetFullPath($VideoDir)),
        "--output-root", ([System.IO.Path]::GetFullPath($FramesDir)),
        "--backend", "ffmpeg",
        "--ffmpeg-hwaccel", "cuda",
        "--mode", "sample_fps",
        "--sample-fps", $SampleFps,
        "--no-dedup",
        "--dedup-threshold", "0.014",
        "--dedup-center-crop-ratio", "0.82",
        "--dedup-resize-width", "224",
        "--dedup-blur-kernel", "5",
        "--min-gap-seconds", "0.0",
        "--image-ext", ".jpg",
        "--jpeg-quality", "95",
        "--workers", $Workers
    )
    if ($RecurseVideos) {
        $extractArgs += "--recursive"
    }
    & .\.sam3_py312\Scripts\python.exe @extractArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
