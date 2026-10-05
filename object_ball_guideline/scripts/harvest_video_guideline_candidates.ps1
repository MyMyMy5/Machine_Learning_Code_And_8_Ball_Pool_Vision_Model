param(
    [string]$VideoDir = "C:\My_Project\SAM_3\guideline_line\video_drop_model_harvest",
    [string]$FramesDir = "C:\My_Project\SAM_3\guideline_line\video_frames_model_harvest",
    [string]$InferenceDir = "C:\My_Project\SAM_3\guideline_line\video_infer_model_harvest",
    [string]$DetectedDir = "C:\My_Project\SAM_3\guideline_line\video_infer_harvest_2",
    [double]$SampleFps = 1.0,
    [int]$ExtractWorkers = 4,
    [double]$DetectionMinScore = -999.0,
    [int]$DetectionMinMaskPixels = 1,
    [int]$CropBatchSize = 32,
    [int]$NumWorkers = 4,
    [switch]$RecurseVideos,
    [switch]$SkipExistingInference
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ExtractScript = "C:\My_Project\SAM_3\guideline_line\extract_video_frames.py"
$HarvestScript = Join-Path $RepoRoot "scripts\run_supervised_best_harvest.ps1"
$FilterScript = Join-Path $RepoRoot "tools\filter_guideline_harvest.py"

if (-not (Test-Path -LiteralPath $ExtractScript)) {
    throw "Missing extractor: $ExtractScript"
}
if (-not (Test-Path -LiteralPath $HarvestScript)) {
    throw "Missing harvest inference script: $HarvestScript"
}
if (-not (Test-Path -LiteralPath $FilterScript)) {
    throw "Missing filter script: $FilterScript"
}

$VideoDir = [System.IO.Path]::GetFullPath($VideoDir)
$FramesDir = [System.IO.Path]::GetFullPath($FramesDir)
$InferenceDir = [System.IO.Path]::GetFullPath($InferenceDir)
$DetectedDir = [System.IO.Path]::GetFullPath($DetectedDir)

$DetectedImagesDir = Join-Path $DetectedDir "Predicted"
$DetectedMasksDir = Join-Path $DetectedDir "Masks_Predicted"
$NotDetectedImagesDir = Join-Path $DetectedDir "No_Prediction"

New-Item -ItemType Directory -Force -Path $VideoDir | Out-Null
New-Item -ItemType Directory -Force -Path $FramesDir | Out-Null
New-Item -ItemType Directory -Force -Path $DetectedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DetectedImagesDir | Out-Null
New-Item -ItemType Directory -Force -Path $DetectedMasksDir | Out-Null
New-Item -ItemType Directory -Force -Path $NotDetectedImagesDir | Out-Null

Write-Host "Video drop folder: $VideoDir"
Write-Host "Extracted frames:  $FramesDir"
Write-Host "Legacy inference:  $InferenceDir"
Write-Host "Detected harvest:  $DetectedDir"
Write-Host "Extraction backend: ffmpeg + cuda decode fallback"

Write-Host "Extracting frames..."
Push-Location "C:\My_Project\SAM_3"
try {
    $extractArgs = @(
        ".\guideline_line\extract_video_frames.py",
        "--video-dir", $VideoDir,
        "--output-root", $FramesDir,
        "--mode", "sample_fps",
        "--sample-fps", $SampleFps,
        "--workers", $ExtractWorkers,
        "--no-dedup",
        "--dedup-threshold", "0.014",
        "--dedup-center-crop-ratio", "0.82",
        "--dedup-resize-width", "224",
        "--dedup-blur-kernel", "5",
        "--min-gap-seconds", "0.0",
        "--image-ext", ".jpg",
        "--jpeg-quality", "95"
    )
    if ($RecurseVideos) {
        $extractArgs += "--recursive"
    }
    & .\.sam3_py312\Scripts\python.exe @extractArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Frame extraction failed."
    }
}
finally {
    Pop-Location
}

$extractedFrames = Get-ChildItem -LiteralPath $FramesDir -Recurse -File | Where-Object {
    $_.Extension -in @(".png", ".jpg", ".jpeg", ".bmp", ".webp")
}
if (-not $extractedFrames) {
    throw "No extracted frames were found under $FramesDir"
}

Write-Host "Backfilling any legacy per-frame inference outputs..."
$filterArgs = @(
    $FilterScript,
    "--frames-dir", $FramesDir,
    "--inference-dir", $InferenceDir,
    "--output-root", $DetectedDir,
    "--detection-min-score", $DetectionMinScore,
    "--detection-min-mask-pixels", $DetectionMinMaskPixels,
    "--recurse-frames"
)
& python @filterArgs
if ($LASTEXITCODE -ne 0) {
    throw "Harvest filtering failed."
}

Write-Host "Running primary-model harvest over extracted frames..."
$HarvestArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $HarvestScript,
    "--frames-dir", $FramesDir,
    "--output-root", $DetectedDir,
    "--recurse",
    "--detection-min-score", $DetectionMinScore,
    "--detection-min-mask-pixels", $DetectionMinMaskPixels,
    "--crop-batch-size", $CropBatchSize,
    "--num-workers", $NumWorkers
)
if ($SkipExistingInference) {
    $HarvestArgs += "--skip-existing"
}
& powershell @HarvestArgs
if ($LASTEXITCODE -ne 0) {
    throw "Harvest inference failed."
}

$summaryPath = Join-Path $DetectedDir "harvest_summary.json"

Write-Host "Done."
Write-Host "Predicted overlays:  $DetectedImagesDir"
Write-Host "Predicted masks:     $DetectedMasksDir"
Write-Host "No-prediction frames:$NotDetectedImagesDir"
Write-Host "Summary: $summaryPath"
