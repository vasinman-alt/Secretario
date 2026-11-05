<#
.SYNOPSIS
    Restructure to Secretario (architecture 2.1).
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path $ProjectRoot

function Op($msg) {
    if ($DryRun) { Write-Host "[DRY-RUN] $msg" -ForegroundColor Yellow }
    else         { Write-Host "[DO] $msg" -ForegroundColor Green }
}

Write-Host "=== Secretario 2.1 Refactor ===" -ForegroundColor Cyan
Write-Host "Root: $root" -ForegroundColor Cyan
Write-Host ""

# 1. Rename project root to Secretario (only if not Dry-Run)
$parent = Split-Path $root -Parent
$currentName = Split-Path $root -Leaf
$newName = "Secretario"
if ($currentName -ne $newName) {
    $newPath = Join-Path $parent $newName
    Op "Renaming '$currentName' -> '$newName'"
    if (-not $DryRun) {
        Rename-Item -Path $root -NewName $newName
        $root = $newPath
        Set-Location $root
    }
}

# 2. Clean __pycache__
Op "Cleaning __pycache__"
if (-not $DryRun) {
    Get-ChildItem -Path $root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. Delete legacy modules
$toDelete = @(
    "core\pipeline.py"
    "core\aligner"
    "core\services\whisper_service.py"
    "core\services\diarization_service.py"
    "core\services\llm_service.py"
    "llm_models"
    "models\models--Systran--faster-whisper-medium"
    "models\models--Systran--faster-whisper-small"
    "models\aligner"
)
foreach ($rel in $toDelete) {
    $p = Join-Path $root $rel
    if (Test-Path $p) {
        Op "Deleting: $rel"
        if (-not $DryRun) { Remove-Item -Path $p -Recurse -Force }
    } else {
        Write-Host "[SKIP] Not found: $rel" -ForegroundColor DarkGray
    }
}

# 4. Move LLM models to models/llm/
$llmSrc = Join-Path $root "llm_models"
$llmDst = Join-Path $root "models\llm"
if (Test-Path $llmSrc) {
    Op "Moving llm_models -> models/llm"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $llmDst -Force | Out-Null
        Get-ChildItem -Path $llmSrc -File | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $llmDst
        }
        Remove-Item $llmSrc -Recurse -Force
    }
}

# 5. Create new directory structure
$dirs = @(
    "core\vad"
    "core\speaker_id"
    "core\asr"
    "core\llm"
    "core\analysis"
    "core\services"
    "ui"
    "enrollment"
    "models\vad"
    "models\speaker_id"
    "models\asr"
    "models\llm"
    "scripts"
    "docs"
)
foreach ($d in $dirs) {
    $p = Join-Path $root $d
    Op "Creating dir: $d"
    if (-not $DryRun -and -not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
}

# 6. Move llm_service.py -> core/llm/qwen_llm.py
$oldLlm = Join-Path $root "core\services\llm_service.py"
$newLlmDir = Join-Path $root "core\llm"
$newLlm = Join-Path $newLlmDir "qwen_llm.py"
if (Test-Path $oldLlm) {
    Op "Moving core/services/llm_service.py -> core/llm/qwen_llm.py"
    if (-not $DryRun) { Move-Item -Path $oldLlm -Destination $newLlm }
} else {
    Op "Creating stub: core/llm/qwen_llm.py"
    if (-not $DryRun) {
        $stub = "Qwen3 LLM - final stage.`r`n# TODO: migrate logic from old core/services/llm_service.py"
        Set-Content -Path $newLlm -Value $stub -Encoding UTF8
    }
}

# 7. Create __init__.py in new packages
$initFiles = @(
    "core\vad\__init__.py"
    "core\speaker_id\__init__.py"
    "core\asr\__init__.py"
    "core\llm\__init__.py"
    "core\analysis\__init__.py"
    "enrollment\__init__.py"
)
foreach ($f in $initFiles) {
    $p = Join-Path $root $f
    Op "Creating: $f"
    if (-not $DryRun -and -not (Test-Path $p)) {
        New-Item -ItemType File -Path $p -Force | Out-Null
    }
}

# 8. Rename log file
$oldLog = Join-Path $root "logs\app.log"
$newLog = Join-Path $root "logs\secretario.log"
if (Test-Path $oldLog) {
    Op "Renaming logs/app.log -> logs/secretario.log"
    if (-not $DryRun) { Rename-Item -Path $oldLog -NewName "secretario.log" }
}

# 9. Move architecture doc
$archDoc = Join-Path $root "architecture_2.1.md"
$newArchDoc = Join-Path $root "docs\ARCHITECTURE.md"
if (Test-Path $archDoc) {
    Op "Moving architecture_2.1.md -> docs/ARCHITECTURE.md"
    if (-not $DryRun) { Move-Item -Path $archDoc -Destination $newArchDoc }
}

# 10. Done
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "This was a Dry-Run. Re-run without -DryRun to apply." -ForegroundColor Yellow
} else {
    Write-Host "Structure updated. Next step: code cleanup." -ForegroundColor Green
}
Write-Host "Note: aligner/ module cancelled - GigaAM produces word timestamps natively." -ForegroundColor DarkCyan
