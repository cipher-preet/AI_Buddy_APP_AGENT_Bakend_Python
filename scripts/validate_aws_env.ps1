param(
    [string]$Path = ".env.aws"
)

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Error "$Path was not found."
    exit 1
}

$values = @{}
Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) {
        return
    }
    $parts = $line -split "=", 2
    if ($parts.Length -eq 2) {
        $values[$parts[0].Trim()] = $parts[1].Trim()
    }
}

$required = @(
    "REDIS_PASSWORD",
    "REDIS_URL",
    "QUEUE_API_SERVICE_TOKEN",
    "QUEUE_API_HMAC_SECRET",
    "MONGODB_URL",
    "MONGODB_DATABASE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_AUDIO_BUCKET",
    "SARVAM_API_KEY",
    "OPENAI_API_KEY",
    "QDRANT_URL"
)

$failed = $false
foreach ($key in $required) {
    if (-not $values.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($values[$key])) {
        Write-Host "MISSING: $key"
        $failed = $true
        continue
    }
    if ($values[$key] -like "replace-with*" -or $values[$key] -eq "your-audio-bucket") {
        Write-Host "PLACEHOLDER: $key"
        $failed = $true
    }
}

if ($values["MONGODB_URL"] -match "localhost|127\.0\.0\.1") {
    Write-Host "INVALID: MONGODB_URL must not use localhost inside Docker. Use host.docker.internal or a real MongoDB host."
    $failed = $true
}

if ($values["REDIS_URL"] -notmatch "@redis:6379") {
    Write-Host "WARNING: REDIS_URL should usually point to redis:6379 for docker-compose.aws.yml."
}

if ($failed) {
    Write-Host "Env validation failed. Update $Path and run again."
    exit 1
}

Write-Host "Env validation passed."
