$ErrorActionPreference = "Stop"
$envPath = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path -LiteralPath $envPath) {
    Write-Host "Existing .env preserved."
    exit 0
}
$bytes = New-Object byte[] 32
$generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $generator.GetBytes($bytes)
} finally {
    $generator.Dispose()
}
$password = [BitConverter]::ToString($bytes).Replace("-", "").ToLowerInvariant()
Set-Content -LiteralPath $envPath -Value "MLFLOW_DB_PASSWORD=$password" -Encoding ASCII
Write-Host "Created ignored local .env with a generated database password."
