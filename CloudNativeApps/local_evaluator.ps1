# Local Evaluator for Entrega 2
# Usage: .\local_evaluator.ps1 -RF "rf004"

param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("all", "rf003", "rf004", "rf005")]
    [string]$RF = "all"
)

# 1. Create evaluator folder
$evalFolder = ".evaluator"
if (-not (Test-Path $evalFolder)) {
    New-Item -ItemType Directory -Path $evalFolder | Out-Null
}

# 2. Define resources
$baseUrl = "https://raw.githubusercontent.com/MISW-4301-Desarrollo-Apps-en-la-Nube/recursos-evaluador/main/entrega2"
$files = @(
    "verify_old_endpoints.json",
    "evaluate_rf003.json",
    "evaluate_rf003_consistency.json",
    "evaluate_rf004.json",
    "evaluate_rf004_consistency.json",
    "evaluate_rf005.json",
    "evaluate_rf005_consistency.json"
)

# 3. Download resources
Write-Host "Downloading test collections..." -ForegroundColor Cyan
foreach ($file in $files) {
    $dest = Join-Path $evalFolder $file
    if (-not (Test-Path $dest)) {
        Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $dest
    }
}

# 4. Get BASE_PATH from config.yaml
if (-not (Test-Path "config.yaml")) {
    Write-Error "config.yaml not found!"
    exit 1
}

$config = Get-Content "config.yaml" -Raw
$basePathMatch = $config -match 'url:\s*"?(https?://[^"\s]+)"?'
if ($matches) {
    $BASE_PATH = $matches[1]
    Write-Host "Target URL: $BASE_PATH" -ForegroundColor Green
} else {
    Write-Error "Could not find 'url' in config.yaml"
    exit 1
}

# 5. Run tests
function Run-Newman($file) {
    $path = Join-Path $evalFolder $file
    Write-Host "Running tests: $file" -ForegroundColor Yellow
    npx newman run $path --env-var "BASE_PATH=$BASE_PATH" --verbose --insecure
}

if ($RF -eq "all" -or $RF -eq "rf003") {
    Run-Newman "evaluate_rf003.json"
}

if ($RF -eq "all" -or $RF -eq "rf004") {
    Run-Newman "evaluate_rf004.json"
}

if ($RF -eq "all" -or $RF -eq "rf005") {
    Run-Newman "evaluate_rf005.json"
}

Write-Host "Evaluation finished." -ForegroundColor Cyan
