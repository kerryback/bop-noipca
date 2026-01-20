# deploy_koyeb.ps1 - Deploy noipca workflow to Koyeb (PowerShell version)
#
# Creates a self-contained Koyeb service that:
# 1. Runs the workflow
# 2. Uploads results to S3
# 3. Auto-deletes itself to stop billing
#
# Usage:
#   .\deploy_koyeb.ps1 <model> <start> <end> [instance_type] [git_repo]
#
# Arguments:
#   model: bgn, kp14, or gs21
#   start: Starting index
#   end: Ending index (exclusive)
#   instance_type: Koyeb instance type (default: 5xlarge)
#   git_repo: GitHub repo in format username/repo (default: YOUR_USERNAME/YOUR_REPO)
#
# Required environment variables:
#   KOYEB_API_TOKEN: Your Koyeb API token
#   AWS_ACCESS_KEY_ID: AWS access key for S3 uploads
#   AWS_SECRET_ACCESS_KEY: AWS secret key for S3 uploads
#
# Optional environment variables:
#   S3_BUCKET: S3 bucket name (default: bop-noipca)
#   AWS_REGION: AWS region (default: us-east-2)
#   KOYEB_APP_NAME: Koyeb app name (default: noipca-app)
#   KOYEB_REGION: Koyeb region (default: was)
#
# Examples:
#   .\deploy_koyeb.ps1 kp14 0 10
#   .\deploy_koyeb.ps1 bgn 0 5 5xlarge myuser/noipca
#   .\deploy_koyeb.ps1 gs21 0 20 6xlarge

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Model,

    [Parameter(Mandatory=$true, Position=1)]
    [int]$Start,

    [Parameter(Mandatory=$true, Position=2)]
    [int]$End,

    [Parameter(Position=3)]
    [string]$InstanceType = "5xlarge",

    [Parameter(Position=4)]
    [string]$GitRepo = "YOUR_USERNAME/YOUR_REPO"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Validate model
$ValidModels = @("bgn", "kp14", "gs21")
if ($Model -notin $ValidModels) {
    Write-Host "ERROR: Invalid model '$Model'" -ForegroundColor Red
    Write-Host "Valid models: bgn, kp14, gs21"
    exit 1
}

# Validate indices
if ($Start -ge $End) {
    Write-Host "ERROR: start ($Start) must be less than end ($End)" -ForegroundColor Red
    exit 1
}

# Check required environment variables
if (-not $env:KOYEB_API_TOKEN) {
    Write-Host "ERROR: KOYEB_API_TOKEN environment variable not set" -ForegroundColor Red
    Write-Host "Set it with: `$env:KOYEB_API_TOKEN = 'your_token'"
    exit 1
}

if (-not $env:AWS_ACCESS_KEY_ID -or -not $env:AWS_SECRET_ACCESS_KEY) {
    Write-Host "ERROR: AWS credentials not set" -ForegroundColor Red
    Write-Host "Set them with:"
    Write-Host "  `$env:AWS_ACCESS_KEY_ID = 'your_key_id'"
    Write-Host "  `$env:AWS_SECRET_ACCESS_KEY = 'your_secret_key'"
    exit 1
}

# Set defaults for optional variables
$S3Bucket = if ($env:S3_BUCKET) { $env:S3_BUCKET } else { "bop-noipca" }
$AwsRegion = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-2" }
$KoyebAppName = if ($env:KOYEB_APP_NAME) { $env:KOYEB_APP_NAME } else { "noipca-app" }
$KoyebRegion = if ($env:KOYEB_REGION) { $env:KOYEB_REGION } else { "was" }

# Generate service name
$ServiceName = "${Model}_${Start}_${End}"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "KOYEB DEPLOYMENT" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Configuration:"
Write-Host "  Model:         $Model"
Write-Host "  Indices:       $Start to $($End-1)"
Write-Host "  Service name:  $ServiceName"
Write-Host "  App name:      $KoyebAppName"
Write-Host "  Instance type: $InstanceType"
Write-Host "  Region:        $KoyebRegion"
Write-Host "  Git repo:      $GitRepo"
Write-Host "  S3 bucket:     $S3Bucket"
Write-Host "  AWS region:    $AwsRegion"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if koyeb CLI is installed
try {
    $null = Get-Command koyeb -ErrorAction Stop
} catch {
    Write-Host "ERROR: koyeb CLI not found" -ForegroundColor Red
    Write-Host "Install it from: https://www.koyeb.com/docs/build-and-deploy/cli"
    exit 1
}

# Create the service
Write-Host "Creating Koyeb service..." -ForegroundColor Yellow

$KoyebArgs = @(
    "services", "create", $ServiceName,
    "--app", $KoyebAppName,
    "--git", "github.com/$GitRepo",
    "--git-branch", "main",
    "--git-run-command", "python main.py $Model $Start $End --koyeb",
    "--instance-type", $InstanceType,
    "--regions", $KoyebRegion,
    "--env", "S3_BUCKET=$S3Bucket",
    "--env", "AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID",
    "--env", "AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY",
    "--env", "AWS_REGION=$AwsRegion",
    "--env", "KOYEB_API_TOKEN=$env:KOYEB_API_TOKEN",
    "--token", $env:KOYEB_API_TOKEN
)

try {
    & koyeb $KoyebArgs
} catch {
    Write-Host "ERROR: Failed to create service" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "SERVICE CREATED SUCCESSFULLY" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The service will:"
Write-Host "  1. Build and deploy your code"
Write-Host "  2. Run: python main.py $Model $Start $End --koyeb"
Write-Host "  3. Upload results to s3://$S3Bucket/koyeb-results/{WORKFLOW_ID}/"
Write-Host "  4. Auto-delete itself to stop billing"
Write-Host ""
Write-Host "Monitor the service:" -ForegroundColor Yellow
Write-Host "  koyeb services get $ServiceName --app $KoyebAppName"
Write-Host ""
Write-Host "View logs in real-time:" -ForegroundColor Yellow
Write-Host "  koyeb services logs $ServiceName --app $KoyebAppName --follow"
Write-Host ""
Write-Host "Download results when complete:" -ForegroundColor Yellow
Write-Host "  aws s3 ls s3://$S3Bucket/koyeb-results/"
Write-Host "  aws s3 sync s3://$S3Bucket/koyeb-results/YYYYMMDD_HHMMSS/ ./results/"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
