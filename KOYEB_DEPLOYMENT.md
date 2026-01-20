# Koyeb Deployment Guide

This guide explains how to deploy noipca workflows to Koyeb with a fully automated, self-contained process.

## Overview

The Koyeb deployment creates a **self-contained workflow** that:

1. ✅ Creates a Koyeb service (you run one command)
2. ✅ Builds and deploys your code automatically
3. ✅ Runs the workflow with optimal settings (N_JOBS=24)
4. ✅ Uploads all results to S3 incrementally
5. ✅ **Auto-deletes the service** to stop billing when done

**No manual cleanup required!** The service deletes itself when the workflow completes.

## Prerequisites

### 1. Koyeb Account

- Sign up at https://www.koyeb.com
- Get your API token from: https://app.koyeb.com/account/api

### 2. AWS S3 Bucket

- Create an S3 bucket for results storage (e.g., `bop-noipca`)
- Create IAM user with S3 access
- Get AWS access key and secret key

See [S3_SETUP.md](S3_SETUP.md) for detailed S3 configuration.

### 3. Koyeb CLI

Install the Koyeb CLI:

```bash
# macOS
brew install koyeb/tap/koyeb-cli

# Linux
curl -fsSL https://raw.githubusercontent.com/koyeb/koyeb-cli/master/install.sh | sh

# Windows (WSL)
curl -fsSL https://raw.githubusercontent.com/koyeb/koyeb-cli/master/install.sh | sh
```

Verify installation:
```bash
koyeb version
```

## Quick Start (Recommended)

### Step 1: Set Environment Variables

**Linux/macOS/Git Bash:**
```bash
export KOYEB_API_TOKEN=your_koyeb_api_token
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

**Windows PowerShell:**
```powershell
$env:KOYEB_API_TOKEN = "your_koyeb_api_token"
$env:AWS_ACCESS_KEY_ID = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
```

Optional (defaults shown):
```bash
# Linux/macOS/Git Bash
export S3_BUCKET=bop-noipca
export AWS_REGION=us-east-2
export KOYEB_APP_NAME=noipca-app

# Windows PowerShell
$env:S3_BUCKET = "bop-noipca"
$env:AWS_REGION = "us-east-2"
$env:KOYEB_APP_NAME = "noipca-app"
```

### Step 2: Deploy Using Helper Script

**Linux/macOS/Git Bash:**
```bash
# Deploy kp14 for indices 0-9
./deploy_koyeb.sh kp14 0 10

# Deploy bgn for indices 0-4
./deploy_koyeb.sh bgn 0 5

# Use larger instance (6xlarge)
./deploy_koyeb.sh gs21 0 20 6xlarge

# Specify custom git repo
./deploy_koyeb.sh kp14 0 10 5xlarge yourusername/yourrepo
```

**Windows PowerShell:**
```powershell
# Deploy kp14 for indices 0-9
.\deploy_koyeb.ps1 kp14 0 10

# Deploy bgn for indices 0-4
.\deploy_koyeb.ps1 bgn 0 5

# Use larger instance (6xlarge)
.\deploy_koyeb.ps1 gs21 0 20 6xlarge

# Specify custom git repo
.\deploy_koyeb.ps1 kp14 0 10 5xlarge yourusername/yourrepo
```

### Step 3: Monitor the Service

```bash
# Watch service status
koyeb services get kp14_0_10 --app noipca-app

# View logs in real-time
koyeb services logs kp14_0_10 --app noipca-app --follow
```

### Step 4: Download Results

The service auto-deletes when done. Download results from S3:

```bash
# List available workflow runs
aws s3 ls s3://bop-noipca/koyeb-results/

# Download specific run
aws s3 sync s3://bop-noipca/koyeb-results/20260120_143022/ ./results/

# Download all kp14 results
aws s3 sync s3://bop-noipca/koyeb-results/ ./results/ --exclude "*" --include "*/kp14_*"
```

## Manual Deployment (Advanced)

If you prefer manual control over the deployment:

```bash
# Set parameters
export MODEL=kp14
export START=0
export END=10
export SERVICE_NAME=${MODEL}_${START}_${END}

# Create service
koyeb services create $SERVICE_NAME \
  --app noipca-app \
  --git github.com/yourusername/yourrepo \
  --git-branch main \
  --git-run-command "python main.py $MODEL $START $END --koyeb" \
  --instance-type 5xlarge \
  --regions was \
  --env S3_BUCKET=bop-noipca \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env AWS_REGION=us-east-2 \
  --env KOYEB_API_TOKEN=$KOYEB_API_TOKEN \
  --token $KOYEB_API_TOKEN

# Monitor logs
koyeb services logs $SERVICE_NAME --app noipca-app --follow
```

## Instance Types and Pricing

Koyeb instance types (as of 2025):

| Instance | vCPU | RAM | Price/hour | Best For |
|----------|------|-----|------------|----------|
| nano     | 0.1  | 128MB | ~$0.001  | Testing only |
| micro    | 0.5  | 512MB | ~$0.005  | Testing only |
| small    | 1    | 1GB   | ~$0.01   | Light workloads |
| medium   | 2    | 2GB   | ~$0.02   | Small panels |
| large    | 4    | 4GB   | ~$0.04   | Medium panels |
| xlarge   | 8    | 8GB   | ~$0.08   | Large panels |
| 2xlarge  | 16   | 16GB  | ~$0.16   | Very large panels |
| 3xlarge  | 24   | 24GB  | ~$0.24   | Huge panels |
| 4xlarge  | 32   | 32GB  | ~$0.32   | Massive panels |
| **5xlarge** | **40** | **40GB** | **~$0.40** | **Default (N_JOBS=24)** |
| 6xlarge  | 48   | 48GB  | ~$0.48   | Maximum performance |

**Recommendation**: Use `5xlarge` (default) for optimal performance with N_JOBS=24.

## Service Auto-Deletion

The service automatically deletes itself when the workflow completes, **stopping billing immediately**.

This happens in [main.py](main.py#L641-L696):

```python
if use_koyeb:
    koyeb_app_name = os.environ.get('KOYEB_APP_NAME')
    koyeb_service_name = os.environ.get('KOYEB_SERVICE_NAME')
    koyeb_api_token = os.environ.get('KOYEB_API_TOKEN')

    # Delete service to stop billing
    subprocess.run([
        'koyeb', 'services', 'delete', koyeb_service_name,
        '--app', koyeb_app_name,
        '--token', koyeb_api_token
    ])
```

**Required environment variables** (set automatically by Koyeb):
- `KOYEB_APP_NAME` - App name (e.g., "noipca-app")
- `KOYEB_SERVICE_NAME` - Service name (e.g., "kp14_0_10")
- `KOYEB_API_TOKEN` - Your API token (you must set this)

**Important**: You **must** set `KOYEB_API_TOKEN` as an environment variable when creating the service, otherwise auto-deletion won't work and you'll need to manually delete the service.

## Workflow Configuration

When running with `--koyeb` flag, the workflow uses Koyeb-optimized settings:

| Setting | Default (jgsrc1) | Koyeb (--koyeb) |
|---------|-----------------|-----------------|
| N_JOBS | 10 | **24** |
| TEMP_DIR | /opt/scratch/keb7 | outputs/ |
| S3 Upload | Optional | **Automatic** |
| Auto-cleanup | No | **Yes** |

See [config.py](config.py) for all configuration options.

## S3 Upload Structure

Results are uploaded to S3 with the following structure:

```
s3://bop-noipca/
└── koyeb-results/
    └── 20260120_143022/          # Timestamp (YYYYMMDD_HHMMSS)
        ├── outputs/
        │   ├── kp14_0_dkkm_6.pkl
        │   ├── kp14_0_dkkm_36.pkl
        │   ├── kp14_0_dkkm_360.pkl
        │   ├── kp14_0_dkkm_3600.pkl
        │   ├── kp14_0_dkkm_36000.pkl
        │   ├── kp14_0_dkkm_6_W.pkl       # Weight matrix
        │   ├── kp14_0_fama.pkl
        │   ├── kp14_0_moments.pkl
        │   └── kp14_0_panel.pkl
        └── logs/
            └── kp14_0_10.log
```

Files are uploaded **incrementally** as they are created (not at the end), so partial results are preserved even if the workflow fails.

## Troubleshooting

### Service doesn't auto-delete

**Problem**: Service stays alive after workflow completes

**Solution**: Check that `KOYEB_API_TOKEN` is set as an environment variable:

```bash
# When creating service, ensure --env KOYEB_API_TOKEN is included
koyeb services create ... --env KOYEB_API_TOKEN=$KOYEB_API_TOKEN ...
```

### Upload fails

**Problem**: Files not appearing in S3

**Solution**:
1. Check AWS credentials are correct
2. Verify S3 bucket exists: `aws s3 ls s3://bop-noipca/`
3. Check service logs: `koyeb services logs <service> --app noipca-app`

### Service fails to start

**Problem**: Service status shows "Error"

**Solution**:
1. Check logs: `koyeb services logs <service> --app noipca-app`
2. Verify git repo is accessible
3. Check git branch exists (default: main)
4. Ensure all environment variables are set

### Out of memory

**Problem**: Service crashes with memory error

**Solution**: Use larger instance type:

```bash
./deploy_koyeb.sh kp14 0 10 6xlarge
```

## Cost Estimation

**Typical workflow** (kp14 0-9 on 5xlarge):
- Instance: $0.40/hour
- Runtime: ~2 hours
- **Total: ~$0.80**

**S3 storage**:
- ~5GB per workflow
- $0.023/GB/month
- **Total: ~$0.12/month**

**Best practices**:
1. Use S3 lifecycle policies to archive old results
2. Service auto-deletes to stop billing immediately
3. Monitor logs to catch failures early

## Examples

### Run multiple models in sequence

```bash
# Deploy all three models
./deploy_koyeb.sh bgn 0 10
./deploy_koyeb.sh kp14 0 10
./deploy_koyeb.sh gs21 0 10

# Monitor all services
koyeb services list --app noipca-app

# Services will auto-delete as they complete
```

### Test with small workload

```bash
# Use smaller instance for testing
./deploy_koyeb.sh bgn 0 1 small

# Monitor
koyeb services logs bgn_0_1 --app noipca-app --follow
```

### Download all results

```bash
# List all workflow runs
aws s3 ls s3://bop-noipca/koyeb-results/

# Download everything
aws s3 sync s3://bop-noipca/koyeb-results/ ./all_results/

# Download only DKKM files
aws s3 sync s3://bop-noipca/koyeb-results/ ./dkkm_only/ \
  --exclude "*" --include "*_dkkm_*.pkl"
```

## Support

- Koyeb documentation: https://www.koyeb.com/docs
- Koyeb CLI reference: https://www.koyeb.com/docs/build-and-deploy/cli/reference
- AWS S3 setup: [S3_SETUP.md](S3_SETUP.md)
- Issue tracker: https://github.com/YOUR_USERNAME/YOUR_REPO/issues
