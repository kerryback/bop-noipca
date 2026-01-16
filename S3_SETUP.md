# S3 Setup for Koyeb Workflows

This document explains how to configure S3 for automatic upload of workflow results from Koyeb containers.

## ⚠️ IMPORTANT: S3 Configuration is REQUIRED

**Koyeb bills for running instances regardless of CPU usage.** Containers are configured to automatically exit after workflows complete to stop billing. **Without S3 configured, your results will be lost when the container exits.**

## Overview

When S3 is configured, the Koyeb workflow will:
1. Run the workflow (e.g., KP14 for indices 0-9)
2. Upload all logs and outputs to S3
3. Exit cleanly (terminating the container and stopping billing)

If S3 is NOT configured, the workflow will:
1. Run the workflow
2. ⚠️ **LOSE ALL RESULTS** (logs and outputs)
3. Exit after 30 seconds to stop billing

## Prerequisites

### 1. AWS S3 Bucket

Create an S3 bucket for storing results:

```bash
# Using AWS CLI
aws s3 mb s3://your-workflow-results

# Or create via AWS Console:
# https://s3.console.aws.amazon.com/s3/bucket/create
```

### 2. AWS IAM User

Create an IAM user with S3 access:

1. Go to AWS IAM Console: https://console.aws.amazon.com/iam/
2. Create new user (e.g., `koyeb-uploader`)
3. Attach policy `AmazonS3FullAccess` (or create custom policy below)
4. Create access key and save credentials

**Custom IAM Policy (recommended)**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-workflow-results",
        "arn:aws:s3:::your-workflow-results/*"
      ]
    }
  ]
}
```

## Koyeb Configuration

### Method 1: Using Koyeb Web Console

1. Go to https://app.koyeb.com
2. Select your app (e.g., `noipca-app`)
3. Select your service (e.g., `kp14-5xlarge`)
4. Click **Settings** → **Environment Variables**
5. Add the following environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `S3_BUCKET` | `your-workflow-results` | S3 bucket name |
| `AWS_ACCESS_KEY_ID` | `AKIAXXXXXXXXXXXXXXXX` | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | `xxxxxxxxxxxxxxxxxxxxx` | AWS secret access key |
| `AWS_REGION` | `us-east-1` | AWS region (optional) |
| `WORKFLOW_ID` | `kp14_run1` | Custom identifier (optional) |

6. Click **Save** (this will trigger a redeploy)

### Method 2: Using Koyeb CLI

```bash
# Set environment variables
koyeb services update kp14-5xlarge \
  --app noipca-app \
  --env S3_BUCKET=your-workflow-results \
  --env AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX \
  --env AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxx \
  --env AWS_REGION=us-east-1 \
  --token YOUR_KOYEB_TOKEN
```

### Method 3: Using Docker (for local testing)

```bash
docker run -e S3_BUCKET=your-workflow-results \
           -e AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX \
           -e AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxx \
           -e AWS_REGION=us-east-1 \
           your-image:latest
```

## Environment Variables Reference

### Required Variables

- **`S3_BUCKET`**: Name of the S3 bucket where results will be uploaded
- **`AWS_ACCESS_KEY_ID`**: AWS access key ID for authentication
- **`AWS_SECRET_ACCESS_KEY`**: AWS secret access key for authentication

### Optional Variables

- **`AWS_REGION`**: AWS region (default: `us-east-1`)
- **`WORKFLOW_ID`**: Custom identifier for this workflow run (default: timestamp like `20260116_143022`)

## S3 Upload Structure

Results are uploaded with the following structure:

```
s3://your-bucket/
└── koyeb-results/
    └── {WORKFLOW_ID}/          # e.g., "20260116_143022" or custom ID
        ├── logs/
        │   └── kp14_0_10.log   # Workflow logs
        └── outputs/
            ├── kp14_0_dkkm_6.pkl
            ├── kp14_0_dkkm_36.pkl
            ├── kp14_0_dkkm_360.pkl
            └── ...             # All output pickle files
```

## Downloading from S3

After workflow completion, download results from S3:

```bash
# Using AWS CLI
aws s3 sync s3://your-workflow-results/koyeb-results/20260116_143022/ ./local_results/

# Or download specific files
aws s3 cp s3://your-workflow-results/koyeb-results/20260116_143022/logs/kp14_0_10.log ./
aws s3 cp s3://your-workflow-results/koyeb-results/20260116_143022/outputs/ ./outputs/ --recursive
```

## Troubleshooting

### Upload Fails with "Access Denied"

**Problem**: AWS credentials are invalid or lack permissions

**Solution**:
- Verify IAM user has S3 permissions
- Check access key is correct
- Ensure bucket name is correct

### Upload Fails with "Bucket not found"

**Problem**: S3 bucket doesn't exist or is in different region

**Solution**:
- Create bucket: `aws s3 mb s3://your-workflow-results`
- Verify region matches `AWS_REGION` variable

### Container Stays Alive Forever

**Problem**: S3 environment variables not set

**Solution**:
- Check environment variables in Koyeb service settings
- At minimum, `S3_BUCKET` and `AWS_ACCESS_KEY_ID` must be set
- Redeploy service after adding variables

### Upload Succeeds but Container Doesn't Exit

**Problem**: Bug in run_kp14.py exit logic

**Solution**:
- Check logs for errors after "✓ Results uploaded successfully"
- Container should exit with "Container exiting..." message

## Security Best Practices

1. **Use IAM policies with minimal permissions** (only S3 access, not full admin)
2. **Rotate access keys regularly** (every 90 days)
3. **Enable S3 bucket encryption** (AES-256 or KMS)
4. **Use S3 bucket policies** to restrict access:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Deny",
         "Principal": "*",
         "Action": "s3:*",
         "Resource": [
           "arn:aws:s3:::your-workflow-results",
           "arn:aws:s3:::your-workflow-results/*"
         ],
         "Condition": {
           "Bool": {
             "aws:SecureTransport": "false"
           }
         }
       }
     ]
   }
   ```
5. **Enable S3 versioning** to protect against accidental deletions
6. **Set lifecycle policies** to archive old results to Glacier

## Testing S3 Upload Locally

Test the upload script locally before deploying:

```bash
# Set environment variables
export S3_BUCKET=your-workflow-results
export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxx
export AWS_REGION=us-east-1
export WORKFLOW_ID=test_run

# Create test files
mkdir -p /app/logs /app/outputs
echo "test log" > /app/logs/test.log
echo "test output" > /app/outputs/test.pkl

# Run upload script
python upload_to_s3.py

# Check S3
aws s3 ls s3://your-workflow-results/koyeb-results/test_run/
```

## Cost Considerations

**S3 Storage Costs** (as of 2025):
- Standard storage: ~$0.023 per GB/month
- Typical workflow: ~500MB logs + 5GB outputs = 5.5GB = **~$0.13/month**

**S3 Transfer Costs**:
- Data transfer OUT from S3 to internet: First 100GB free/month
- Koyeb to S3 upload: Free (within AWS)

**Recommendation**: Use S3 lifecycle policies to:
- Transition to Glacier after 30 days: ~$0.004 per GB/month
- Delete after 1 year if results are no longer needed

## Alternative: Skip S3 and Keep Container Alive

If you prefer manual download, simply **don't set S3 environment variables**.

The workflow will:
1. Complete normally
2. Print instructions for manual download
3. Keep container alive indefinitely

You can then download results with:
```bash
koyeb instances cp <INSTANCE_ID>:/app/logs/ ./local_logs/
koyeb instances cp <INSTANCE_ID>:/app/outputs/ ./local_outputs/
```

Then delete the service:
```bash
koyeb services delete kp14-5xlarge --app noipca-app --token YOUR_TOKEN
```
