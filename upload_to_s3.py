"""
Upload workflow results to AWS S3.

This script uploads logs and output files from the Koyeb container to S3
after workflow completion, allowing the container to exit cleanly.

Required environment variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    S3_BUCKET: Target S3 bucket name
    AWS_REGION: AWS region (optional, defaults to us-east-1)
    WORKFLOW_ID: Identifier for this workflow run (optional)

Usage:
    python upload_to_s3.py
"""

import boto3
import os
import sys
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


def upload_file_to_s3(s3_client, local_path, bucket_name, s3_key):
    """
    Upload a single file to S3 with progress tracking.

    Args:
        s3_client: Boto3 S3 client
        local_path: Path to local file
        bucket_name: S3 bucket name
        s3_key: S3 object key

    Returns:
        True if successful, False otherwise
    """
    try:
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"  Uploading {local_path.name} ({file_size:.2f} MB)...")

        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        print(f"    ✓ Uploaded to s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"    ✗ Failed to upload {local_path.name}: {e}")
        return False
    except Exception as e:
        print(f"    ✗ Error uploading {local_path.name}: {e}")
        return False


def upload_directory_to_s3(local_dir, bucket_name, s3_prefix, s3_client):
    """
    Upload entire directory to S3, preserving directory structure.

    Args:
        local_dir: Local directory path
        bucket_name: S3 bucket name
        s3_prefix: S3 key prefix
        s3_client: Boto3 S3 client

    Returns:
        Tuple of (success_count, fail_count)
    """
    local_path = Path(local_dir)

    if not local_path.exists():
        print(f"  Directory not found: {local_dir}")
        return 0, 0

    files = list(local_path.rglob('*'))
    files = [f for f in files if f.is_file()]

    if not files:
        print(f"  No files found in {local_dir}")
        return 0, 0

    print(f"  Found {len(files)} files to upload")

    success_count = 0
    fail_count = 0

    for file_path in files:
        # Get relative path for S3 key
        relative_path = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')

        if upload_file_to_s3(s3_client, file_path, bucket_name, s3_key):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def main():
    """Main execution function."""
    print("="*70)
    print("S3 UPLOAD - WORKFLOW RESULTS")
    print("="*70)
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get configuration from environment
    bucket_name = os.environ.get('S3_BUCKET')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    workflow_id = os.environ.get('WORKFLOW_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Validate required environment variables
    if not bucket_name:
        print("ERROR: S3_BUCKET environment variable not set")
        print("\nRequired environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - S3_BUCKET")
        print("  - AWS_REGION (optional)")
        sys.exit(1)

    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("ERROR: AWS credentials not set")
        print("\nRequired environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  S3 Bucket: {bucket_name}")
    print(f"  AWS Region: {aws_region}")
    print(f"  Workflow ID: {workflow_id}")
    print()

    # Initialize S3 client
    try:
        s3_client = boto3.client('s3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=aws_region
        )

        # Test S3 connection
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Connected to S3 bucket: {bucket_name}")
        print()
    except NoCredentialsError:
        print("ERROR: AWS credentials are invalid or not found")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"ERROR: S3 bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            print(f"ERROR: Access denied to S3 bucket '{bucket_name}'")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to connect to S3: {e}")
        sys.exit(1)

    # Upload logs
    print("-"*70)
    print("Uploading logs...")
    print("-"*70)
    logs_success, logs_fail = upload_directory_to_s3(
        '/app/logs',
        bucket_name,
        f'koyeb-results/{workflow_id}/logs',
        s3_client
    )

    # Upload outputs
    print()
    print("-"*70)
    print("Uploading outputs...")
    print("-"*70)
    outputs_success, outputs_fail = upload_directory_to_s3(
        '/app/outputs',
        bucket_name,
        f'koyeb-results/{workflow_id}/outputs',
        s3_client
    )

    # Summary
    print()
    print("="*70)
    print("UPLOAD COMPLETE")
    print("="*70)
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Results:")
    print(f"  Logs:    {logs_success} uploaded, {logs_fail} failed")
    print(f"  Outputs: {outputs_success} uploaded, {outputs_fail} failed")
    print(f"  Total:   {logs_success + outputs_success} uploaded, {logs_fail + outputs_fail} failed")
    print()
    print(f"S3 Location: s3://{bucket_name}/koyeb-results/{workflow_id}/")
    print("="*70)

    # Exit with error code if any uploads failed
    if logs_fail + outputs_fail > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
