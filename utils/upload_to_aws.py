"""
AWS S3 upload utilities for incremental file uploads.

Provides minimal-output functions to upload files to S3 as they are created.

Usage:
    from utils.upload_to_aws import upload_file, is_s3_configured

    # Check if S3 is configured
    if is_s3_configured():
        # Upload a file after creation
        upload_file('/path/to/file.pkl')
"""

import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime


def is_s3_configured():
    """
    Check if S3 environment variables are configured.

    Returns:
        bool: True if S3_BUCKET and AWS credentials are set
    """
    return bool(
        os.environ.get('S3_BUCKET') and
        os.environ.get('AWS_ACCESS_KEY_ID') and
        os.environ.get('AWS_SECRET_ACCESS_KEY')
    )


def get_s3_client():
    """
    Create and return a boto3 S3 client.

    Returns:
        boto3.client or None: S3 client if credentials are valid, None otherwise
    """
    if not is_s3_configured():
        return None

    try:
        aws_region = os.environ.get('AWS_REGION', 'us-east-2')
        s3_client = boto3.client('s3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=aws_region
        )
        return s3_client
    except Exception:
        return None


def upload_file(filepath, verbose=True):
    """
    Upload a single file to S3 with minimal output.

    Args:
        filepath: Path to local file (absolute or relative)
        verbose: If True, print success message (default: True)

    Returns:
        bool: True if uploaded successfully, False otherwise
    """
    if not is_s3_configured():
        return False

    filepath = Path(filepath)
    if not filepath.exists():
        if verbose:
            print(f"⚠ File not found: {filepath.name}")
        return False

    try:
        s3_client = get_s3_client()
        if s3_client is None:
            return False

        bucket_name = os.environ.get('S3_BUCKET')
        workflow_id = os.environ.get('WORKFLOW_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))

        # Determine S3 key based on file location
        # If file is in outputs/, use that; otherwise use logs/
        if '/outputs/' in str(filepath) or '\\outputs\\' in str(filepath):
            s3_key = f'koyeb-results/{workflow_id}/outputs/{filepath.name}'
        elif '/logs/' in str(filepath) or '\\logs\\' in str(filepath):
            s3_key = f'koyeb-results/{workflow_id}/logs/{filepath.name}'
        else:
            # Default: use outputs
            s3_key = f'koyeb-results/{workflow_id}/outputs/{filepath.name}'

        # Upload file
        s3_client.upload_file(str(filepath), bucket_name, s3_key)

        if verbose:
            print(f"✓ Uploaded {filepath.name} to S3")

        return True

    except (ClientError, NoCredentialsError, Exception):
        # Silently fail - don't interrupt workflow
        return False


def upload_directory(dirpath, verbose=True):
    """
    Upload all files in a directory to S3.

    Args:
        dirpath: Path to directory
        verbose: If True, print progress (default: True)

    Returns:
        tuple: (success_count, fail_count)
    """
    if not is_s3_configured():
        return 0, 0

    dirpath = Path(dirpath)
    if not dirpath.exists() or not dirpath.is_dir():
        return 0, 0

    files = list(dirpath.rglob('*'))
    files = [f for f in files if f.is_file()]

    if not files:
        return 0, 0

    success_count = 0
    fail_count = 0

    for filepath in files:
        if upload_file(filepath, verbose=verbose):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count
