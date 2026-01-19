"""
Download fama and dkkm pickle files from S3 koyeb-results folders.

This script:
1. Scans all timestamp folders in s3://bop-noipca/koyeb-results/
2. Downloads all *_fama.pkl and *_dkkm_*.pkl files
3. Renames them to include the timestamp folder: {model}_{timestamp}_{n}_{type}.pkl

Example:
    s3://bop-noipca/koyeb-results/20260117_092453/outputs/bgn_0_fama.pkl
    -> noipca/outputs/bgn_20260117_092453_0_fama.pkl

Usage:
    python download_s3.py [--bucket BUCKET] [--region REGION]

Arguments:
    --bucket: S3 bucket name (default: bop-noipca)
    --region: AWS region (default: us-east-2)

Environment:
    Loads AWS credentials from .env file using python-dotenv
    Required variables: AWS_KEY_ID, AWS_SECRET_KEY
"""

import os
import sys
import boto3
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_BUCKET = 'bop-noipca'
DEFAULT_REGION = 'us-east-2'
DEFAULT_PREFIX = 'koyeb-results/'

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Download fama and dkkm files from S3')
    parser.add_argument('--bucket', default=DEFAULT_BUCKET, help='S3 bucket name')
    parser.add_argument('--region', default=DEFAULT_REGION, help='AWS region')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help='S3 prefix to scan')
    parser.add_argument('--output-dir', default='outputs', help='Local output directory')
    return parser.parse_args()


def get_s3_client(region):
    """Create S3 client with credentials from environment."""
    return boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=os.environ.get('AWS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
    )


def list_timestamp_folders(s3_client, bucket, prefix):
    """List all timestamp folders in S3 prefix."""
    print(f"Scanning S3 bucket: {bucket}/{prefix}")

    paginator = s3_client.get_paginator('list_objects_v2')
    folders = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        # CommonPrefixes contains the "folders" (keys ending with /)
        for prefix_info in page.get('CommonPrefixes', []):
            folder = prefix_info['Prefix']
            # Extract timestamp folder name (e.g., "20260117_092453")
            folder_name = folder.rstrip('/').split('/')[-1]
            if folder_name:  # Skip empty
                folders.add(folder_name)

    return sorted(list(folders))


def list_files_to_download(s3_client, bucket, timestamp_folder):
    """List all fama and dkkm files in a timestamp folder."""
    prefix = f"koyeb-results/{timestamp_folder}/outputs/"

    paginator = s3_client.get_paginator('list_objects_v2')
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = os.path.basename(key)

            # Check if it's a fama or dkkm file
            if filename.endswith('_fama.pkl') or '_dkkm_' in filename and filename.endswith('.pkl'):
                files.append({
                    'key': key,
                    'filename': filename,
                    'size': obj['Size']
                })

    return files


def generate_local_filename(original_filename, timestamp_folder):
    """
    Generate local filename with timestamp folder embedded.

    Examples:
        bgn_0_fama.pkl -> bgn_20260117_092453_0_fama.pkl
        kp14_5_dkkm_360.pkl -> kp14_20260117_092453_5_dkkm_360.pkl
    """
    # Split filename: model_n_type.pkl or model_n_dkkm_features.pkl
    parts = original_filename.replace('.pkl', '').split('_')

    if len(parts) < 3:
        # Shouldn't happen, but handle gracefully
        return f"{timestamp_folder}_{original_filename}"

    model = parts[0]  # bgn, kp14, gs21
    panel_idx = parts[1]  # 0, 1, 2, ...
    rest = '_'.join(parts[2:])  # fama, dkkm_360, etc.

    return f"{model}_{timestamp_folder}_{panel_idx}_{rest}.pkl"


def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3."""
    s3_client.download_file(bucket, s3_key, str(local_path))


def main():
    """Main execution function."""
    args = parse_args()

    print("="*70)
    print("S3 DOWNLOAD - FAMA AND DKKM FILES")
    print("="*70)
    print()
    print(f"Bucket: {args.bucket}")
    print(f"Region: {args.region}")
    print(f"Prefix: {args.prefix}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Check AWS credentials
    if not os.environ.get('AWS_KEY_ID') or not os.environ.get('AWS_SECRET_KEY'):
        print("ERROR: AWS credentials not found in environment")
        print("Please set AWS_KEY_ID and AWS_SECRET_KEY in .env file")
        sys.exit(1)

    # Create S3 client
    try:
        s3_client = get_s3_client(args.region)
    except Exception as e:
        print(f"ERROR: Failed to create S3 client: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # List timestamp folders
    print("Discovering timestamp folders...")
    try:
        timestamp_folders = list_timestamp_folders(s3_client, args.bucket, args.prefix)
    except Exception as e:
        print(f"ERROR: Failed to list timestamp folders: {e}")
        sys.exit(1)

    if not timestamp_folders:
        print("No timestamp folders found")
        sys.exit(0)

    print(f"Found {len(timestamp_folders)} timestamp folders:")
    for folder in timestamp_folders:
        print(f"  - {folder}")
    print()

    # Download files from each timestamp folder
    total_files = 0
    total_bytes = 0

    for timestamp_folder in timestamp_folders:
        print(f"Processing {timestamp_folder}...")

        # List files to download
        try:
            files = list_files_to_download(s3_client, args.bucket, timestamp_folder)
        except Exception as e:
            print(f"  ERROR: Failed to list files: {e}")
            continue

        if not files:
            print(f"  No fama/dkkm files found")
            continue

        print(f"  Found {len(files)} files")

        # Download each file
        for file_info in files:
            original_filename = file_info['filename']
            s3_key = file_info['key']
            size_mb = file_info['size'] / (1024 * 1024)

            # Generate local filename
            local_filename = generate_local_filename(original_filename, timestamp_folder)
            local_path = output_dir / local_filename

            # Download
            try:
                print(f"    Downloading {original_filename} -> {local_filename} ({size_mb:.2f} MB)...")
                download_file(s3_client, args.bucket, s3_key, local_path)
                total_files += 1
                total_bytes += file_info['size']
            except Exception as e:
                print(f"    ERROR: Failed to download {original_filename}: {e}")
                continue

        print()

    # Summary
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Total files downloaded: {total_files}")
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)")
    print(f"Output directory: {output_dir.absolute()}")
    print()


if __name__ == "__main__":
    main()
