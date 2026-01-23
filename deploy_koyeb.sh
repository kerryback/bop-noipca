#!/bin/bash
# deploy_koyeb.sh - Deploy noipca workflow to Koyeb
#
# Creates a self-contained Koyeb service that:
# 1. Runs the workflow
# 2. Uploads results to S3
# 3. Auto-deletes itself to stop billing
#
# Usage:
#   ./deploy_koyeb.sh <model> <start> <end> [instance_type] [git_repo]
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
#   KOYEB_APP_NAME: Koyeb app name (default: noipca-<model>-<start>-<end>)
#   KOYEB_REGION: Koyeb region (default: was)
#   MONITOR_URL: URL of koyeb-monitor service for self-termination
#
# Examples:
#   ./deploy_koyeb.sh kp14 0 10
#   ./deploy_koyeb.sh bgn 0 5 5xlarge myuser/noipca
#   ./deploy_koyeb.sh gs21 0 20 6xlarge

set -e  # Exit on error

# Check arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Missing required arguments"
    echo ""
    echo "Usage: $0 <model> <start> <end> [instance_type] [git_repo]"
    echo ""
    echo "Arguments:"
    echo "  model:         bgn, kp14, or gs21"
    echo "  start:         Starting index"
    echo "  end:           Ending index (exclusive)"
    echo "  instance_type: Koyeb instance (default: 5xlarge)"
    echo "  git_repo:      GitHub repo username/repo (default: YOUR_USERNAME/YOUR_REPO)"
    echo ""
    echo "Examples:"
    echo "  $0 kp14 0 10"
    echo "  $0 bgn 0 5 5xlarge myuser/noipca"
    exit 1
fi

# Parse arguments
MODEL=$1
START=$2
END=$3
INSTANCE_TYPE=${4:-5xlarge}
GIT_REPO=${5:-}

# Auto-detect git repository if not provided
if [ -z "$GIT_REPO" ]; then
    echo "Auto-detecting git repository..."
    if git remote get-url origin &> /dev/null; then
        # Extract username/repo from git remote URL
        # Handles both HTTPS and SSH formats
        REMOTE_URL=$(git remote get-url origin)

        # Extract from HTTPS: https://github.com/user/repo.git -> user/repo
        if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+/[^/]+)(\.git)?$ ]]; then
            GIT_REPO="${BASH_REMATCH[1]}"
            GIT_REPO="${GIT_REPO%.git}"  # Remove .git suffix if present
            echo "✓ Detected repository: $GIT_REPO"
        else
            echo "ERROR: Could not parse git remote URL: $REMOTE_URL"
            echo "Please specify git repo manually: $0 $MODEL $START $END $INSTANCE_TYPE username/repo"
            exit 1
        fi
    else
        echo "ERROR: No git repository detected and none specified"
        echo "Either run from a git repository or specify manually:"
        echo "  $0 $MODEL $START $END $INSTANCE_TYPE username/repo"
        exit 1
    fi
fi

# Validate model
if [[ ! "$MODEL" =~ ^(bgn|kp14|gs21)$ ]]; then
    echo "ERROR: Invalid model '$MODEL'"
    echo "Valid models: bgn, kp14, gs21"
    exit 1
fi

# Validate indices
if [ "$START" -ge "$END" ]; then
    echo "ERROR: start ($START) must be less than end ($END)"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading credentials from .env file..."
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue

        # Remove leading/trailing whitespace and quotes
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        value="${value%\"}"
        value="${value#\"}"

        # Only set if not already set (explicit env vars take priority)
        if [ -z "${!key}" ]; then
            export "$key=$value"
        fi
    done < .env
    echo "Loaded credentials from .env"
fi

# Check required environment variables
if [ -z "$KOYEB_API_TOKEN" ]; then
    echo "ERROR: KOYEB_API_TOKEN environment variable not set"
    echo "Set it with: export KOYEB_API_TOKEN=your_token"
    echo "Or add it to .env file: KOYEB_API_TOKEN=your_token"
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: AWS credentials not set"
    echo "Set them with:"
    echo "  export AWS_ACCESS_KEY_ID=your_key_id"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "Or add them to .env file:"
    echo "  AWS_ACCESS_KEY_ID=your_key_id"
    echo "  AWS_SECRET_ACCESS_KEY=your_secret_key"
    exit 1
fi

# Set defaults for optional variables
S3_BUCKET=${S3_BUCKET:-bop-noipca}
AWS_REGION=${AWS_REGION:-us-east-2}
KOYEB_APP_NAME=${KOYEB_APP_NAME:-noipca-${MODEL}-${START}-${END}}
KOYEB_REGION=${KOYEB_REGION:-was}
MONITOR_URL=${MONITOR_URL:-https://koyeb-monitor-kerrybackapps-c07b20b0.koyeb.app}

# Auto-detect git branch (default: current branch)
if [ -z "$GIT_BRANCH" ]; then
    GIT_BRANCH=$(git branch --show-current)
    if [ -z "$GIT_BRANCH" ]; then
        GIT_BRANCH="main"  # Fallback to main if detection fails
    fi
fi

# Generate service name from job arguments
SERVICE_NAME="${MODEL}-${START}-${END}"

echo "=========================================="
echo "KOYEB DEPLOYMENT"
echo "=========================================="
echo "Configuration:"
echo "  Model:         $MODEL"
echo "  Indices:       $START to $((END-1))"
echo "  Service name:  $SERVICE_NAME"
echo "  App name:      $KOYEB_APP_NAME"
echo "  Instance type: $INSTANCE_TYPE"
echo "  Region:        $KOYEB_REGION"
echo "  Git repo:      $GIT_REPO"
echo "  Git branch:    $GIT_BRANCH"
echo "  S3 bucket:     $S3_BUCKET"
echo "  AWS region:    $AWS_REGION"
echo "=========================================="
echo ""

# Check if koyeb CLI is installed
if ! command -v koyeb &> /dev/null; then
    echo "ERROR: koyeb CLI not found"
    echo "Install it from: https://www.koyeb.com/docs/build-and-deploy/cli"
    exit 1
fi

# Check if app exists, create if it doesn't
echo "Checking if app '$KOYEB_APP_NAME' exists..."
if ! koyeb apps get "$KOYEB_APP_NAME" --token "$KOYEB_API_TOKEN" &> /dev/null; then
    echo "App doesn't exist, creating '$KOYEB_APP_NAME'..."
    koyeb apps create "$KOYEB_APP_NAME" --token "$KOYEB_API_TOKEN"
    echo "✓ App created successfully"
else
    echo "✓ App already exists"
fi

# Create the service
echo "Creating Koyeb service..."
koyeb services create "$SERVICE_NAME" \
  --app "$KOYEB_APP_NAME" \
  --type worker \
  --git "github.com/$GIT_REPO" \
  --git-branch "$GIT_BRANCH" \
  --git-run-command "python main.py $MODEL $START $END --koyeb && curl -s -X POST \$MONITOR_URL/kill -H \"Content-Type: application/json\" -d \"{\\\"service_name\\\": \\\"\$KOYEB_SERVICE_NAME\\\", \\\"app_name\\\": \\\"\$KOYEB_APP_NAME\\\"}\"" \
  --instance-type "$INSTANCE_TYPE" \
  --regions "$KOYEB_REGION" \
  --env MONITOR_URL="$MONITOR_URL" \
  --env S3_BUCKET="$S3_BUCKET" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env AWS_REGION="$AWS_REGION" \
  --env KOYEB_API_TOKEN="$KOYEB_API_TOKEN" \
  --env KOYEB_APP_NAME="$KOYEB_APP_NAME" \
  --env KOYEB_SERVICE_NAME="$SERVICE_NAME" \
  --token "$KOYEB_API_TOKEN"

echo ""
echo "=========================================="
echo "SERVICE CREATED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "The service will:"
echo "  1. Build and deploy your code"
echo "  2. Run: python main.py $MODEL $START $END --koyeb"
echo "  3. Upload results to s3://$S3_BUCKET/koyeb-results/{WORKFLOW_ID}/"
echo "  4. Auto-delete itself to stop billing"
echo ""
echo "Monitor the service:"
echo "  koyeb services get $SERVICE_NAME --app $KOYEB_APP_NAME"
echo ""
echo "View logs in real-time:"
echo "  koyeb services logs $SERVICE_NAME --app $KOYEB_APP_NAME --follow"
echo ""
echo "Download results when complete:"
echo "  aws s3 ls s3://$S3_BUCKET/koyeb-results/"
echo "  aws s3 sync s3://$S3_BUCKET/koyeb-results/YYYYMMDD_HHMMSS/ ./results/"
echo ""
echo "=========================================="
