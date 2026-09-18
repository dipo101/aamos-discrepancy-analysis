#!/bin/bash
set -e

# Configuration
PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
JOB_NAME="permutation-job-v3"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}"
BUCKET_NAME="patient-concordance-permutation-v3"

echo "========================================"
echo "Deploying Cloud Run JOB"
echo "========================================"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job: $JOB_NAME"
echo "Image: $IMAGE_NAME"
echo ""

# Navigate to job-worker directory
cd "$(dirname "$0")/../job-worker"

# Stage the shared analysis package into the build context. It is the single
# implementation of the join/categorisation/permutation logic and lives at the
# repository root; the copy here is build-only and git-ignored.
REPO_ROOT="$(cd ../.. && pwd)"
rm -rf ./aamos_concordance
cp -R "$REPO_ROOT/aamos_concordance" ./aamos_concordance
find ./aamos_concordance -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
# Stamp the commit into the package so results carry provenance without git in the container.
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_DIRTY="False"; [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ] && GIT_DIRTY="True"
printf 'GIT_COMMIT = "%s"\nGIT_DIRTY = %s\n' "$GIT_COMMIT" "$GIT_DIRTY" > ./aamos_concordance/_build_info.py
trap 'rm -rf ./aamos_concordance' EXIT

# Build and push container
echo "Building container..."
gcloud builds submit \
  --project $PROJECT_ID \
  --tag $IMAGE_NAME

echo ""
echo "Creating/Updating Cloud Run Job..."
gcloud run jobs create $JOB_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --project $PROJECT_ID \
  --set-env-vars BUCKET_NAME=$BUCKET_NAME \
  --set-env-vars RANDOM_SEED=42 \
  --set-env-vars CORRELATION_TYPE=spearman \
  --memory 4Gi \
  --cpu 2 \
  --max-retries 0 \
  --task-timeout 3600s \
  --parallelism 100 \
  2>/dev/null || gcloud run jobs update $JOB_NAME \
    --image $IMAGE_NAME \
    --region $REGION \
    --project $PROJECT_ID \
    --set-env-vars BUCKET_NAME=$BUCKET_NAME \
    --set-env-vars RANDOM_SEED=42 \
    --set-env-vars CORRELATION_TYPE=spearman \
    --memory 4Gi \
    --cpu 2 \
    --max-retries 0 \
    --task-timeout 3600s \
    --parallelism 100

echo ""
echo "========================================"
echo " Deployment complete!"
echo "========================================"
echo ""
echo "Job ready to execute. Use orchestrator to submit tasks."

