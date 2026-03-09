#!/bin/bash
# Script 1: Setup GCP Infrastructure
# Run once to initialize your GCP project

set -e

# Load config
PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
BUCKET_NAME="patient-concordance-permutation-v3"

echo "=========================================="
echo "GCP INFRASTRUCTURE SETUP"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo "=========================================="
echo ""

# 1. Set project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
echo ""
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com

# 3. Create GCS bucket
echo ""
echo "Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/ || echo "  (Bucket already exists)"

# 4. Create folder structure
echo ""
echo "Creating GCS folder structure..."
echo "" | gsutil cp - gs://$BUCKET_NAME/data/.keep
echo "" | gsutil cp - gs://$BUCKET_NAME/results/.keep
echo "" | gsutil cp - gs://$BUCKET_NAME/final/.keep

# 5. Set bucket permissions (if needed)
echo ""
echo "Setting bucket permissions..."
# The default service account should have access, but if needed:
# gsutil iam ch serviceAccount:[SERVICE_ACCOUNT]:objectAdmin gs://$BUCKET_NAME

echo ""
echo "GCP infrastructure setup complete!"
