#!/bin/bash
# Script 3: Build and Deploy Cloud Run Worker
# Run to deploy the permutation worker service

set -e

PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
SERVICE_NAME="permutation-worker-v3"

echo "=========================================="
echo "DEPLOYING CLOUD RUN WORKER"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "=========================================="
echo ""

cd ../worker

echo " Building and deploying..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region=$REGION \
    --project=$PROJECT_ID \
    --platform=managed \
    --allow-unauthenticated \
    --cpu=2 \
    --memory=2Gi \
    --timeout=3600 \
    --max-instances=100 \
    --min-instances=0 \
    --concurrency=1

echo ""
echo " Deployment complete!"
echo ""
echo "Get service URL:"
echo "  gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format='value(status.url)'"
echo ""
echo "Test health endpoint:"
echo "  curl \$(gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format='value(status.url)')/health"
echo ""
echo "Next step:"
echo "  Run orchestrator: cd ../orchestrator && python orchestrate.py"
echo ""

