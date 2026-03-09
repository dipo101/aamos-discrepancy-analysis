#!/bin/bash
# Deploy Multiple Cloud Run Workers for Parallel Execution
# This deploys 5 workers to distribute load and avoid rate limits

set -e

PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
BASE_SERVICE_NAME="permutation-worker-v3"

# Deploy 5 workers (suffixes: a, b, c, d, e)
WORKERS=("a" "b" "c" "d" "e")

echo "=========================================="
echo "DEPLOYING 5 CLOUD RUN WORKERS"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Base name: $BASE_SERVICE_NAME"
echo "Workers: ${WORKERS[@]}"
echo "=========================================="
echo ""

cd ../worker

for suffix in "${WORKERS[@]}"; do
    SERVICE_NAME="${BASE_SERVICE_NAME}-${suffix}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Deploying: $SERVICE_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --region=$REGION \
        --project=$PROJECT_ID \
        --platform=managed \
        --allow-unauthenticated \
        --cpu=2 \
        --memory=2Gi \
        --timeout=3600 \
        --max-instances=20 \
        --min-instances=0 \
        --concurrency=1 \
        --quiet
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format='value(status.url)')
    
    echo " Deployed: $SERVICE_URL"
done

echo ""
echo "=========================================="
echo " ALL 5 WORKERS DEPLOYED"
echo "=========================================="
echo ""
echo "Service URLs:"
for suffix in "${WORKERS[@]}"; do
    SERVICE_NAME="${BASE_SERVICE_NAME}-${suffix}"
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format='value(status.url)' 2>/dev/null || echo "N/A")
    echo "  $SERVICE_NAME: $SERVICE_URL"
done

echo ""
echo "Next step:"
echo "  cd ../orchestrator && python orchestrate_multi.py"
echo ""

