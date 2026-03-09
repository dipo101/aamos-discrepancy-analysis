#!/bin/bash
# Cleanup Script: Remove all Cloud Run resources
# Use to reset everything or clean up after analysis

set -e

PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
SERVICE_NAME="permutation-worker-v3"
BUCKET_NAME="patient-concordance-permutation-v3"

echo "=========================================="
echo "CLEANUP CLOUD RUN RESOURCES"
echo "=========================================="
echo "  WARNING: This will DELETE:"
echo "  - Cloud Run service: $SERVICE_NAME"
echo "  - GCS bucket: gs://$BUCKET_NAME (optional)"
echo "=========================================="
echo ""

read -p "Delete Cloud Run service? (yes/no): " confirm_service
if [ "$confirm_service" = "yes" ]; then
    echo "  Deleting Cloud Run service..."
    gcloud run services delete $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --quiet || echo "  (Service not found)"
    echo " Service deleted"
fi

read -p "Delete GCS bucket (including all data and results)? (yes/no): " confirm_bucket
if [ "$confirm_bucket" = "yes" ]; then
    echo "  Deleting GCS bucket..."
    gsutil -m rm -r gs://$BUCKET_NAME/ || echo "  (Bucket not found)"
    echo " Bucket deleted"
fi

echo ""
echo " Cleanup complete!"
echo ""

