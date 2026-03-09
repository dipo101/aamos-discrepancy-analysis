#!/bin/bash

# Keep 20 instances warm per service to avoid cold starts
# This costs ~$10/hour while running but eliminates ALL startup issues

PROJECT_ID="power-line-monitoring-476216"
REGION="us-central1"
MIN_INSTANCES=20

echo "Setting min-instances=$MIN_INSTANCES for all workers..."
echo ""

for suffix in a b c d e; do
  SERVICE="permutation-worker-v3-$suffix"
  echo "Updating $SERVICE..."
  
  gcloud run services update "$SERVICE" \
    --min-instances=$MIN_INSTANCES \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --quiet
  
  echo " $SERVICE: min-instances set to $MIN_INSTANCES"
  echo ""
done

echo " All services updated!"
echo ""
echo ""
echo "To save costs after analysis, run:"
echo "  gcloud run services update SERVICE --min-instances=0"

