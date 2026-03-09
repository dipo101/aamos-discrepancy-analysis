#!/bin/bash
# Test Worker Locally: Build and run Docker container on your machine
# This lets you verify the worker logic before deploying to Cloud Run

set -e

echo "=========================================="
echo "LOCAL DOCKER BUILD & TEST"
echo "=========================================="
echo ""

cd ../worker

echo " Building Docker image..."
docker build -t permutation-worker-local:latest .

echo ""
echo " Build complete!"
echo ""
echo " Starting local container..."
echo "   The worker will be available at: http://localhost:8080"
echo ""
echo "   Test health endpoint:"
echo "   curl http://localhost:8080/health"
echo ""
echo "   To stop: Press Ctrl+C"
echo ""

# Run container with Google Cloud credentials mounted
# This allows the local container to access GCS
docker run -p 8080:8080 \
    -e GOOGLE_CLOUD_PROJECT=power-line-monitoring-476216 \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/key.json \
    -v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/key.json:ro \
    permutation-worker-local:latest

