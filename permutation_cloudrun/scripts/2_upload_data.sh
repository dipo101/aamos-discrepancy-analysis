#!/bin/bash
# Script 2: Upload Data Files to GCS
# Run to upload your data files before running the analysis

set -e

BUCKET_NAME="patient-concordance-permutation-v3"
DATA_DIR="../.."  # Parent directory where data files are

echo "=========================================="
echo "UPLOADING DATA TO GCS"
echo "=========================================="
echo "Bucket: gs://$BUCKET_NAME/data/"
echo "=========================================="
echo ""

# Upload patient info
echo " Uploading patient_info..."
gsutil cp $DATA_DIR/anonym_aamos00_patient_info.csv \
    gs://$BUCKET_NAME/data/anonym_aamos00_patient_info.csv

# Upload questionnaire
echo " Uploading questionnaire..."
gsutil cp $DATA_DIR/anonym_aamos00_dailyquestionnaire_dt.csv \
    gs://$BUCKET_NAME/data/anonym_aamos00_dailyquestionnaire_dt.csv

# Upload inhaler
echo " Uploading inhaler..."
gsutil cp $DATA_DIR/anonym_aamos00_smartinhaler_dt.csv \
    gs://$BUCKET_NAME/data/anonym_aamos00_smartinhaler_dt.csv

echo ""
echo " Data upload complete!"
echo ""
echo "Verify upload:"
echo "  gsutil ls gs://$BUCKET_NAME/data/"
echo ""
echo "Next step:"
echo "  Deploy job:    ./scripts/deploy_job.sh"
echo ""

