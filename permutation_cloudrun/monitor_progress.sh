#!/bin/bash

# Monitor Cloud Run Permutation Analysis Progress

echo "=========================================="
echo "PERMUTATION ANALYSIS V3 - PROGRESS MONITOR"
echo "=========================================="
echo ""

BUCKET="patient-concordance-permutation-v3"

echo "Checking GCS for completed patient results..."
echo ""

# Count completed patients
COMPLETED=$(gsutil ls "gs://${BUCKET}/permutations/" 2>/dev/null | grep -c "patient_" || echo "0")
TOTAL=22

echo "Patients completed: ${COMPLETED}/${TOTAL}"

if [ "$COMPLETED" -gt 0 ] 2>/dev/null; then
    echo ""
    echo " Completed patients:"
    gsutil ls "gs://${BUCKET}/permutations/" 2>/dev/null | grep "patient_" | while read line; do
        patient_id=$(echo $line | sed 's/.*patient_//' | sed 's/_permutations.*//')
        echo "   Patient ${patient_id}"
    done
fi

echo ""
echo "==========================================
"
echo " To check detailed logs:"
echo "   tail -f /tmp/orchestrate_log_final.txt"
echo ""
echo " To check Cloud Run logs:"
echo "   gcloud logging read 'resource.type=cloud_run_revision' \\"
echo "     --project power-line-monitoring-476216 \\"
echo "     --limit 50 \\"
echo "     --freshness 5m"
echo ""
echo " Script to update progress:"
echo "   ./permutation_cloudrun/monitor_progress.sh"
echo ""

