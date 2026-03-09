"""
Multi-Task Orchestrator for Cloud Run Jobs
Submits one job execution per patient with multiple tasks running in parallel
"""
import sys
import subprocess
import time
import logging
from pathlib import Path
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "power-line-monitoring-476216"
REGION = "us-central1"
JOB_NAME = "permutation-job-v3"
BUCKET_NAME = "patient-concordance-permutation-v3"

# Analysis parameters - CONFIGURED FOR FULL RUN
TOTAL_PERMS = 10000  # Full 10,000 permutations per patient
PERMS_PER_TASK = 75  # Safe for 1-hour timeout (75 × 39s = 48.75 min)
TASKS_PER_PATIENT = (TOTAL_PERMS + PERMS_PER_TASK - 1) // PERMS_PER_TASK  # 134 tasks

# Dataset collection - set to 'true' to save full categorized datasets as Parquet
# This increases storage costs (~20-30 GB total) and adds ~5-10s per task
COLLECT_DATASETS = 'true'  # Change to 'true' for sensitivity analysis

# All 22 patients
ALL_PATIENTS = [113, 190, 217, 278, 294, 328, 343, 398, 447, 454, 473, 514, 
                562, 625, 701, 702, 748, 764, 808, 867, 917, 939]

# Problematic patients (0 valid correlations - can skip if desired)
PROBLEMATIC_PATIENTS = [217, 278, 764, 808, 867]

# Default to all patients (set SKIP_PROBLEMATIC=True to exclude problem patients)
SKIP_PROBLEMATIC = False
PATIENTS = [p for p in ALL_PATIENTS if p not in PROBLEMATIC_PATIENTS] if SKIP_PROBLEMATIC else ALL_PATIENTS


def get_completed_patients() -> set:
    """Check GCS for patients that already have complete results."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix='results/')
    
    # Count batches per patient
    patient_batches = {}
    for blob in blobs:
        if 'patient_' in blob.name and '_batch_' in blob.name:
            # Extract patient_id and batch_id from filename
            parts = blob.name.split('/')[-1].replace('.json', '').split('_')
            if len(parts) >= 4:
                patient_id = int(parts[1])
                if patient_id not in patient_batches:
                    patient_batches[patient_id] = set()
                batch_id = int(parts[3])
                patient_batches[patient_id].add(batch_id)
    
    # Check which patients have all tasks complete
    completed = set()
    for patient_id, batches in patient_batches.items():
        if len(batches) >= TASKS_PER_PATIENT:
            completed.add(patient_id)
            logger.info(f"   Patient {patient_id}: {len(batches)}/{TASKS_PER_PATIENT} tasks complete")
        elif len(batches) > 0:
            logger.info(f"   Patient {patient_id}: {len(batches)}/{TASKS_PER_PATIENT} tasks complete (will re-run)")
    
    return completed


def submit_patient_job(patient_id: int) -> bool:
    """
    Submit a multi-task job execution for one patient.
    Cloud Run will automatically create TASKS_PER_PATIENT tasks.
    """
    logger.info(f"\nPatient {patient_id}: Submitting job with {TASKS_PER_PATIENT} tasks...")
    
    # Build environment variables
    env_vars = f"PATIENT_ID={patient_id},"
    env_vars += f"TOTAL_PERMS={TOTAL_PERMS},"
    env_vars += f"PERMS_PER_TASK={PERMS_PER_TASK},"
    env_vars += f"RANDOM_SEED=42,"
    env_vars += f"CORRELATION_TYPE=spearman,"
    env_vars += f"BUCKET_NAME={BUCKET_NAME},"
    env_vars += f"COLLECT_DATASETS={COLLECT_DATASETS}"
    
    cmd = [
        'gcloud', 'run', 'jobs', 'execute', JOB_NAME,
        '--project', PROJECT_ID,
        '--region', REGION,
        '--tasks', str(TASKS_PER_PATIENT),  # Create multiple tasks
        '--update-env-vars', env_vars,
        '--async',  # Don't wait for completion
        '--format', 'value(metadata.name)'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            execution_name = result.stdout.strip()
            logger.info(f"   Submitted: {execution_name}")
            logger.info(f"   {TASKS_PER_PATIENT} tasks will process {TOTAL_PERMS} permutations")
            return True
        else:
            logger.error(f"   Failed: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"   Error: {e}")
        return False


def main():
    logger.info("="*80)
    logger.info("MULTI-TASK CLOUD RUN JOBS ORCHESTRATOR")
    logger.info("="*80)
    logger.info("")
    logger.info(f" Configuration:")
    logger.info(f"   Total patients: {len(ALL_PATIENTS)}")
    if SKIP_PROBLEMATIC:
        logger.info(f"   Problematic patients (skipped): {len(PROBLEMATIC_PATIENTS)} - {PROBLEMATIC_PATIENTS}")
        logger.info(f"   Valid patients to process: {len(PATIENTS)}")
    else:
        logger.info(f"   Processing all patients: {len(PATIENTS)}")
    logger.info(f"   Permutations per patient: {TOTAL_PERMS:,}")
    logger.info(f"   Permutations per task: {PERMS_PER_TASK}")
    logger.info(f"   Tasks per patient: {TASKS_PER_PATIENT}")
    logger.info(f"   Total tasks: {len(PATIENTS) * TASKS_PER_PATIENT:,}")
    logger.info("")
    logger.info(f"  Resources:")
    logger.info(f"   Max parallel tasks: 500 (parallelism setting)")
    logger.info(f"   CPU per task: 2 vCPU (500 parallel × 2 = 1,000 vCPU)")
    logger.info(f"   Memory per task: 4GB (500 parallel × 4GB = 2,000 GB)")
    logger.info(f"   Timeout per task: 60 minutes")
    logger.info(f"   Region: us-central1")
    logger.info("")
    
    # Calculate estimates
    total_tasks = len(PATIENTS) * TASKS_PER_PATIENT
    time_per_task_min = (PERMS_PER_TASK * 39) / 60  # 39s per permutation
    parallelism = 500
    rounds = (total_tasks + parallelism - 1) // parallelism  # Ceiling division
    est_time_hours = rounds * time_per_task_min / 60
    
    logger.info(f"Time Estimates:")
    logger.info(f"   Time per task: ~{time_per_task_min:.1f} minutes")
    logger.info(f"   Processing rounds: ~{rounds} (with {parallelism} parallel)")
    logger.info(f"   Estimated total time: ~{est_time_hours:.1f} hours")
    logger.info("")
    logger.info("="*80)
    
    # Check for already-completed patients
    logger.info("\n Checking for completed patients in GCS...")
    completed_patients = get_completed_patients()
    
    # Filter out completed patients
    patients_to_run = [p for p in PATIENTS if p not in completed_patients]
    
    if len(completed_patients) > 0:
        logger.info(f"\n Found {len(completed_patients)} patients with complete results - SKIPPING")
    
    if len(patients_to_run) == 0:
        logger.info("\n All patients already have complete results!")
        logger.info("   Nothing to do.")
        return
    
    logger.info(f"\n Summary:")
    logger.info(f"   Total patients: {len(PATIENTS)}")
    logger.info(f"   Already complete: {len(completed_patients)}")
    logger.info(f"   To process: {len(patients_to_run)}")
    logger.info(f"   Total tasks to submit: {len(patients_to_run) * TASKS_PER_PATIENT}")
    
    # Confirm
    response = input(f"\nSubmit {len(patients_to_run)} job executions? (yes/no): ").strip().lower()
    if response != 'yes':
        logger.info(" Aborted")
        return
    
    logger.info("\n Submitting job executions...")
    logger.info("="*80)
    
    successes = 0
    failures = 0
    
    for patient_id in patients_to_run:
        success = submit_patient_job(patient_id)
        if success:
            successes += 1
        else:
            failures += 1
        
        # Small delay between submissions
        time.sleep(1)
    
    logger.info("\n" + "="*80)
    logger.info(" SUBMISSION COMPLETE")
    logger.info("="*80)
    logger.info(f"   Successfully submitted: {successes}/{len(PATIENTS)} patients")
    logger.info(f"   Failed: {failures}")
    logger.info(f"   Total tasks queued: {successes * TASKS_PER_PATIENT}")
    logger.info("")
    logger.info(" Monitor progress:")
    logger.info(f"   gcloud run jobs executions list {JOB_NAME} --project {PROJECT_ID} --region {REGION}")
    logger.info("")
    logger.info(" Check results:")
    logger.info(f"   gsutil ls -l gs://{BUCKET_NAME}/results/ | wc -l")
    logger.info("")


if __name__ == '__main__':
    main()

