"""
Resume orchestrator - only runs jobs that don't have results yet.
"""
import sys
import yaml
from pathlib import Path

# Add parent directory to path to import orchestrate_multi
sys.path.insert(0, str(Path(__file__).parent))

from orchestrate_multi import (
    load_config, get_service_urls, get_patient_list,
    generate_job_batches, run_multi_service, save_execution_log
)
from google.cloud import storage
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_completed_jobs(bucket_name: str) -> set:
    """Get set of completed job IDs from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='results/')
    
    completed = set()
    for blob in blobs:
        if 'patient_' in blob.name and '_batch_' in blob.name:
            # Extract patient_id and batch_id from filename
            # Format: results/patient_XXX_batch_YY.json
            parts = blob.name.split('/')[-1].replace('.json', '').split('_')
            if len(parts) >= 4:
                patient_id = int(parts[1])
                batch_id = int(parts[3])
                completed.add((patient_id, batch_id))
    
    logger.info(f"Found {len(completed)} completed jobs in GCS")
    return completed

def filter_jobs(all_jobs: list, completed: set) -> list:
    """Filter out jobs that are already completed."""
    remaining = []
    for job in all_jobs:
        job_id = (job['patient_id'], job['batch_id'])
        if job_id not in completed:
            remaining.append(job)
    
    return remaining

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Resume orchestrator')
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Skip confirmation prompt')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PERMUTATION ANALYSIS V3 - RESUME ORCHESTRATOR")
    logger.info("="*80 + "\n")
    
    # Load config
    config = load_config()
    project_id = config['gcp']['project_id']
    region = config['gcp']['region']
    bucket_name = config['gcp']['bucket_name']
    
    # Get service URLs
    services = get_service_urls(project_id, region, "permutation-worker-v3")
    
    logger.info(" Services configured:")
    for svc in services:
        logger.info(f"   {svc['name']}: {svc['url']}")
    logger.info("")
    
    # Get all jobs
    logger.info(" Loading patient list...")
    patients = get_patient_list(config)
    
    logger.info(" Generating all job batches...")
    all_jobs = generate_job_batches(config, patients)
    
    # Get completed jobs
    logger.info(" Checking for completed jobs...")
    completed = get_completed_jobs(bucket_name)
    
    # Filter to get remaining jobs
    remaining_jobs = filter_jobs(all_jobs, completed)
    
    # Summary
    print("\n" + "="*80)
    print("RESUME SUMMARY")
    print("="*80)
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining to run: {len(remaining_jobs)}")
    print("="*80)
    
    if len(remaining_jobs) == 0:
        logger.info("\n All jobs already completed!")
        return
    
    if not args.yes:
        proceed = input("\nProceed with remaining jobs? (yes/no): ").strip().lower()
        if proceed != 'yes':
            logger.info(" Aborted by user")
            return
    else:
        logger.info("\n Auto-confirming launch (--yes flag provided)")
    
    # Distribute remaining jobs across services
    logger.info("\n Launching remaining jobs...")
    results = run_multi_service(services, remaining_jobs, max_per_service=20)
    
    # Save log
    save_execution_log(results, config)
    
    logger.info("\n Resume orchestration complete!")

if __name__ == '__main__':
    main()

