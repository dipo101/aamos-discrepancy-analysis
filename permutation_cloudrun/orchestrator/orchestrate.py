"""
Orchestrator: Launch Cloud Run Permutation Jobs
Manages the execution of 220 Cloud Run invocations in waves.
"""

import yaml
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import requests
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = '../config.yaml') -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_patient_list(config: dict) -> list:
    """Get list of patients from GCS data."""
    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Load patient info from GCS
    blob = bucket.blob('data/anonym_aamos00_patient_info.csv')
    import pandas as pd
    patient_df = pd.read_csv(blob.open('r'))
    
    # Get eligible patients (matching existing criteria)
    # For now, return all unique patient IDs
    patients = sorted(patient_df['user_key'].unique())
    
    logger.info(f"Found {len(patients)} patients: {patients}")
    return [int(p) for p in patients]


def generate_job_batches(config: dict, patients: list) -> list:
    """
    Generate all job configurations.
    
    Returns list of job dicts:
    [{patient_id, batch_id, perm_start, perm_end, ...}, ...]
    """
    n_perms = config['analysis']['n_permutations']
    perms_per_batch = config['analysis']['perms_per_batch']
    batches_per_patient = config['analysis']['batches_per_patient']
    
    jobs = []
    
    for patient_id in patients:
        for batch_id in range(batches_per_patient):
            perm_start = batch_id * perms_per_batch
            perm_end = min(perm_start + perms_per_batch, n_perms)
            
            jobs.append({
                'patient_id': patient_id,
                'batch_id': batch_id,
                'perm_start': perm_start,
                'perm_end': perm_end,
                'random_seed': config['analysis']['random_seed'],
                'correlation_type': config['analysis']['correlation_type'],
                'bucket_name': config['gcp']['bucket_name']
            })
    
    logger.info(f"Generated {len(jobs)} total jobs")
    return jobs


def invoke_cloud_run(
    service_url: str,
    job: dict,
    timeout: int = 3600,
    max_retries: int = 3
) -> dict:
    """
    Invoke a single Cloud Run instance with retry logic.
    
    Returns:
        Dict with status and results
    """
    import time
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{service_url}/process",
                json=job,
                timeout=timeout
            )
        
            if response.status_code == 200:
                result = response.json()
                return {
                    'status': 'success',
                    'patient_id': job['patient_id'],
                    'batch_id': job['batch_id'],
                    'duration': result.get('duration_seconds', 0),
                    'response': result
                }
            elif response.status_code == 429:
                # Rate limit - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    logger.debug(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'status': 'error',
                        'patient_id': job['patient_id'],
                        'batch_id': job['batch_id'],
                        'error': f"HTTP 429 after {max_retries} attempts: Rate exceeded"
                    }
            else:
                return {
                    'status': 'error',
                    'patient_id': job['patient_id'],
                    'batch_id': job['batch_id'],
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.debug(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
            else:
                return {
                    'status': 'error',
                    'patient_id': job['patient_id'],
                    'batch_id': job['batch_id'],
                    'error': str(e)
                }
    
    # Should never reach here
    return {
        'status': 'error',
        'patient_id': job['patient_id'],
        'batch_id': job['batch_id'],
        'error': 'Unknown error after retries'
    }


def run_jobs_in_waves(
    service_url: str,
    jobs: list,
    max_concurrent: int,
    timeout: int
) -> list:
    """
    Run all jobs in waves to respect quota limits.
    
    Args:
        service_url: Cloud Run service URL
        jobs: List of job configurations
        max_concurrent: Max concurrent instances per wave
        timeout: Timeout per job (seconds)
    
    Returns:
        List of results
    """
    total_jobs = len(jobs)
    n_waves = (total_jobs + max_concurrent - 1) // max_concurrent
    
    logger.info(f"\n{'='*80}")
    logger.info(f"LAUNCHING {total_jobs} JOBS IN {n_waves} WAVES")
    logger.info(f"Max concurrent: {max_concurrent}")
    logger.info(f"Timeout per job: {timeout}s ({timeout/60:.1f} min)")
    logger.info(f"{'='*80}\n")
    
    all_results = []
    overall_start = datetime.now()
    
    for wave_idx in range(n_waves):
        wave_start_idx = wave_idx * max_concurrent
        wave_end_idx = min(wave_start_idx + max_concurrent, total_jobs)
        wave_jobs = jobs[wave_start_idx:wave_end_idx]
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"WAVE {wave_idx + 1}/{n_waves}")
        logger.info(f"Jobs: {wave_start_idx + 1}-{wave_end_idx} of {total_jobs}")
        logger.info(f"Concurrent instances: {len(wave_jobs)}")
        logger.info(f"{'─'*80}\n")
        
        wave_start = datetime.now()
        
        # Launch jobs gradually to avoid Cloud Run's startup rate limit (~1-2/sec)
        # Submit jobs one at a time with 1 second delay
        with ThreadPoolExecutor(max_workers=len(wave_jobs)) as executor:
            futures = {}
            for i, job in enumerate(wave_jobs):
                future = executor.submit(invoke_cloud_run, service_url, job, timeout)
                futures[future] = job
                # Small delay between submissions to respect Cloud Run rate limits
                # After first 10 jobs, we can speed up a bit as containers are warm
                if i < 10:
                    time.sleep(1.0)  # 1 job per second initially
                else:
                    time.sleep(0.5)  # 2 jobs per second after warm-up
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                all_results.append(result)
                
                status_emoji = "Y" if result['status'] == 'success' else "N"
                duration = result.get('duration', 0)
                
                if result['status'] == 'error':
                    error_msg = result.get('error', 'Unknown error')
                    # Truncate long errors
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    logger.info(
                        f"  [{completed}/{len(wave_jobs)}] {status_emoji} "
                        f"Patient {result['patient_id']}, Batch {result['batch_id']} | "
                        f"ERROR: {error_msg}"
                    )
                else:
                    logger.info(
                        f"  [{completed}/{len(wave_jobs)}] {status_emoji} "
                        f"Patient {result['patient_id']}, Batch {result['batch_id']} | "
                        f"{duration:.1f}s"
                    )
        
        wave_elapsed = (datetime.now() - wave_start).total_seconds()
        
        # Count successes/failures for this wave
        wave_successes = sum(1 for r in all_results[-len(wave_jobs):] if r['status'] == 'success')
        wave_failures = sum(1 for r in all_results[-len(wave_jobs):] if r['status'] == 'error')
        
        logger.info(f"\n Wave {wave_idx + 1} complete in {wave_elapsed/60:.1f} minutes")
        logger.info(f"   Success: {wave_successes}, Failed: {wave_failures}")
        
        # If many failures, show first unique error
        if wave_failures > 0:
            error_msgs = [r.get('error', '') for r in all_results[-len(wave_jobs):] if r['status'] == 'error']
            if error_msgs:
                first_error = error_msgs[0]
                if len(first_error) > 200:
                    first_error = first_error[:200] + "..."
                logger.info(f"   First error: {first_error}")
        
        # Brief pause between waves
        if wave_idx < n_waves - 1:
            logger.info("Pausing 10 seconds before next wave...")
            time.sleep(10)
    
    overall_elapsed = (datetime.now() - overall_start).total_seconds()
    
    # Summary
    successes = sum(1 for r in all_results if r['status'] == 'success')
    failures = sum(1 for r in all_results if r['status'] == 'error')
    
    logger.info(f"\n{'='*80}")
    logger.info("EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.2f} hours)")
    logger.info(f"Total jobs: {total_jobs}")
    logger.info(f" Successes: {successes}")
    logger.info(f" Failures: {failures}")
    logger.info(f"{'='*80}\n")
    
    return all_results


def save_execution_log(results: list, config: dict):
    """Save execution log to GCS."""
    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"final/execution_log_{timestamp}.json"
    
    blob = bucket.blob(filename)
    blob.upload_from_string(
        json.dumps(results, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"💾 Saved execution log to gs://{bucket_name}/{filename}")


def main():
    """Main orchestration function."""
    logger.info("="*80)
    logger.info("PERMUTATION ANALYSIS V3 - CLOUD RUN ORCHESTRATOR")
    logger.info("="*80 + "\n")
    
    # Load config
    config = load_config()
    logger.info(f" Configuration loaded")
    logger.info(f"   Project: {config['gcp']['project_id']}")
    logger.info(f"   Region: {config['gcp']['region']}")
    logger.info(f"   Bucket: {config['gcp']['bucket_name']}")
    logger.info(f"   Permutations: {config['analysis']['n_permutations']:,}")
    logger.info(f"   Perms per batch: {config['analysis']['perms_per_batch']}")
    
    # Get Cloud Run service URL
    service_name = config['cloud_run']['service_name']
    project_id = config['gcp']['project_id']
    region = config['gcp']['region']
    
    # User must provide the service URL after deployment
    print("\n" + "="*80)
    print("⚠️  IMPORTANT: You need the Cloud Run service URL")
    print("="*80)
    service_url = input("\nEnter Cloud Run service URL (or press Enter to use default): ").strip()
    
    if not service_url:
        service_url = f"https://{service_name}-{region}-{project_id}.run.app"
        print(f"Using default URL: {service_url}")
    
    print()
    
    # Get patient list
    logger.info(" Loading patient list...")
    patients = get_patient_list(config)
    
    # Generate job batches
    logger.info(" Generating job batches...")
    jobs = generate_job_batches(config, patients)
    
    # Confirm before proceeding
    print("\n" + "="*80)
    print("READY TO LAUNCH")
    print("="*80)
    print(f"Total patients: {len(patients)}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Max concurrent: {config['cloud_run']['max_instances']}")
    print(f"Estimated time: ~{(len(jobs) / config['cloud_run']['max_instances']) * 33 / 60:.1f} hours")
    print(f"Estimated cost: ~${len(jobs) * 0.14:.2f}")
    print("="*80)
    
    proceed = input("\nProceed with launch? (yes/no): ").strip().lower()
    if proceed != 'yes':
        logger.info(" Aborted by user")
        sys.exit(0)
    
    # Run jobs
    results = run_jobs_in_waves(
        service_url,
        jobs,
        config['cloud_run']['max_instances'],
        config['cloud_run']['timeout']
    )
    
    # Save execution log
    save_execution_log(results, config)
    
    logger.info("\n Orchestration complete!")
    logger.info("   Next step: Run aggregator to combine results")


if __name__ == '__main__':
    main()

