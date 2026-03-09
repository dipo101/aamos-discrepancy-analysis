"""
Multi-Service Orchestrator: Distribute jobs across 5 Cloud Run workers
This avoids per-service rate limits by spreading load.
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
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = '../config.yaml') -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_service_urls(project_id: str, region: str, base_name: str) -> list:
    """Get URLs for all deployed workers."""
    suffixes = ['a', 'b', 'c', 'd', 'e']
    urls = []
    
    for suffix in suffixes:
        service_name = f"{base_name}-{suffix}"
        url = f"https://{service_name}-vm24s3cl4q-uc.a.run.app"  # Standard Cloud Run URL format
        urls.append({'name': service_name, 'url': url})
    
    return urls


def get_patient_list(config: dict) -> list:
    """Get list of patients from GCS data."""
    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob('data/anonym_aamos00_patient_info.csv')
    import pandas as pd
    patient_df = pd.read_csv(blob.open('r'))
    patients = sorted(patient_df['user_key'].unique())
    
    logger.info(f"Found {len(patients)} patients")
    return [int(p) for p in patients]


def generate_job_batches(config: dict, patients: list) -> list:
    """Generate all job configurations."""
    n_perms = config['analysis']['n_permutations']
    perms_per_batch = config['analysis']['perms_per_batch']
    batches_per_patient = n_perms // perms_per_batch
    
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
    
    return jobs


def invoke_cloud_run(service_url: str, job: dict, timeout: int = 3600) -> dict:
    """Invoke a single Cloud Run instance."""
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
                'duration': result.get('duration_seconds', 0)
            }
        else:
            return {
                'status': 'error',
                'patient_id': job['patient_id'],
                'batch_id': job['batch_id'],
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'status': 'error',
            'patient_id': job['patient_id'],
            'batch_id': job['batch_id'],
            'error': str(e)
        }


def run_multi_service(services: list, jobs: list, max_per_service: int) -> list:
    """
    Distribute jobs across multiple services and run in parallel.
    
    Args:
        services: List of service dicts with 'name' and 'url'
        jobs: List of all jobs
        max_per_service: Max concurrent jobs per service
    """
    n_services = len(services)
    jobs_per_service = len(jobs) // n_services
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MULTI-SERVICE EXECUTION")
    logger.info(f"{'='*80}")
    logger.info(f"Total jobs: {len(jobs)}")
    logger.info(f"Services: {n_services}")
    logger.info(f"Jobs per service: ~{jobs_per_service}")
    logger.info(f"Max concurrent per service: {max_per_service}")
    logger.info(f"Total max concurrent: {n_services * max_per_service}")
    logger.info(f"{'='*80}\n")
    
    # Distribute jobs across services
    service_job_batches = []
    for i, service in enumerate(services):
        start_idx = i * jobs_per_service
        end_idx = start_idx + jobs_per_service if i < n_services - 1 else len(jobs)
        service_jobs = jobs[start_idx:end_idx]
        service_job_batches.append({
            'service': service,
            'jobs': service_jobs
        })
        logger.info(f"{service['name']}: {len(service_jobs)} jobs")
    
    logger.info("")
    
    # Run all services in parallel
    start_time = datetime.now()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=n_services) as service_executor:
        service_futures = {}
        
        for batch in service_job_batches:
            future = service_executor.submit(
                run_service_jobs,
                batch['service'],
                batch['jobs'],
                max_per_service
            )
            service_futures[future] = batch['service']['name']
        
        # Monitor service completion
        for future in as_completed(service_futures):
            service_name = service_futures[future]
            results = future.result()
            all_results.extend(results)
            
            successes = sum(1 for r in results if r['status'] == 'success')
            failures = sum(1 for r in results if r['status'] == 'error')
            
            logger.info(f" {service_name} complete: {successes} success, {failures} failed")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Final summary
    total_success = sum(1 for r in all_results if r['status'] == 'success')
    total_failures = sum(1 for r in all_results if r['status'] == 'error')
    
    logger.info(f"\n{'='*80}")
    logger.info("EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    logger.info(f"Total jobs: {len(all_results)}")
    logger.info(f"Successes: {total_success}")
    logger.info(f"Failures: {total_failures}")
    logger.info(f"{'='*80}\n")
    
    return all_results


def run_service_jobs(service: dict, jobs: list, max_concurrent: int = 10) -> list:
    """Run all jobs for a single service with gradual ramp-up."""
    service_name = service['name']
    service_url = service['url']
    
    logger.info(f" {service_name}: Starting {len(jobs)} jobs...")
    
    results = []
    start_time = datetime.now()
    
    # GRADUAL RAMP-UP to avoid "no available instance" errors
    # Start with 5 concurrent, then increase to max_concurrent
    ramp_up_batches = [
        (5, 2.0),   # 5 jobs, 2 sec delay between jobs
        (10, 1.0),  # 10 jobs, 1 sec delay
        (max_concurrent, 0.5)  # Full capacity, 0.5 sec delay
    ]
    
    job_idx = 0
    for batch_size, delay in ramp_up_batches:
        if job_idx >= len(jobs):
            break
        
        batch_end = min(job_idx + batch_size, len(jobs))
        batch_jobs = jobs[job_idx:batch_end]
        
        with ThreadPoolExecutor(max_workers=len(batch_jobs)) as executor:
            futures = {}
            
            # Launch jobs with delay to avoid overwhelming Cloud Run
            for job in batch_jobs:
                future = executor.submit(invoke_cloud_run, service_url, job)
                futures[future] = job
                time.sleep(delay)
            
            # Collect results - NO TIMEOUT on as_completed (it causes hangs)
            # The timeout is in invoke_cloud_run (3600 sec)
            completed_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()  # Will raise if job failed
                    results.append(result)
                    completed_count += 1
                except Exception as e:
                    job = futures[future]
                    logger.warning(f"  {service_name}: Job failed - Patient {job['patient_id']}, Batch {job['batch_id']}: {e}")
                    results.append({
                        'status': 'error',
                        'patient_id': job['patient_id'],
                        'batch_id': job['batch_id'],
                        'error': str(e)
                    })
                    completed_count += 1
        
        job_idx = batch_end
        
        # Progress update
        if job_idx < len(jobs):
            progress = (job_idx / len(jobs)) * 100
            logger.info(f"  {service_name}: {progress:.0f}% ({job_idx}/{len(jobs)})")
    
    # Process remaining jobs at full capacity
    if job_idx < len(jobs):
        remaining = jobs[job_idx:]
        batch_size = max_concurrent
        
        for i in range(0, len(remaining), batch_size):
            batch_jobs = remaining[i:i+batch_size]
            
            with ThreadPoolExecutor(max_workers=len(batch_jobs)) as executor:
                futures = {
                    executor.submit(invoke_cloud_run, service_url, job): job
                    for job in batch_jobs
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        job = futures[future]
                        logger.warning(f"  {service_name}: Job failed - Patient {job['patient_id']}, Batch {job['batch_id']}: {e}")
                        results.append({
                            'status': 'error',
                            'patient_id': job['patient_id'],
                            'batch_id': job['batch_id'],
                            'error': str(e)
                        })
            
            job_idx += len(batch_jobs)
            if job_idx < len(jobs):
                progress = (job_idx / len(jobs)) * 100
                logger.info(f"  {service_name}: {progress:.0f}% ({job_idx}/{len(jobs)})")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    successes = sum(1 for r in results if r['status'] == 'success')
    
    logger.info(f"  {service_name}: Done in {elapsed/60:.1f} min ({successes}/{len(jobs)} success)")
    
    return results


def save_execution_log(results: list, config: dict):
    """Save execution log to GCS."""
    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"final/execution_log_multi_{timestamp}.json"
    
    blob = bucket.blob(filename)
    blob.upload_from_string(
        json.dumps(results, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"💾 Saved execution log to gs://{bucket_name}/{filename}")


def main():
    """Main orchestration function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-service Cloud Run orchestrator')
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Skip confirmation prompt and proceed automatically')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PERMUTATION ANALYSIS V3 - MULTI-SERVICE ORCHESTRATOR")
    logger.info("="*80 + "\n")
    
    # Load config
    config = load_config()
    project_id = config['gcp']['project_id']
    region = config['gcp']['region']
    
    # Get service URLs
    services = get_service_urls(project_id, region, "permutation-worker-v3")
    
    logger.info(" Services configured:")
    for svc in services:
        logger.info(f"   {svc['name']}: {svc['url']}")
    logger.info("")
    
    # Get patients and generate jobs
    logger.info(" Loading patient list...")
    patients = get_patient_list(config)
    
    logger.info("🔨 Generating job batches...")
    jobs = generate_job_batches(config, patients)
    
    # Confirm
    print("\n" + "="*80)
    print("READY TO LAUNCH")
    print("="*80)
    print(f"Total patients: {len(patients)}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Services: {len(services)}")
    print(f"Jobs per service: ~{len(jobs) // len(services)}")
    print(f"Max concurrent per service: 20")
    print(f"Total max concurrent: {len(services) * 20} (100 total)")
    print(f"Estimated time: ~60 minutes")
    print(f"Estimated cost: ~$15-20")
    print("="*80)
    
    if not args.yes:
        proceed = input("\nProceed with launch? (yes/no): ").strip().lower()
        if proceed != 'yes':
            logger.info(" Aborted by user")
            sys.exit(0)
    else:
        logger.info("\nAuto-confirming launch (--yes flag provided)")
    
    # Run jobs
    results = run_multi_service(services, jobs, max_per_service=20)
    
    # Save log
    save_execution_log(results, config)
    
    logger.info("\nMulti-service orchestration complete!")
    logger.info("   Next step: Run aggregator to combine results")


if __name__ == '__main__':
    main()

