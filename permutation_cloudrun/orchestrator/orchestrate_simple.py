"""
Simple Sequential Orchestrator with extensive logging
No ThreadPoolExecutor, no complexity, just clear sequential execution
"""
import sys
import yaml
import requests
import time
from pathlib import Path
from datetime import datetime
from google.cloud import storage
import logging

# DETAILED logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from orchestrate_resume import (
    load_config, get_completed_jobs, filter_jobs, 
    generate_job_batches, get_patient_list
)

def invoke_job_with_logging(service_url: str, job: dict, timeout: int = 1200) -> dict:
    """Invoke ONE job with detailed logging."""
    patient_id = job['patient_id']
    batch_id = job['batch_id']
    
    logger.info(f" LAUNCHING: Patient {patient_id}, Batch {batch_id} -> {service_url}")
    
    start = time.time()
    try:
        response = requests.post(
            f"{service_url}/process",
            json=job,
            timeout=timeout
        )
        duration = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            n_valid = result.get('n_valid', 0)
            logger.info(f" SUCCESS: Patient {patient_id}, Batch {batch_id} - {duration:.1f}s - {n_valid} valid")
            return {'status': 'success', 'patient_id': patient_id, 'batch_id': batch_id, 'n_valid': n_valid}
        else:
            logger.error(f" HTTP {response.status_code}: Patient {patient_id}, Batch {batch_id}")
            return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id, 'error': f"HTTP {response.status_code}"}
    
    except requests.exceptions.Timeout:
        logger.error(f" TIMEOUT: Patient {patient_id}, Batch {batch_id} after {timeout}s")
        return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id, 'error': 'timeout'}
    
    except Exception as e:
        logger.error(f" EXCEPTION: Patient {patient_id}, Batch {batch_id}: {e}")
        return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id, 'error': str(e)}


def main():
    logger.info("="*80)
    logger.info("SIMPLE SEQUENTIAL ORCHESTRATOR - WITH EXTENSIVE LOGGING")
    logger.info("="*80)
    logger.info("")
    
    # Load config
    config = load_config()
    project_id = config['gcp']['project_id']
    region = config['gcp']['region']
    bucket_name = config['gcp']['bucket_name']
    
    # Get services
    from orchestrate_multi import get_service_urls
    services = get_service_urls(project_id, region, "permutation-worker-v3")
    
    logger.info(" Services:")
    for svc in services:
        logger.info(f"   {svc['name']}: {svc['url']}")
    logger.info("")
    
    # Get remaining jobs
    logger.info(" Loading patient list...")
    patients = get_patient_list(config)
    logger.info(f"Found {len(patients)} patients")
    
    logger.info(" Generating all job batches...")
    all_jobs = generate_job_batches(config, patients)
    logger.info(f"Total jobs: {len(all_jobs)}")
    
    logger.info(" Checking for completed jobs...")
    completed = get_completed_jobs(bucket_name)
    logger.info(f"Completed: {len(completed)}")
    
    remaining = filter_jobs(all_jobs, completed)
    logger.info(f"Remaining: {len(remaining)}")
    
    if len(remaining) == 0:
        logger.info("\n All jobs already completed!")
        return
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"LAUNCHING {len(remaining)} JOBS SEQUENTIALLY")
    logger.info("="*80)
    logger.info("")
    
    # Launch sequentially with 100 concurrent (simple batching)
    results = []
    batch_size = 100
    
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        logger.info(f"\n BATCH {batch_start//batch_size + 1}: Processing {len(batch)} jobs")
        logger.info("="*80)
        
        # Launch all jobs in this batch
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # Submit all
            futures = {}
            for i, job in enumerate(batch):
                service = services[i % len(services)]
                future = executor.submit(invoke_job_with_logging, service['url'], job)
                futures[future] = job
                logger.info(f"  [{i+1}/{len(batch)}] Submitted: Patient {job['patient_id']}, Batch {job['batch_id']}")
            
            # Wait for all with timeout
            logger.info(f"\n Waiting for {len(batch)} jobs to complete (max 20 min each)...")
            for i, future in enumerate(concurrent.futures.as_completed(futures, timeout=1200)):
                try:
                    result = future.result()
                    results.append(result)
                    job = futures[future]
                    logger.info(f"  [{i+1}/{len(batch)}] Completed: Patient {job['patient_id']}, Batch {job['batch_id']}")
                except Exception as e:
                    job = futures[future]
                    logger.error(f"  [{i+1}/{len(batch)}] FAILED: Patient {job['patient_id']}, Batch {job['batch_id']}: {e}")
                    results.append({'status': 'error', 'patient_id': job['patient_id'], 'batch_id': job['batch_id'], 'error': str(e)})
        
        # Progress
        successes = sum(1 for r in results if r['status'] == 'success')
        failures = len(results) - successes
        logger.info(f"\n Progress: {len(results)}/{len(remaining)} -  {successes} success,  {failures} failed")
    
    logger.info("\n" + "="*80)
    logger.info(" ALL JOBS COMPLETE!")
    logger.info("="*80)
    
    successes = sum(1 for r in results if r['status'] == 'success')
    logger.info(f" Successes: {successes}")
    logger.info(f" Failures: {len(results) - successes}")

if __name__ == '__main__':
    main()

