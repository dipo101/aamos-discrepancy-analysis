"""
Working Orchestrator - 10 concurrent jobs at a time
Stays within CPU quota and rate limits
"""
import sys
import yaml
import requests
import time
from pathlib import Path
from datetime import datetime
import concurrent.futures
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from orchestrate_resume import (
    load_config, get_completed_jobs, filter_jobs, 
    generate_job_batches, get_patient_list
)
from orchestrate_multi import get_service_urls

def invoke_job(service_url: str, job: dict) -> dict:
    """Invoke one job with timeout."""
    patient_id = job['patient_id']
    batch_id = job['batch_id']
    
    logger.info(f" LAUNCHING: Patient {patient_id:3d} Batch {batch_id:2d} -> {service_url[:50]}...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{service_url}/process",
            json=job,
            timeout=900  # 15 min
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            n_valid = result.get('n_valid', 0)
            logger.info(f" Patient {patient_id:3d} Batch {batch_id:2d} - {n_valid:3d} valid ({elapsed:.1f}s)")
            return {'status': 'success', 'patient_id': patient_id, 'batch_id': batch_id, 'n_valid': n_valid}
        else:
            logger.error(f" Patient {patient_id:3d} Batch {batch_id:2d} - HTTP {response.status_code} ({elapsed:.1f}s)")
            logger.error(f"   Response: {response.text[:200]}")
            return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id}
    
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start_time
        logger.error(f" TIMEOUT: Patient {patient_id:3d} Batch {batch_id:2d} after {elapsed:.1f}s")
        return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id}
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start_time
        logger.error(f" CONNECTION ERROR: Patient {patient_id:3d} Batch {batch_id:2d} - {str(e)[:100]}")
        return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f" Patient {patient_id:3d} Batch {batch_id:2d} - {type(e).__name__}: {str(e)[:100]} ({elapsed:.1f}s)")
        return {'status': 'error', 'patient_id': patient_id, 'batch_id': batch_id}


def main():
    logger.info("="*80)
    logger.info("WORKING ORCHESTRATOR - 10 CONCURRENT JOBS AT A TIME")
    logger.info("="*80)
    
    # Load config
    config = load_config()
    services = get_service_urls(
        config['gcp']['project_id'],
        config['gcp']['region'],
        "permutation-worker-v3"
    )
    
    logger.info(f" {len(services)} services ready")
    
    # Get remaining jobs
    patients = get_patient_list(config)
    all_jobs = generate_job_batches(config, patients)
    completed = get_completed_jobs(config['gcp']['bucket_name'])
    remaining = filter_jobs(all_jobs, completed)
    
    logger.info(f" Total: {len(all_jobs)} jobs")
    logger.info(f" Completed: {len(completed)} jobs")
    logger.info(f" Remaining: {len(remaining)} jobs")
    
    if len(remaining) == 0:
        logger.info(" All done!")
        return
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"PROCESSING {len(remaining)} JOBS IN BATCHES OF 10")
    logger.info("="*80)
    logger.info("")
    
    # Process in batches of 10
    BATCH_SIZE = 10
    results = []
    
    for batch_num in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_num:batch_num + BATCH_SIZE]
        batch_id = batch_num // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f" BATCH {batch_id}/{total_batches} - Processing {len(batch)} jobs")
        logger.info(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Launch batch
        batch_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for i, job in enumerate(batch):
                service = services[i % len(services)]
                future = executor.submit(invoke_job, service['url'], job)
                futures[future] = job
            
            logger.info(f"   Launched {len(batch)} requests, waiting for responses...")
            
            # Wait for batch to complete with timeout
            try:
                for future in concurrent.futures.as_completed(futures, timeout=1800):  # 30 min max per batch
                    try:
                        result = future.result(timeout=10)  # 10 sec to get result once complete
                        results.append(result)
                    except Exception as e:
                        job = futures[future]
                        logger.error(f"Failed: Patient {job['patient_id']} Batch {job['batch_id']}: {e}")
                        results.append({'status': 'error', 'patient_id': job['patient_id'], 'batch_id': job['batch_id']})
            except concurrent.futures.TimeoutError:
                logger.error(f" BATCH {batch_id} TIMEOUT - Some jobs took >30min")
                # Mark remaining jobs as errors
                for future, job in futures.items():
                    if not future.done():
                        logger.error(f"  Incomplete: Patient {job['patient_id']} Batch {job['batch_id']}")
                        results.append({'status': 'error', 'patient_id': job['patient_id'], 'batch_id': job['batch_id']})
        
        # Progress
        batch_elapsed = time.time() - batch_start
        completed_count = len(results)
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"    Batch {batch_id} complete in {batch_elapsed:.1f}s")
        logger.info(f" Progress: {completed_count}/{len(remaining)} ({completed_count*100//len(remaining)}%) - {success_count} success")
        logger.info("")
        
        # Small delay between batches to avoid overwhelming Cloud Run
        if batch_num + BATCH_SIZE < len(remaining):
            time.sleep(2)
    
    logger.info("="*80)
    logger.info(" ALL JOBS COMPLETE!")
    logger.info("="*80)
    
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes
    logger.info(f" Successes: {successes}")
    logger.info(f" Failures: {failures}")


if __name__ == '__main__':
    main()

