"""
Cloud Run JOBS Orchestrator
Submits batch jobs instead of HTTP requests - much more reliable!
"""
import yaml
import subprocess
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from google.cloud import storage
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


def submit_job_execution(project_id: str, region: str, job_name: str, 
                        env_vars: dict, task_name: str) -> bool:
    """Submit a single task execution to Cloud Run Job."""
    # Build environment variable string
    env_str = ','.join([f"{k}={v}" for k, v in env_vars.items()])
    
    cmd = [
        'gcloud', 'run', 'jobs', 'execute', job_name,
        '--project', project_id,
        '--region', region,
        '--update-env-vars', env_str,
        '--async',  # Don't wait for execution to complete
        '--format', 'value(metadata.name)'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10  # Should be instant with --async
        )
        
        if result.returncode == 0:
            execution_name = result.stdout.strip()
            logger.info(f"   Submitted: {task_name} -> {execution_name}")
            return True
        else:
            logger.error(f"   Failed to submit {task_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"   Timeout submitting {task_name}")
        return False
    except Exception as e:
        logger.error(f"   Error submitting {task_name}: {e}")
        return False


def main():
    logger.info("="*80)
    logger.info("CLOUD RUN JOBS ORCHESTRATOR")
    logger.info("="*80)
    
    # Load config
    config = load_config()
    project_id = config['gcp']['project_id']
    region = config['gcp']['region']
    bucket_name = config['gcp']['bucket_name']
    job_name = "permutation-job-v3"
    
    # Get remaining jobs
    patients = get_patient_list(config)
    all_jobs = generate_job_batches(config, patients)
    completed = get_completed_jobs(bucket_name)
    remaining = filter_jobs(all_jobs, completed)
    
    logger.info(f" Total: {len(all_jobs)} jobs")
    logger.info(f" Completed: {len(completed)} jobs")
    logger.info(f" Remaining: {len(remaining)} jobs")
    
    if len(remaining) == 0:
        logger.info("All done!")
        return
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"SUBMITTING {len(remaining)} JOBS")
    logger.info("="*80)
    logger.info("")
    
    # Submit jobs in batches to avoid overwhelming the API
    BATCH_SIZE = 20
    submitted_count = 0
    failed_count = 0
    
    for batch_num in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_num:batch_num + BATCH_SIZE]
        batch_id = batch_num // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f" BATCH {batch_id}/{total_batches} - Submitting {len(batch)} jobs")
        
        for job in batch:
            env_vars = {
                'PATIENT_ID': str(job['patient_id']),
                'BATCH_ID': str(job['batch_id']),
                'PERM_START': str(job['perm_start']),
                'PERM_END': str(job['perm_end']),
                'RANDOM_SEED': str(job['random_seed']),
                'CORRELATION_TYPE': job['correlation_type'],
                'BUCKET_NAME': job['bucket_name']
            }
            
            task_name = f"patient_{job['patient_id']}_batch_{job['batch_id']}"
            
            success = submit_job_execution(
                project_id, region, job_name, env_vars, task_name
            )
            
            if success:
                submitted_count += 1
            else:
                failed_count += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        logger.info(f"  Batch {batch_id} complete: {len(batch)} submitted")
        logger.info(f"  Total progress: {submitted_count + failed_count}/{len(remaining)}")
        logger.info("")
        
        # Delay between batches
        if batch_num + BATCH_SIZE < len(remaining):
            time.sleep(2)
    
    logger.info("="*80)
    logger.info(" ALL JOBS SUBMITTED!")
    logger.info("="*80)
    logger.info(f" Submitted: {submitted_count}")
    logger.info(f" Failed: {failed_count}")
    logger.info("")
    logger.info("Jobs are now running in parallel on Cloud Run.")
    logger.info("Monitor progress:")
    logger.info(f"  gcloud run jobs executions list {job_name} --project {project_id} --region {region}")


if __name__ == '__main__':
    main()

