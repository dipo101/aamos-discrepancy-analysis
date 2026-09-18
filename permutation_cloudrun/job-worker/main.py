"""
Cloud Run JOB Worker - Processes one batch of permutations for one patient and exits.

No HTTP server needed - reads job params from environment variables.

This file is a thin I/O wrapper. All analysis logic (window join,
categorisation, 132-config multiverse, permutation kernel, summary statistics)
lives in the ``aamos_concordance`` package at the repository root so it can be
tested locally and shared with the per-patient and downstream scripts. The
package is copied into the build context by scripts/deploy_job.sh.

Environment variables:
    PATIENT_ID        (required) patient user_key
    BUCKET_NAME       (required) GCS bucket holding data/ and receiving results/
    RANDOM_SEED       default 42; permutation i shuffles with seed RANDOM_SEED + i
    CORRELATION_TYPE  'spearman' (default) or 'pearson'
    COLLECT_DATASETS  'true' to also upload the categorised frames as Parquet
    (the per-configuration Fisher Z vectors for both correlation types are always
     uploaded, to <results prefix>perconfig/patient_<id>_batch_<b>.parquet)
    WORLD_ID          world id (default span=union__case=A). Only the baseline is implemented so
                      far; any other value exits with an error rather than silently
                      computing the baseline. Results go to gs://<bucket>/<results prefix for the world>/
    TOTAL_PERMS + PERMS_PER_TASK + CLOUD_RUN_TASK_INDEX   multi-task mode, or
    BATCH_ID + PERM_START + PERM_END                       single-task mode
"""
import gc
import hashlib
import io
import json
import logging
import os
import sys
import time
import warnings
from io import StringIO
from typing import Tuple

import pandas as pd
from google.cloud import storage
from scipy import stats

from aamos_concordance import (
    build_record,
    generate_param_combinations,
    run_batch,
)
from aamos_concordance.data import RAW_FILES
from aamos_concordance.worlds import BASELINE, as_world, gcs_results_prefix

# Suppress ConstantInputWarning - it's expected and handled with np.nan
warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_gcs(bucket_name: str) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict]:
    """Load the raw CSVs from gs://<bucket>/data/ and return them with their SHA-256 hashes."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    frames, hashes = [], {}
    for name in RAW_FILES:
        raw_bytes = bucket.blob(f'data/{name}').download_as_bytes()
        hashes[name] = hashlib.sha256(raw_bytes).hexdigest()
        frames.append(pd.read_csv(StringIO(raw_bytes.decode('utf-8'))))
    provenance = {'data_dir': f'gs://{bucket_name}/data', 'sha256': hashes, 'manifest_verified': False}
    return tuple(frames), provenance


def resolve_batch() -> Tuple[int, int, int]:
    """Return (batch_id, perm_start, perm_end) from the environment."""
    if 'TOTAL_PERMS' in os.environ:
        task_index = int(os.environ.get('CLOUD_RUN_TASK_INDEX', 0))
        total_perms = int(os.environ['TOTAL_PERMS'])
        perms_per_task = int(os.environ.get('PERMS_PER_TASK', 75))
        perm_start = task_index * perms_per_task
        perm_end = min(perm_start + perms_per_task, total_perms)
        logger.info(f"TASK STARTED (multi-task mode): task {task_index}, perms {perm_start}-{perm_end - 1} of {total_perms}")
        return task_index, perm_start, perm_end
    batch_id = int(os.environ['BATCH_ID'])
    perm_start = int(os.environ['PERM_START'])
    perm_end = int(os.environ['PERM_END'])
    logger.info(f"JOB STARTED (single-task mode): batch {batch_id}, perms {perm_start}-{perm_end - 1}")
    return batch_id, perm_start, perm_end


def main():
    job_start_time = time.time()

    patient_id = int(os.environ['PATIENT_ID'])
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    correlation_type = os.environ.get('CORRELATION_TYPE', 'spearman')
    bucket_name = os.environ['BUCKET_NAME']
    collect_datasets = os.environ.get('COLLECT_DATASETS', 'false').lower() == 'true'
    world = as_world(os.environ.get('WORLD_ID', str(BASELINE)))
    if world != BASELINE:
        logger.error(f"WORLD_ID={world}: span/absence-case handling is not implemented in the worker yet; "
                     "refusing to run so that baseline results are not written under a non-baseline world.")
        sys.exit(2)
    results_prefix = gcs_results_prefix(world)
    batch_id, perm_start, perm_end = resolve_batch()

    logger.info(f"   Patient {patient_id} | seed {random_seed} | {correlation_type} | bucket {bucket_name}")

    logger.info(f"Loading data from gs://{bucket_name}/data/...")
    load_start = time.time()
    (_, questionnaire_df, inhaler_df), data_provenance = load_data_from_gcs(bucket_name)
    logger.info(f"Data loaded in {time.time() - load_start:.2f}s: "
                f"{len(questionnaire_df)} questionnaires, {len(inhaler_df)} inhaler records")

    patient_questionnaire = questionnaire_df[questionnaire_df['user_key'] == patient_id].copy()
    patient_inhaler = inhaler_df[inhaler_df['user_key'] == patient_id].copy()
    if len(patient_questionnaire) == 0:
        logger.error(f"No questionnaire data for patient {patient_id}")
        sys.exit(1)
    logger.info(f"Patient {patient_id}: {len(patient_questionnaire)} questionnaires, {len(patient_inhaler)} inhaler records")

    param_combinations = generate_param_combinations()
    logger.info(f"Generated {len(param_combinations)} parameter combinations")

    n_perms = perm_end - perm_start
    batch_start_time = time.time()
    last_time = [batch_start_time]

    def progress(perm_idx, record):
        perm_iteration = perm_idx - perm_start + 1
        now = time.time()
        if perm_iteration <= 5 or perm_iteration % 10 == 0:
            est_remaining = ((now - batch_start_time) / perm_iteration) * (n_perms - perm_iteration)
            logger.info(
                f"Patient {patient_id}, permutation {perm_idx}: {record['n_valid_configs']}/{len(param_combinations)} valid configs, "
                f"{now - last_time[0]:.2f}s | {perm_iteration}/{n_perms} | est. remaining {est_remaining / 60:.1f} min"
            )
        if perm_iteration % 50 == 0:
            gc.collect()
        last_time[0] = now

    all_correlations, per_config, all_datasets = run_batch(
        patient_id, patient_questionnaire, patient_inhaler, param_combinations,
        perm_start, perm_end, random_seed, correlation_type,
        collect_datasets=collect_datasets, progress=progress,
    )
    valid_count = sum(1 for r in all_correlations if r['n_valid_configs'] > 0)

    total_elapsed = time.time() - batch_start_time
    logger.info(f"Patient {patient_id}, batch {batch_id} complete: {n_perms} permutations in {total_elapsed / 60:.2f} min, "
                f"{valid_count}/{n_perms} valid")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    result_data = {
        'patient_id': patient_id,
        'batch_id': batch_id,
        'perm_start': perm_start,
        'perm_end': perm_end,
        'n_valid': valid_count,
        'permutations': all_correlations,
        'timing': {
            'total_seconds': time.time() - job_start_time,
            'avg_seconds_per_permutation': (time.time() - job_start_time) / n_perms,
        },
        'provenance': build_record(
            config={'world_id': str(world), 'random_seed': random_seed, 'primary_correlation_type': correlation_type,
                    'correlation_types_stored': ['spearman', 'pearson'],
                    'n_configs': len(param_combinations), 'perm_start': perm_start, 'perm_end': perm_end},
            data=data_provenance,
        ),
    }
    output_path = f'{results_prefix}patient_{patient_id}_batch_{batch_id}.json'
    bucket.blob(output_path).upload_from_string(json.dumps(result_data))
    logger.info(f"Results saved to gs://{bucket_name}/{output_path}")

    buf = io.BytesIO()
    per_config.to_parquet(buf, engine='pyarrow', compression='snappy', index=False)
    buf.seek(0)
    per_config_path = f'{results_prefix}perconfig/patient_{patient_id}_batch_{batch_id}.parquet'
    bucket.blob(per_config_path).upload_from_file(buf, content_type='application/octet-stream')
    logger.info(f"Per-config Z ({len(per_config)} rows) saved to gs://{bucket_name}/{per_config_path}")

    if collect_datasets and all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        buf = io.BytesIO()
        combined_df.to_parquet(buf, engine='pyarrow', compression='snappy', index=False)
        buf.seek(0)
        parquet_path = f'datasets/patient_{patient_id}_batch_{batch_id}.parquet'
        bucket.blob(parquet_path).upload_from_file(buf, content_type='application/octet-stream')
        logger.info(f"Datasets ({len(combined_df)} rows) saved to gs://{bucket_name}/{parquet_path}")

    logger.info(f"JOB COMPLETE: patient {patient_id}, batch {batch_id}, {n_perms} permutations in "
                f"{(time.time() - job_start_time) / 60:.2f} min")
    sys.exit(0)


if __name__ == '__main__':
    main()
