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
    run_single_permutation,
    summarize_correlations,
)
from aamos_concordance.data import RAW_FILES

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
    all_correlations = []
    all_datasets = []
    valid_count = 0
    batch_start_time = time.time()

    for i in range(perm_start, perm_end):
        perm_iteration = i - perm_start + 1
        perm_start_time = time.time()

        correlations, datasets = run_single_permutation(
            patient_id, i, patient_questionnaire, patient_inhaler,
            param_combinations, random_seed, correlation_type,
            collect_datasets=collect_datasets,
        )
        if collect_datasets and datasets:
            all_datasets.extend(datasets)

        summary = summarize_correlations(correlations)
        if summary['n_valid_configs'] > 0:
            valid_count += 1
        all_correlations.append({'permutation_idx': i, **summary})

        if perm_iteration <= 5 or perm_iteration % 10 == 0:
            elapsed_total = time.time() - batch_start_time
            est_remaining = (elapsed_total / perm_iteration) * (n_perms - perm_iteration)
            logger.info(
                f"Patient {patient_id}, permutation {i}: {summary['n_valid_configs']}/{len(param_combinations)} valid configs, "
                f"{time.time() - perm_start_time:.2f}s | {perm_iteration}/{n_perms} | est. remaining {est_remaining / 60:.1f} min"
            )
        if perm_iteration % 50 == 0:
            gc.collect()

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
            config={'random_seed': random_seed, 'correlation_type': correlation_type,
                    'n_configs': len(param_combinations), 'perm_start': perm_start, 'perm_end': perm_end},
            data=data_provenance,
        ),
    }
    output_path = f'results/patient_{patient_id}_batch_{batch_id}.json'
    bucket.blob(output_path).upload_from_string(json.dumps(result_data))
    logger.info(f"Results saved to gs://{bucket_name}/{output_path}")

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
