"""
Cloud Run JOB Worker - Processes one batch of permutations and exits
No HTTP server needed - reads job params from environment variables
"""
import os
import sys
import json
import logging
import gc
import warnings
import time
import io
import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import storage
from pathlib import Path

# Suppress ConstantInputWarning - it's expected and handled with np.nan
warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from data_loader_simplified import (
    load_data_from_gcs,
    join_questionnaire_with_inhaler,
    categorize_inhaler_usage
)


def generate_param_combinations():
    """Generate 132 UNIQUE parameter combinations.
    
    Note: 192 raw combinations exist, but 60 are duplicates because:
    - When use_calendar_days=True, use_daily_max_windows is ignored
    - When use_calendar_days=True, timestamp_window 24 and 36 both map to 1 day
    """
    timestamp_windows = [12, 24, 36, 48]
    use_daily_max_windows = [False, True]
    use_calendar_days = [False, True]
    filter_out_zero_usage = [False, True]
    categorization_methods = [
        'one_hot',
        'lower_bound',
        'midpoint',
        'midpoint_with_inhaler',
        'upper_bound',
        'upper_bound_with_inhaler'
    ]
    
    combinations = []
    seen_effective_configs = set()
    
    for window in timestamp_windows:
        for daily_max in use_daily_max_windows:
            for calendar in use_calendar_days:
                for filter_zeros in filter_out_zero_usage:
                    for cat_method in categorization_methods:
                        # Compute effective config key to detect duplicates
                        if calendar:
                            # Calendar mode: daily_max is ignored, 24 and 36 both map to day 1
                            effective_window = window // 24  # 12->0, 24->1, 36->1, 48->2
                            effective_key = ('calendar', effective_window, filter_zeros, cat_method)
                        else:
                            # Non-calendar mode: all params matter
                            mode = 'fixed_chunk' if daily_max else 'rolling'
                            effective_key = (mode, window, filter_zeros, cat_method)
                        
                        # Skip if we've seen this effective config
                        if effective_key in seen_effective_configs:
                            continue
                        seen_effective_configs.add(effective_key)
                        
                        combinations.append({
                            'timestamp_window': window,
                            'use_daily_max_windows': daily_max,
                            'use_calendar_days': calendar,
                            'filter_out_zero_usage': filter_zeros,
                            'categorization_method': cat_method
                        })
    
    return combinations


def run_single_permutation(patient_id, perm_idx, questionnaire_df, inhaler_df, 
                          param_combinations, random_seed, correlation_type,
                          collect_datasets=False):
    """Run one permutation across all 132 unique configs with detailed failure tracking.
    
    Args:
        collect_datasets: If True, also return the categorized datasets for each config.
    
    Returns:
        correlations: List of Fisher Z values (one per config)
        datasets: List of dataset records (if collect_datasets=True), else empty list
    """
    start_time = time.time()
    
    # Log BEFORE any processing for Perm 0
    if perm_idx == 0:
        logger.info(f"   🔍 BEFORE SHUFFLE (Perm 0):")
        logger.info(f"      Questionnaire input rows: {len(questionnaire_df)}")
        logger.info(f"      daily_relief_inhaler unique: {questionnaire_df['daily_relief_inhaler'].nunique()}")
        logger.info(f"      daily_relief_inhaler values: {sorted(questionnaire_df['daily_relief_inhaler'].unique())[:10]}")
        logger.info(f"      daily_relief_inhaler dtype: {questionnaire_df['daily_relief_inhaler'].dtype}")
    
    # Shuffle symptom data
    perm_seed = random_seed + perm_idx
    np.random.seed(perm_seed)
    
    shuffled_questionnaire = questionnaire_df.copy()
    # Use .values to avoid index mismatch that causes NaN
    shuffled_questionnaire['daily_relief_inhaler'] = (
        shuffled_questionnaire['daily_relief_inhaler']
        .sample(frac=1, random_state=perm_seed)
        .values  # Get numpy array, not Series with reset indices
    )
    
    # Log AFTER shuffle for Perm 0
    if perm_idx == 0:
        logger.info(f"   🔍 AFTER SHUFFLE (Perm 0):")
        logger.info(f"      Shuffled questionnaire rows: {len(shuffled_questionnaire)}")
        logger.info(f"      daily_relief_inhaler unique: {shuffled_questionnaire['daily_relief_inhaler'].nunique()}")
        logger.info(f"      daily_relief_inhaler values: {sorted(shuffled_questionnaire['daily_relief_inhaler'].unique())[:10]}")
        logger.info(f"      daily_relief_inhaler dtype: {shuffled_questionnaire['daily_relief_inhaler'].dtype}")
    
    # Track failure reasons
    fail_reasons = {
        'join_null': 0,
        'join_too_few_rows': 0,
        'categorize_null': 0,
        'categorize_too_few_rows': 0,
        'correlation_invalid': 0,
        'exception': 0
    }
    
    # Run all configs (132 unique)
    correlations = []
    datasets = []  # Collect categorized datasets if requested
    n_configs = len(param_combinations)
    
    for config_idx, params in enumerate(param_combinations):
        try:
            # Join data
            merged_df = join_questionnaire_with_inhaler(
                shuffled_questionnaire,
                inhaler_df,  # This is already patient-filtered from function parameter
                patient_id,
                params['timestamp_window'],
                params['use_daily_max_windows'],
                params['use_calendar_days']
            )
            
            if merged_df is None:
                fail_reasons['join_null'] += 1
                correlations.append(np.nan)
                continue
            
            if len(merged_df) < 3:
                fail_reasons['join_too_few_rows'] += 1
                correlations.append(np.nan)
                del merged_df
                continue
            
            # Log PRE-categorization for first config of Perm 0
            if perm_idx == 0 and config_idx == 0:
                logger.info(f"   🔍 PRE-categorization (Config 0):")
                logger.info(f"      Merged rows: {len(merged_df)}")
                logger.info(f"      Inhaler usage unique: {merged_df['inhaler_usage'].nunique()}, values: {sorted(merged_df['inhaler_usage'].unique())[:10]}")
                logger.info(f"      Relief inhaler unique: {merged_df['daily_relief_inhaler'].nunique()}, values: {sorted(merged_df['daily_relief_inhaler'].unique())[:10]}")
            
            # Categorize
            categorized_df = categorize_inhaler_usage(
                merged_df,
                params['categorization_method'],
                params['filter_out_zero_usage']
            )
            del merged_df  # Free memory immediately
            
            if categorized_df is None:
                fail_reasons['categorize_null'] += 1
                correlations.append(np.nan)
                continue
            
            if len(categorized_df) < 3:
                fail_reasons['categorize_too_few_rows'] += 1
                correlations.append(np.nan)
                del categorized_df
                continue
            
            # Log POST-categorization for first config of Perm 0
            if perm_idx == 0 and config_idx == 0:
                logger.info(f"   POST-categorization (method='{params['categorization_method']}'):")
                logger.info(f"      Categorized rows: {len(categorized_df)}")
                logger.info(f"      Inhaler unique: {categorized_df['inhaler_usage'].nunique()}, values: {sorted(categorized_df['inhaler_usage'].unique())[:10]}")
                logger.info(f"      Relief unique: {categorized_df['daily_relief_inhaler'].nunique()}, values: {sorted(categorized_df['daily_relief_inhaler'].unique())[:10]}")
            
            # Check for constant values before correlation (causes correlation = NaN)
            inhaler_unique = categorized_df['inhaler_usage'].nunique()
            relief_unique = categorized_df['daily_relief_inhaler'].nunique()
            
            if inhaler_unique == 1 or relief_unique == 1:
                # Constant input - correlation undefined
                fail_reasons['correlation_invalid'] += 1
                correlations.append(np.nan)
                del categorized_df
                continue
            
            # Collect dataset if requested (before we delete it)
            if collect_datasets:
                # Add metadata to each row
                dataset_records = categorized_df.copy()
                dataset_records['permutation_idx'] = perm_idx
                dataset_records['config_idx'] = config_idx
                dataset_records['timestamp_window'] = params['timestamp_window']
                dataset_records['use_daily_max_windows'] = params['use_daily_max_windows']
                dataset_records['use_calendar_days'] = params['use_calendar_days']
                dataset_records['categorization_method'] = params['categorization_method']
                dataset_records['filter_out_zero_usage'] = params['filter_out_zero_usage']
                datasets.append(dataset_records)
            
            # Compute correlation
            if correlation_type == 'spearman':
                corr, _ = stats.spearmanr(
                    categorized_df['inhaler_usage'],
                    categorized_df['daily_relief_inhaler']
                )
            else:  # pearson
                corr, _ = stats.pearsonr(
                    categorized_df['inhaler_usage'],
                    categorized_df['daily_relief_inhaler']
                )
            
            del categorized_df  # Free memory immediately
            
            # Fisher Z transform
            if np.isfinite(corr) and -1 < corr < 1:
                fisher_z = np.arctanh(corr)
                correlations.append(fisher_z)
            else:
                fail_reasons['correlation_invalid'] += 1
                correlations.append(np.nan)
                
        except Exception as e:
            fail_reasons['exception'] += 1
            # Log unexpected exceptions (not ConstantInputWarning)
            if not isinstance(e, (stats.ConstantInputWarning, UserWarning)):
                logger.warning(f"Patient {patient_id}, Perm {perm_idx}, Config {config_idx}: {type(e).__name__}: {str(e)[:100]}")
            correlations.append(np.nan)
    
    elapsed = time.time() - start_time
    n_valid = int(np.sum(np.isfinite(correlations)))
    n_failed = n_configs - n_valid
    
    # Log failure breakdown if there are significant failures
    if n_failed > 0 and perm_idx % 10 == 0:  # Log every 10th permutation if there are failures
        logger.info(f"   Patient {patient_id}, Perm {perm_idx} failures ({n_failed}/{n_configs}): "
                   f"join_null={fail_reasons['join_null']}, "
                   f"join_short={fail_reasons['join_too_few_rows']}, "
                   f"cat_null={fail_reasons['categorize_null']}, "
                   f"cat_short={fail_reasons['categorize_too_few_rows']}, "
                   f"corr_invalid={fail_reasons['correlation_invalid']}, "
                   f"exceptions={fail_reasons['exception']}")
    
    logger.debug(f"  Patient {patient_id}, Perm {perm_idx}: {n_valid}/{n_configs} valid configs in {elapsed:.2f}s")
    
    return correlations, datasets


def main():
    """Main entry point - reads env vars and processes job.
    
    Supports two modes:
    1. Single-task mode: BATCH_ID, PERM_START, PERM_END provided
    2. Multi-task mode: Uses CLOUD_RUN_TASK_INDEX to auto-calculate batch
    """
    job_start_time = time.time()
    
    # Read common parameters
    patient_id = int(os.environ['PATIENT_ID'])
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    correlation_type = os.environ.get('CORRELATION_TYPE', 'spearman')
    bucket_name = os.environ['BUCKET_NAME']
    
    # Check for multi-task mode
    # If TOTAL_PERMS is provided, use multi-task calculation
    # CLOUD_RUN_TASK_INDEX is auto-set by Cloud Run (defaults to 0 for single task)
    if 'TOTAL_PERMS' in os.environ:
        # Multi-task mode: Calculate batch parameters from task index
        task_index = int(os.environ.get('CLOUD_RUN_TASK_INDEX', 0))
        total_perms = int(os.environ['TOTAL_PERMS'])
        perms_per_task = int(os.environ.get('PERMS_PER_TASK', 75))
        
        batch_id = task_index
        perm_start = task_index * perms_per_task
        perm_end = min(perm_start + perms_per_task, total_perms)
        
        logger.info("="*80)
        logger.info(f" TASK STARTED (Multi-task mode)")
        logger.info(f"   Patient ID: {patient_id}")
        logger.info(f"   Task Index: {task_index} (Batch {batch_id})")
        logger.info(f"   Permutation Range: {perm_start} to {perm_end-1} ({perm_end - perm_start} permutations)")
        logger.info(f"   Total Permutations for Patient: {total_perms}, Task Size: {perms_per_task}")
        logger.info(f"   Random Seed: {random_seed}")
        logger.info(f"   Correlation Type: {correlation_type}")
        logger.info(f"   GCS Bucket: {bucket_name}")
        logger.info("="*80)
    else:
        # Legacy single-task mode: Read batch parameters directly
        batch_id = int(os.environ['BATCH_ID'])
        perm_start = int(os.environ['PERM_START'])
        perm_end = int(os.environ['PERM_END'])
        
        logger.info("="*80)
        logger.info(f" JOB STARTED (Single-task mode)")
        logger.info(f"   Patient ID: {patient_id}")
        logger.info(f"   Batch ID: {batch_id}")
        logger.info(f"   Permutation Range: {perm_start} to {perm_end-1} ({perm_end - perm_start} permutations)")
        logger.info(f"   Random Seed: {random_seed}")
        logger.info(f"   Correlation Type: {correlation_type}")
        logger.info(f"   GCS Bucket: {bucket_name}")
        logger.info("="*80)
    
    # Load data from GCS
    logger.info(f" Loading data from gs://{bucket_name}/data/...")
    load_start = time.time()
    patient_info_df, questionnaire_df, inhaler_df = load_data_from_gcs(bucket_name)
    load_time = time.time() - load_start
    logger.info(f" Data loaded in {load_time:.2f}s: {len(questionnaire_df)} total questionnaires, {len(inhaler_df)} total inhaler records")
    
    # DEBUG: Log relief inhaler values immediately after GCS load
    logger.info(f"    IMMEDIATELY AFTER GCS LOAD:")
    logger.info(f"      daily_relief_inhaler dtype: {questionnaire_df['daily_relief_inhaler'].dtype}")
    logger.info(f"      daily_relief_inhaler unique count: {questionnaire_df['daily_relief_inhaler'].nunique()}")
    logger.info(f"      daily_relief_inhaler sample values: {sorted(questionnaire_df['daily_relief_inhaler'].dropna().unique())[:15]}")
    logger.info(f"      daily_relief_inhaler range: {questionnaire_df['daily_relief_inhaler'].min()} to {questionnaire_df['daily_relief_inhaler'].max()}")
    
    # Filter for this patient
    logger.info(f" Filtering data for patient {patient_id}...")
    patient_questionnaire = questionnaire_df[questionnaire_df['user_key'] == patient_id].copy()
    patient_inhaler = inhaler_df[inhaler_df['user_key'] == patient_id].copy()
    
    if len(patient_questionnaire) == 0:
        logger.error(f" No questionnaire data for patient {patient_id}")
        sys.exit(1)
    
    logger.info(f" Patient {patient_id} data: {len(patient_questionnaire)} questionnaires, {len(patient_inhaler)} inhaler records")
    
    # DEBUG: Log relief inhaler values AFTER filtering for this patient
    logger.info(f"    AFTER FILTERING FOR PATIENT {patient_id}:")
    logger.info(f"      daily_relief_inhaler dtype: {patient_questionnaire['daily_relief_inhaler'].dtype}")
    logger.info(f"      daily_relief_inhaler unique values: {sorted(patient_questionnaire['daily_relief_inhaler'].dropna().unique())[:15]}")
    logger.info(f"      daily_relief_inhaler range: {patient_questionnaire['daily_relief_inhaler'].min()} to {patient_questionnaire['daily_relief_inhaler'].max()}")
    if len(patient_inhaler) > 0:
        logger.info(f"   Questionnaire date range: {patient_questionnaire['date'].min()} to {patient_questionnaire['date'].max()}")
        logger.info(f"   Inhaler date range: {patient_inhaler['date'].min()} to {patient_inhaler['date'].max()}")
    
    # Generate parameter combinations
    logger.info(f"  Generating parameter combinations...")
    param_combinations = generate_param_combinations()
    logger.info(f" Generated {len(param_combinations)} parameter combinations")
    
    # Run permutations
    n_perms = perm_end - perm_start
    logger.info(f" Starting {n_perms} permutations for Patient {patient_id} (Batch {batch_id})")
    logger.info(f"   Permutation range: {perm_start} to {perm_end-1}")
    
    # Check if we should collect full datasets (optional, for sensitivity analysis)
    collect_datasets = os.environ.get('COLLECT_DATASETS', 'false').lower() == 'true'
    if collect_datasets:
        logger.info(f" Dataset collection ENABLED - will save categorized datasets to Parquet")
    
    all_correlations = []
    all_datasets = []  # Accumulated datasets (only if collect_datasets=True)
    valid_count = 0
    batch_start_time = time.time()
    last_log_time = batch_start_time
    
    for i in range(perm_start, perm_end):
        perm_iteration = i - perm_start + 1
        
        # Log at the start of each permutation (first 5, then every 10th)
        if perm_iteration <= 5 or perm_iteration % 10 == 0:
            logger.info(f"  Patient {patient_id}, Starting Permutation {i} ({perm_iteration}/{n_perms})")
        
        perm_start_time = time.time()
        correlations, datasets = run_single_permutation(
            patient_id, i, patient_questionnaire, patient_inhaler,
            param_combinations, random_seed, correlation_type,
            collect_datasets=collect_datasets
        )
        perm_elapsed = time.time() - perm_start_time
        
        # Accumulate datasets if collecting
        if collect_datasets and datasets:
            all_datasets.extend(datasets)
        
        # Compute summary statistics for this permutation
        valid_correlations = [c for c in correlations if np.isfinite(c)]
        n_valid_configs = len(valid_correlations)
        
        if n_valid_configs > 0:
            # Basic stats
            median_z = float(np.median(valid_correlations))
            mean_z = float(np.mean(valid_correlations))
            min_z = float(np.min(valid_correlations))
            max_z = float(np.max(valid_correlations))
            q25_z = float(np.percentile(valid_correlations, 25))
            q75_z = float(np.percentile(valid_correlations, 75))
            std_z = float(np.std(valid_correlations))
            
            # IQR and outlier detection (boxplot style)
            iqr = q75_z - q25_z
            lower_fence = q25_z - 1.5 * iqr
            upper_fence = q75_z + 1.5 * iqr
            
            # Whiskers: furthest non-outlier values
            non_outliers = [c for c in valid_correlations if lower_fence <= c <= upper_fence]
            if non_outliers:
                lower_whisker = float(min(non_outliers))
                upper_whisker = float(max(non_outliers))
            else:
                # Edge case: all values are outliers (shouldn't happen normally)
                lower_whisker = min_z
                upper_whisker = max_z
            
            # Outliers
            outliers = [float(c) for c in valid_correlations if c < lower_fence or c > upper_fence]
            n_outliers = len(outliers)
            
            valid_count += 1
        else:
            # No valid correlations - set all to None
            median_z = mean_z = min_z = max_z = q25_z = q75_z = std_z = None
            iqr = lower_fence = upper_fence = lower_whisker = upper_whisker = None
            outliers = []
            n_outliers = 0
        
        all_correlations.append({
            'permutation_idx': i,
            'n_valid_configs': n_valid_configs,
            # Basic stats
            'median_fisher_z': median_z,
            'mean_fisher_z': mean_z,
            'min_fisher_z': min_z,
            'max_fisher_z': max_z,
            'q25_fisher_z': q25_z,
            'q75_fisher_z': q75_z,
            'std_fisher_z': std_z,
            # Boxplot stats
            'iqr_fisher_z': float(iqr) if iqr is not None else None,
            'lower_fence': float(lower_fence) if lower_fence is not None else None,
            'upper_fence': float(upper_fence) if upper_fence is not None else None,
            'lower_whisker': lower_whisker,
            'upper_whisker': upper_whisker,
            'n_outliers': n_outliers,
            'outliers': outliers if n_outliers <= 20 else outliers[:20],  # Cap to avoid huge JSON
        })
        
        # Log completion (first 5, then every 10th)
        if perm_iteration <= 5 or perm_iteration % 10 == 0:
            elapsed_total = time.time() - batch_start_time
            avg_time_per_perm = elapsed_total / perm_iteration
            remaining_perms = n_perms - perm_iteration
            est_remaining_time = avg_time_per_perm * remaining_perms
            
            logger.info(
                f" Patient {patient_id}, Permutation {i} complete: "
                f"{n_valid_configs}/{len(param_combinations)} valid configs, took {perm_elapsed:.2f}s | "
                f"Progress: {perm_iteration}/{n_perms} ({100*perm_iteration/n_perms:.1f}%) | "
                f"Valid perms: {valid_count}/{perm_iteration} | "
                f"Est. remaining: {est_remaining_time/60:.1f} min"
            )
        
        # Force garbage collection every 50 permutations to prevent memory accumulation error I saw before
        if perm_iteration % 50 == 0:
            gc.collect()
            logger.info(f"  Garbage collection performed at permutation {perm_iteration}")
    
    total_elapsed = time.time() - batch_start_time
    logger.info(
        f" Patient {patient_id}, Batch {batch_id} COMPLETE: "
        f"{n_perms} permutations in {total_elapsed/60:.2f} min "
        f"({total_elapsed/n_perms:.2f}s avg per perm) | "
        f"{valid_count}/{n_perms} valid"
    )
    
    # Save results to GCS
    logger.info(f" Saving results to GCS...")
    save_start = time.time()
    
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
            'avg_seconds_per_permutation': (time.time() - job_start_time) / n_perms
        }
    }
    
    output_path = f'results/patient_{patient_id}_batch_{batch_id}.json'
    blob = bucket.blob(output_path)
    blob.upload_from_string(json.dumps(result_data))
    save_time = time.time() - save_start
    
    logger.info(f" Results saved in {save_time:.2f}s to gs://{bucket_name}/{output_path}")
    
    # Save datasets as Parquet if collected
    if collect_datasets and all_datasets:
        parquet_start = time.time()
        logger.info(f" Saving {len(all_datasets)} dataset records as Parquet...")
        
        # Concatenate all collected DataFrames
        combined_df = pd.concat(all_datasets, ignore_index=True)
        logger.info(f"   Combined DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        logger.info(f"   In-memory size: {combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Save to Parquet (efficient compression)
        parquet_buffer = io.BytesIO()
        combined_df.to_parquet(parquet_buffer, engine='pyarrow', compression='snappy', index=False)
        parquet_size_mb = parquet_buffer.tell() / 1024 / 1024
        
        # Upload to GCS
        parquet_buffer.seek(0)
        parquet_path = f'datasets/patient_{patient_id}_batch_{batch_id}.parquet'
        parquet_blob = bucket.blob(parquet_path)
        parquet_blob.upload_from_file(parquet_buffer, content_type='application/octet-stream')
        
        parquet_time = time.time() - parquet_start
        logger.info(f" Datasets saved in {parquet_time:.2f}s to gs://{bucket_name}/{parquet_path}")
        logger.info(f"   Parquet file size: {parquet_size_mb:.2f} MB")
        
        # Free memory
        del combined_df, all_datasets
        gc.collect()
    
    total_job_time = time.time() - job_start_time
    logger.info("="*80)
    logger.info(f" JOB COMPLETE")
    logger.info(f"   Patient {patient_id}, Batch {batch_id}")
    logger.info(f"   Processed {n_perms} permutations in {total_job_time/60:.2f} minutes")
    logger.info(f"   Valid permutations: {valid_count}/{n_perms} ({100*valid_count/n_perms:.1f}%)")
    logger.info(f"   Average time per permutation: {total_job_time/n_perms:.2f}s")
    logger.info("="*80)
    
    sys.exit(0)


if __name__ == '__main__':
    main()

