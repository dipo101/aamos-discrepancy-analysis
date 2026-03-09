"""
Cloud Run Worker: Permutation Analysis V3
Handles batched permutation testing for a single patient.

Uses this permutation approach:
1. Shuffle symptom vector
2. Re-run ALL 192 correlations
3. Compute median Fisher Z
4. Repeat for 500 permutations per batch
"""

import os
import json
import logging
import itertools
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd
import numpy as np
from scipy import stats
from data_loader_simplified import load_and_merge_patient_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# GCS client (lazy-loaded to avoid initialization errors)
_storage_client = None

def get_storage_client():
    """Get or create GCS storage client."""
    global _storage_client
    if _storage_client is None:
        # Try to get project from environment, fallback to default
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'power-line-monitoring-476216')
        _storage_client = storage.Client(project=project_id)
    return _storage_client


# ============================================================================
# PARAMETER COMBINATIONS: ALL 192 CONFIGS
# ============================================================================

def generate_all_param_combinations():
    """Generate all 192 parameter combinations."""
    timestamp_windows = [12, 24, 36, 48]
    use_daily_max_options = [False, True]
    use_calendar_day_options = [False, True]
    filter_zero_options = [False, True]
    categorization_methods = [
        'lower_bound',
        'midpoint',
        'midpoint_with_inhaler',
        'upper_bound',
        'upper_bound_with_inhaler',
        'one_hot'
    ]
    
    combos = list(itertools.product(
        timestamp_windows,
        use_daily_max_options,
        use_calendar_day_options,
        filter_zero_options,
        categorization_methods
    ))
    
    logger.info(f"Generated {len(combos)} parameter combinations")
    return combos


PARAM_COMBINATIONS = generate_all_param_combinations()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_gcs(bucket_name: str) -> dict:
    """Load all data files from GCS."""
    logger.info(f"Loading data from gs://{bucket_name}/data/")
    
    bucket = get_storage_client().bucket(bucket_name)
    
    # Load patient info
    patient_blob = bucket.blob('data/anonym_aamos00_patient_info.csv')
    patient_df = pd.read_csv(patient_blob.open('r'))
    
    # Load questionnaire
    quest_blob = bucket.blob('data/anonym_aamos00_dailyquestionnaire_dt.csv')
    questionnaire_df = pd.read_csv(quest_blob.open('r'))
    
    # Load inhaler
    inhaler_blob = bucket.blob('data/anonym_aamos00_smartinhaler_dt.csv')
    inhaler_df = pd.read_csv(inhaler_blob.open('r'))
    
    logger.info(f" Data loaded: {len(patient_df)} patients, "
               f"{len(questionnaire_df)} questionnaires, "
               f"{len(inhaler_df)} inhaler records")
    
    return {
        'patient_info': patient_df,
        'questionnaire': questionnaire_df,
        'inhaler': inhaler_df
    }


# ============================================================================
# CORE MERGE LOGIC
# ============================================================================

def merge_patient_data(
    patient_quest: pd.DataFrame,
    patient_inhaler: pd.DataFrame,
    timestamp_window: int,
    use_daily_max: bool,
    use_calendar_days: bool,
    filter_zeros: bool,
    categorization_method: str
) -> pd.DataFrame:
    """
    Merge questionnaire and inhaler data with specified parameters.
    
    This implements the core logic from data_loader.py in a simplified way.
    """
    # Make copies
    quest = patient_quest.copy()
    inhaler = patient_inhaler.copy()
    
    # Parse dates
    quest['date'] = pd.to_datetime(quest['date'])
    inhaler['date'] = pd.to_datetime(inhaler['date'])
    
    # Count inhaler usage for each questionnaire entry
    inhaler_counts = []
    
    for _, row in quest.iterrows():
        quest_date = row['date']
        
        if use_calendar_days:
            # Calendar day boundaries
            days_back = timestamp_window // 24
            start_date = quest_date - pd.Timedelta(days=days_back)
            end_date = quest_date
            mask = (inhaler['date'] >= start_date) & (inhaler['date'] <= end_date)
        else:
            # Rolling hours
            window = pd.Timedelta(hours=timestamp_window)
            start_time = quest_date - window
            end_time = quest_date
            mask = (inhaler['date'] >= start_time) & (inhaler['date'] <= end_time)
        
        count = len(inhaler[mask])
        
        # Apply daily max if requested
        if use_daily_max:
            # Group by day and take max
            inhaler_in_window = inhaler[mask].copy()
            if len(inhaler_in_window) > 0:
                inhaler_in_window['day'] = inhaler_in_window['date'].dt.date
                daily_counts = inhaler_in_window.groupby('day').size()
                count = daily_counts.max() if len(daily_counts) > 0 else 0
            else:
                count = 0
        
        inhaler_counts.append(count)
    
    quest['inhaler_usage'] = inhaler_counts
    
    # Filter zeros if requested
    if filter_zeros:
        quest = quest[quest['inhaler_usage'] > 0].copy()
    
    # Apply categorization to symptom column
    symptom_col = 'daily_relief_inhaler'
    
    if categorization_method == 'one_hot':
        # One-hot encoding
        quest[symptom_col] = quest[symptom_col].astype('category')
    elif categorization_method == 'lower_bound':
        quest[symptom_col] = quest[symptom_col].apply(lambda x: 0 if x < 12 else 12)
    elif categorization_method == 'midpoint':
        quest[symptom_col] = quest[symptom_col].apply(lambda x: x if x < 12 else 12)
    elif categorization_method == 'upper_bound':
        quest[symptom_col] = quest[symptom_col].apply(lambda x: x if x < 12 else 12)
    elif 'with_inhaler' in categorization_method:
        # Simplified: treat same as non-inhaler version
        quest[symptom_col] = quest[symptom_col].apply(lambda x: x if x < 12 else 12)
    
    return quest


# ============================================================================
# PERMUTATION ENGINE
# ============================================================================

def run_permutation_batch(
    patient_id: int,
    perm_start: int,
    perm_end: int,
    random_seed: int,
    correlation_type: str,
    data: dict
) -> list:
    """
    Run a batch of permutations for one patient.
    
    Returns:
        List of null median Z-scores (one per permutation)
    """
    logger.info(f"Patient {patient_id}: Running permutations {perm_start}-{perm_end}")
    
    # Filter patient data
    patient_quest = data['questionnaire'][
        data['questionnaire']['user_key'] == patient_id
    ].copy()
    
    patient_inhaler = data['inhaler'][
        data['inhaler']['user_key'] == patient_id
    ].copy()
    
    if len(patient_quest) == 0:
        logger.warning(f"Patient {patient_id}: No questionnaire data")
        return [np.nan] * (perm_end - perm_start)
    
    symptom_col = 'daily_relief_inhaler'
    symptom_values = patient_quest[symptom_col].values
    
    null_medians = []
    
    for perm_idx in range(perm_start, perm_end):
        # Shuffle symptom values with reproducible seed
        rng = np.random.RandomState(random_seed + perm_idx)
        shuffled_symptoms = rng.permutation(symptom_values)
        
        # Create shuffled questionnaire
        shuffled_quest = patient_quest.copy()
        shuffled_quest[symptom_col] = shuffled_symptoms
        
        # Compute correlations for ALL 192 configurations
        z_scores = []
        
        for (window, daily_max, calendar, filter_z, cat_method) in PARAM_COMBINATIONS:
            try:
                merged = load_and_merge_patient_data(
                    shuffled_quest,
                    patient_inhaler,
                    window,
                    daily_max,
                    calendar,
                    filter_z,
                    cat_method
                )
                
                if merged is None or len(merged) < 5:
                    continue
                
                # Get symptom and inhaler usage columns
                x = merged['inhaler_usage'].values
                y = merged[symptom_col].values
                
                # Check for constant arrays
                if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
                    continue
                
                if correlation_type == 'spearman':
                    corr, _ = stats.spearmanr(x, y)
                else:  # pearson
                    corr, _ = stats.pearsonr(x, y)
                
                # Fisher Z-transform
                if not np.isnan(corr) and -1 < corr < 1:
                    z = np.arctanh(corr)
                    if np.isfinite(z):
                        z_scores.append(z)
            
            except Exception as e:
                # Skip failed correlations
                continue
        
        # Compute median Z for this permutation
        if len(z_scores) > 0:
            null_medians.append(float(np.median(z_scores)))
        else:
            null_medians.append(np.nan)
    
    valid_count = sum(1 for x in null_medians if not np.isnan(x))
    logger.info(f"Patient {patient_id}: Completed {len(null_medians)} permutations "
               f"({valid_count} valid)")
    
    return null_medians


# ============================================================================
# GCS SAVE
# ============================================================================

def save_results_to_gcs(
    bucket_name: str,
    patient_id: int,
    batch_id: int,
    results: dict
):
    """Save batch results to GCS."""
    bucket = get_storage_client().bucket(bucket_name)
    
    filename = f"results/patient_{patient_id}_batch_{batch_id}.json"
    blob = bucket.blob(filename)
    
    blob.upload_from_string(
        json.dumps(results),
        content_type='application/json'
    )
    
    logger.info(f"Saved results to gs://{bucket_name}/{filename}")


# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'configs': len(PARAM_COMBINATIONS)}), 200


@app.route('/process', methods=['POST'])
def process():
    """
    Main processing endpoint.
    
    Expected JSON body:
    {
        "patient_id": 113,
        "batch_id": 0,
        "perm_start": 0,
        "perm_end": 500,
        "random_seed": 42,
        "correlation_type": "spearman",
        "bucket_name": "patient-concordance-permutation-v3"
    }
    """
    try:
        # Parse request
        request_json = request.get_json()
        
        patient_id = request_json['patient_id']
        batch_id = request_json['batch_id']
        perm_start = request_json['perm_start']
        perm_end = request_json['perm_end']
        random_seed = request_json['random_seed']
        correlation_type = request_json['correlation_type']
        bucket_name = request_json['bucket_name']
        
        logger.info(f"Starting job: Patient {patient_id}, Batch {batch_id}, "
                   f"Perms {perm_start}-{perm_end}")
        
        start_time = datetime.now()
        
        # Load data
        data = load_data_from_gcs(bucket_name)
        
        # Run permutations
        null_medians = run_permutation_batch(
            patient_id,
            perm_start,
            perm_end,
            random_seed,
            correlation_type,
            data
        )
        
        # Prepare results
        results = {
            'patient_id': patient_id,
            'batch_id': batch_id,
            'perm_start': perm_start,
            'perm_end': perm_end,
            'null_medians': null_medians,
            'n_valid': sum(1 for x in null_medians if not np.isnan(x)),
            'n_configs': len(PARAM_COMBINATIONS),
            'started_at': start_time.isoformat(),
            'completed_at': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        # Save to GCS
        save_results_to_gcs(bucket_name, patient_id, batch_id, results)
        
        logger.info(f" Job complete: Patient {patient_id}, Batch {batch_id}, "
                   f"Duration: {results['duration_seconds']:.1f}s, "
                   f"Configs: {results['n_configs']}")
        
        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'batch_id': batch_id,
            'n_permutations': len(null_medians),
            'n_configs': results['n_configs'],
            'duration_seconds': results['duration_seconds']
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # For local testing
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Using {len(PARAM_COMBINATIONS)} parameter combinations")
    app.run(host='0.0.0.0', port=port)
