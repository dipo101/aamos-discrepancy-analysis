"""
Data loading utilities for Cloud Run Jobs
Includes GCS loading and exact replication of merging/categorization logic
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from google.cloud import storage
from io import StringIO


def load_data_from_gcs(bucket_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data files from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Load patient info
    blob = bucket.blob('data/anonym_aamos00_patient_info.csv')
    patient_info_df = pd.read_csv(StringIO(blob.download_as_text()))
    
    # Load questionnaire
    blob = bucket.blob('data/anonym_aamos00_dailyquestionnaire_dt.csv')
    questionnaire_df = pd.read_csv(StringIO(blob.download_as_text()))
    
    # Load inhaler
    blob = bucket.blob('data/anonym_aamos00_smartinhaler_dt.csv')
    inhaler_df = pd.read_csv(StringIO(blob.download_as_text()))
    
    return patient_info_df, questionnaire_df, inhaler_df


def join_questionnaire_with_inhaler(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    user_key: int,
    timestamp_window: int,
    use_daily_max_windows: bool,
    use_calendar_days: bool
) -> Optional[pd.DataFrame]:
    """
    Exact copy of _join_questionnaire_with_inhaler from per_patient_correlation_analysis.py
    """
    if len(inhaler_df) == 0:
        # No inhaler data - set usage to 0
        questionnaire_df['inhaler_usage'] = 0
        return questionnaire_df
    
    # Make copies to avoid modifying the original dataframes
    inhaler_df = inhaler_df.copy()
    questionnaire_df = questionnaire_df.copy()
    
    # DEBUG: Log input data for first call only (any patient, using function attribute)
    if not hasattr(join_questionnaire_with_inhaler, '_first_call_logged'):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"      🔍 JOIN FUNCTION FIRST CALL (Patient {user_key}):")
        logger.info(f"         Input questionnaire rows: {len(questionnaire_df)}")
        logger.info(f"         Columns: {list(questionnaire_df.columns)}")
        logger.info(f"         daily_relief_inhaler dtype: {questionnaire_df['daily_relief_inhaler'].dtype}")
        logger.info(f"         daily_relief_inhaler unique: {questionnaire_df['daily_relief_inhaler'].nunique()}, values: {sorted(questionnaire_df['daily_relief_inhaler'].dropna().unique())[:10]}")
        join_questionnaire_with_inhaler._first_call_logged = True
    
    # Convert time strings to datetime (EXACT same logic)
    reference_date = pd.Timestamp('2000-01-01')
    
    def combine_date_time(row):
        base_date = reference_date + pd.Timedelta(days=row['date'])
        time_obj = pd.to_datetime(row['time']).time()
        return pd.Timestamp.combine(base_date.date(), time_obj)
    
    inhaler_df['timestamp'] = inhaler_df.apply(combine_date_time, axis=1)
    questionnaire_df['timestamp'] = questionnaire_df.apply(combine_date_time, axis=1)
    
    def aggregate_window(row):
        # Note: inhaler_df is already patient-filtered in main.py, so no user_mask needed
        if use_calendar_days:
            # Use calendar day boundaries
            days_back = timestamp_window // 24
            target_day = row['date'] - days_back
            time_mask = inhaler_df['date'] == target_day
        else:
            window_hours = pd.Timedelta(hours=timestamp_window)
            
            if use_daily_max_windows:
                # Use 24-hour chunks
                window_start = row['timestamp'] - window_hours
                window_end = window_start + pd.Timedelta(hours=24)
                time_mask = (
                    (inhaler_df['timestamp'] >= window_start) & 
                    (inhaler_df['timestamp'] < window_end)
                )
            else:
                # Rolling window
                time_mask = (
                    (inhaler_df['timestamp'] <= row['timestamp']) & 
                    (inhaler_df['timestamp'] >= row['timestamp'] - window_hours)
                )
        
        # Count matching records
        return len(inhaler_df[time_mask])
    
    questionnaire_df['inhaler_usage'] = questionnaire_df.apply(aggregate_window, axis=1)
    
    # Log output data for first call only
    if hasattr(join_questionnaire_with_inhaler, '_first_call_logged') and not hasattr(join_questionnaire_with_inhaler, '_first_exit_logged'):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"      🔍 JOIN FUNCTION EXIT (Patient {user_key}):")
        logger.info(f"         Output questionnaire rows: {len(questionnaire_df)}")
        logger.info(f"         daily_relief_inhaler dtype: {questionnaire_df['daily_relief_inhaler'].dtype}")
        logger.info(f"         daily_relief_inhaler unique: {questionnaire_df['daily_relief_inhaler'].nunique()}, values: {sorted(questionnaire_df['daily_relief_inhaler'].dropna().unique())[:10]}")
        join_questionnaire_with_inhaler._first_exit_logged = True
    
    return questionnaire_df


def categorize_inhaler_usage(
    df: pd.DataFrame,
    categorization_method: str,
    filter_out_zero_usage: bool = False
) -> Optional[pd.DataFrame]:
    """
    Exact copy of _categorize_inhaler_usage from data_loader.py (lines 170-271)
    """
    categorized_df = df.copy()
    
    # CORRECT 6 methods from data_loader.py
    if categorization_method == 'one_hot':
        categorize_fn = lambda x: 1 if x is not None and x >= 1 else 0
        
    elif categorization_method == 'lower_bound':
        def categorize_fn(x):
            if x is None or x == 0:
                return 0
            elif x >= 1 and x <= 2:
                return 1
            elif x >= 3 and x <= 4:
                return 3
            elif x >= 5 and x <= 8:
                return 5
            elif x >= 9 and x <= 12:
                return 9
            else:
                return 12
                
    elif categorization_method == 'midpoint':
        def categorize_fn(x):
            if x is None or x == 0:
                return 0
            elif x >= 1 and x <= 2:
                return 1.5
            elif x >= 3 and x <= 4:
                return 3.5
            elif x >= 5 and x <= 8:
                return 6.5
            elif x >= 9 and x <= 12:
                return 10.5
            else:
                return 12
                
    elif categorization_method == 'upper_bound':
        def categorize_fn(x):
            if x is None or x == 0:
                return 0
            elif x >= 1 and x <= 2:
                return 2
            elif x >= 3 and x <= 4:
                return 4
            elif x >= 5 and x <= 8:
                return 8
            elif x >= 9 and x <= 12:
                return 12
            else:
                return 12
                
    elif categorization_method == 'midpoint_with_inhaler':
        midpoint_for_greater_than_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].mean()
        def categorize_fn(x):
            if x is None or x == 0:
                return 0
            elif x >= 1 and x <= 2:
                return 1.5
            elif x >= 3 and x <= 4:
                return 3.5
            elif x >= 5 and x <= 8:
                return 6.5
            elif x >= 9 and x <= 12:
                return 10.5
            else:
                return midpoint_for_greater_than_12
                
    elif categorization_method == 'upper_bound_with_inhaler':
        max_above_12 = df[df['inhaler_usage'] > 12]['inhaler_usage'].max()
        if max_above_12 > 24:
            sd_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].std()
            median_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].median()
            upper_bound_for_greater_than_12 = 3 * sd_above_12 + median_above_12
        else:
            upper_bound_for_greater_than_12 = max_above_12
        
        def categorize_fn(x):
            if x is None or x == 0:
                return 0
            elif x >= 1 and x <= 2:
                return 2
            elif x >= 3 and x <= 4:
                return 4
            elif x >= 5 and x <= 8:
                return 8
            elif x >= 9 and x <= 12:
                return 12
            else:
                return upper_bound_for_greater_than_12
    else:
        # Unknown method - return as-is
        return categorized_df
    
    # Apply categorization to BOTH columns (exact same as original - lines 268-269)
    categorized_df['inhaler_usage'] = categorized_df['inhaler_usage'].apply(categorize_fn)
    categorized_df['daily_relief_inhaler'] = categorized_df['daily_relief_inhaler'].apply(categorize_fn)
    
    # Filter out zeros if requested
    if filter_out_zero_usage:
        categorized_df = categorized_df[
            (categorized_df['inhaler_usage'] > 0) | 
            (categorized_df['daily_relief_inhaler'] > 0)
        ]
    
    if len(categorized_df) == 0:
        return None
    
    return categorized_df

