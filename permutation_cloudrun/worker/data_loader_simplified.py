"""
Replication of per_patient_correlation_analysis.py data loading logic.
It ensures our Cloud Run produces identical results to local execution.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_and_merge_patient_data(
    patient_quest: pd.DataFrame,
    patient_inhaler: pd.DataFrame,
    timestamp_window: int,
    use_daily_max: bool,
    use_calendar_days: bool,
    filter_zeros: bool,
    categorization_method: str
) -> Optional[pd.DataFrame]:
    """
    Replication of PerPatientDataLoader.load_patient_data()
    from per_patient_correlation_analysis.py
    """
    if len(patient_quest) == 0:
        return None
    
    # Make copies
    quest = patient_quest.copy()
    inhaler = patient_inhaler.copy()
    
    # Join questionnaire with inhaler data
    merged = _join_questionnaire_with_inhaler(
        quest,
        inhaler,
        timestamp_window,
        use_daily_max,
        use_calendar_days
    )
    
    # Apply categorization
    categorized = _categorize_inhaler_usage(
        merged,
        categorization_method
    )
    
    # Filter out zero usage entries if configured
    if filter_zeros:
        categorized = categorized[
            (categorized['inhaler_usage'] > 0) | 
            (categorized['daily_relief_inhaler'] > 0)
        ]
    
    if len(categorized) == 0:
        return None
    
    return categorized


def _join_questionnaire_with_inhaler(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    timestamp_window: int,
    use_daily_max_windows: bool,
    use_calendar_days: bool
) -> pd.DataFrame:
    """
    Copy of _join_questionnaire_with_inhaler from per_patient_correlation_analysis.py
    """
    if len(inhaler_df) == 0:
        # No inhaler data - set usage to 0
        questionnaire_df['inhaler_usage'] = 0
        return questionnaire_df
    
    # Convert time strings to datetime (Exact same logic)
    reference_date = pd.Timestamp('2000-01-01')
    
    def combine_date_time(row):
        base_date = reference_date + pd.Timedelta(days=row['date'])
        time_obj = pd.to_datetime(row['time']).time()
        return pd.Timestamp.combine(base_date.date(), time_obj)
    
    inhaler_df['timestamp'] = inhaler_df.apply(combine_date_time, axis=1)
    questionnaire_df['timestamp'] = questionnaire_df.apply(combine_date_time, axis=1)
    
    def aggregate_window(row):
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
        
        return len(inhaler_df[time_mask])
    
    questionnaire_df['inhaler_usage'] = questionnaire_df.apply(aggregate_window, axis=1)
    return questionnaire_df


def _categorize_inhaler_usage(
    df: pd.DataFrame,
    categorization_method: str
) -> pd.DataFrame:
    """
    Copy of _categorize_inhaler_usage from per_patient_correlation_analysis.py
    """
    categorized_df = df.copy()
    
    # Define categorization functions (Exact same logic)
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
        above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage']
        midpoint_for_greater_than_12 = above_12.mean() if len(above_12) > 0 else 12
        
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
        above_12 = df[df['inhaler_usage'] > 12]['inhaler_usage']
        if len(above_12) > 0 and above_12.max() > 24:
            sd_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].std()
            median_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].median()
            upper_bound_for_greater_than_12 = 3 * sd_above_12 + median_above_12
        else:
            upper_bound_for_greater_than_12 = above_12.max() if len(above_12) > 0 else 12
        
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
    
    # Apply categorization to BOTH columns (Exact same as original)
    categorized_df['inhaler_usage'] = categorized_df['inhaler_usage'].apply(categorize_fn)
    categorized_df['daily_relief_inhaler'] = categorized_df['daily_relief_inhaler'].apply(categorize_fn)
    
    return categorized_df
