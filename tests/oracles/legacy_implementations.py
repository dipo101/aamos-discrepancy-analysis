"""Verbatim copies of the three pre-refactor implementations, frozen from commit 55a606a.

These are test oracles. Do not edit. The shared package in aamos_concordance/ must
reproduce their behaviour exactly; tests/test_join_equivalence.py checks that.
"""
import logging
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CategorizationMethod(Enum):
    ONE_HOT = 'one_hot'
    LOWER_BOUND = 'lower_bound'
    MIDPOINT = 'midpoint'
    MIDPOINT_WITH_INHALER = 'midpoint_with_inhaler'
    UPPER_BOUND = 'upper_bound'
    UPPER_BOUND_WITH_INHALER = 'upper_bound_with_inhaler'


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class DataLoaderOracle:
    """data_loader.AsthmaDataLoader join/categorise, verbatim (55a606a)."""
    def __init__(self, config, inhaler_data, daily_questionnaire):
        self.config = config
        self.inhaler_data = inhaler_data
        self.daily_questionnaire = daily_questionnaire

    def _join_questionnaire_with_inhaler_data(self) -> pd.DataFrame:
        """Join questionnaire data with inhaler data using specified time windows."""
        inhaler_df = self.inhaler_data.copy()
        questionnaire_df = self.daily_questionnaire.copy()
        
        # Convert time strings to datetime.time objects and combine with date
        # Note: date is days from study start, so we'll use a reference date and add timedelta
        reference_date = pd.Timestamp('2000-01-01')  # arbitrary reference date
        
        def combine_date_time(row):
            # Convert days to timedelta and add to reference date
            base_date = reference_date + pd.Timedelta(days=row['date'])
            # Parse time string and combine with base date
            time_obj = pd.to_datetime(row['time']).time()
            return pd.Timestamp.combine(base_date.date(), time_obj)
        
        # Create timestamp columns
        inhaler_df['timestamp'] = inhaler_df.apply(combine_date_time, axis=1)
        questionnaire_df['timestamp'] = questionnaire_df.apply(combine_date_time, axis=1)
        
        def aggregate_window(row):
            user_mask = inhaler_df['user_key'] == row['user_key']
            
            if self.config.use_calendar_days:
                # Calculate target day based on timestamp_windows
                days_back = self.config.timestamp_window_hours // 24
                target_day = row['date'] - days_back
                time_mask = inhaler_df['date'] == target_day
                
            else:
                window_hours = pd.Timedelta(hours=self.config.timestamp_window_hours)
                
                if self.config.use_daily_max_windows:
                    # Use 24-hour chunks
                    window_start = row['timestamp'] - window_hours
                    window_end = window_start + pd.Timedelta(hours=24)
                    time_mask = (
                        (inhaler_df['timestamp'] >= window_start) & 
                        (inhaler_df['timestamp'] < window_end)
                    )
                else:
                    # Original behavior
                    time_mask = (
                        (inhaler_df['timestamp'] <= row['timestamp']) & 
                        (inhaler_df['timestamp'] >= row['timestamp'] - window_hours)
                    )
            
            relevant_records = inhaler_df[user_mask & time_mask]
            return len(relevant_records)  # Count occurrences instead of summing usage_count
        
        questionnaire_df['inhaler_usage'] = questionnaire_df.apply(aggregate_window, axis=1)
        return questionnaire_df
    

    def _categorize_inhaler_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the selected categorization method"""
        categorized_df = df.copy()
        
        match self.config.categorization_method:
            case CategorizationMethod.ONE_HOT:
                categorize_fn = lambda x: 1 if x is not None and x >= 1 else 0
                
            case CategorizationMethod.LOWER_BOUND:
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
                        
            case CategorizationMethod.MIDPOINT:
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
                        
            case CategorizationMethod.UPPER_BOUND:
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
                        
            case CategorizationMethod.MIDPOINT_WITH_INHALER:
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
                        
            case CategorizationMethod.UPPER_BOUND_WITH_INHALER: 
                max_above_12 = df[df['inhaler_usage'] > 12]['inhaler_usage'].max()
                if max_above_12 > 24:
                    # find standard deviation for points above 12
                    sd_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].std()
                    # define max as 3*sd + median of above 12s
                    median_above_12 = df[df['inhaler_usage'] >= 12]['inhaler_usage'].median()
                    upper_bound_for_greater_than_12 = 3*sd_above_12 + median_above_12
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
                        
            case _:
                raise ValueError(f"Invalid categorization method: {self.config.categorization_method}")
        
        # Apply the categorization function to both columns
        categorized_df['inhaler_usage'] = categorized_df['inhaler_usage'].apply(categorize_fn)
        categorized_df['daily_relief_inhaler'] = categorized_df['daily_relief_inhaler'].apply(categorize_fn)

        return categorized_df
    


class PerPatientOracle:
    """per_patient_correlation_analysis.PerPatientDataLoader join/categorise, verbatim (55a606a)."""
    def _join_questionnaire_with_inhaler(
        self,
        questionnaire_df: pd.DataFrame,
        inhaler_df: pd.DataFrame,
        timestamp_window: int,
        use_daily_max_windows: bool,
        use_calendar_days: bool
    ) -> pd.DataFrame:
        """Join questionnaire data with inhaler data using time windows"""
        
        if len(inhaler_df) == 0:
            # No inhaler data - set usage to 0
            questionnaire_df['inhaler_usage'] = 0
            return questionnaire_df
        
        # Convert time strings to datetime
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
        self,
        df: pd.DataFrame,
        categorization_method: CategorizationMethod
    ) -> pd.DataFrame:
        """Apply categorization to inhaler usage values"""
        categorized_df = df.copy()
        
        # Define categorization functions
        if categorization_method == CategorizationMethod.ONE_HOT:
            categorize_fn = lambda x: 1 if x is not None and x >= 1 else 0
            
        elif categorization_method == CategorizationMethod.LOWER_BOUND:
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
                    
        elif categorization_method == CategorizationMethod.MIDPOINT:
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
                    
        elif categorization_method == CategorizationMethod.UPPER_BOUND:
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
                    
        elif categorization_method == CategorizationMethod.MIDPOINT_WITH_INHALER:
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
                    
        elif categorization_method == CategorizationMethod.UPPER_BOUND_WITH_INHALER:
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
            raise ValueError(f"Invalid categorization method: {categorization_method}")
        
        # Apply categorization to both columns
        categorized_df['inhaler_usage'] = categorized_df['inhaler_usage'].apply(categorize_fn)
        categorized_df['daily_relief_inhaler'] = categorized_df['daily_relief_inhaler'].apply(categorize_fn)
        
        return categorized_df




# ---- job-worker data_loader_simplified.py, verbatim (55a606a), minus the GCS loader ----
def jobworker_join_questionnaire_with_inhaler(
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


def jobworker_categorize_inhaler_usage(
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



# ---- job-worker main.py, verbatim (55a606a): config generation, permutation kernel, summary block ----
import time
from scipy import stats

def jobworker_generate_param_combinations():
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



def jobworker_run_single_permutation(patient_id, perm_idx, questionnaire_df, inhaler_df, 
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
            merged_df = jobworker_join_questionnaire_with_inhaler(
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
            categorized_df = jobworker_categorize_inhaler_usage(
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



def jobworker_summarize(correlations, i=0):
    all_correlations = []
    valid_count = 0
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
    return all_correlations[0]


def perpatient_generate_param_combinations(cfg):
    """per_patient_correlation_analysis.run_analysis combination loop, verbatim (55a606a)."""
    self = _Cfg(config=cfg)
    param_combinations = []
    seen_effective_configs = set()
    
    for window in self.config.timestamp_windows:
        for use_daily_max in self.config.use_daily_max_windows:
            for use_calendar in self.config.use_calendar_days:
                for filter_zeros in self.config.filter_out_zero_usage_entries:
                    for cat_method in self.config.categorization_methods:
                        # Compute effective config key to detect duplicates
                        if use_calendar:
                            # Calendar mode: daily_max is ignored, 24 and 36 both map to day 1
                            effective_window = window // 24  # 12->0, 24->1, 36->1, 48->2
                            effective_key = ('calendar', effective_window, filter_zeros, cat_method)
                        else:
                            # Non-calendar mode: all params matter
                            mode = 'fixed_chunk' if use_daily_max else 'rolling'
                            effective_key = (mode, window, filter_zeros, cat_method)
                        
                        # Skip if we've seen this effective config
                        if effective_key in seen_effective_configs:
                            continue
                        seen_effective_configs.add(effective_key)
                        
                        param_combinations.append(
                            (window, use_daily_max, use_calendar, filter_zeros, cat_method)
                        )
    
    return param_combinations
