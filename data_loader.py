import logging
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
import pandas as pd
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)

class DataJoiningMethod(Enum):
    TIMESTAMP_LEVEL = "timestamp_level"

class CategorizationMethod(Enum):
    ONE_HOT = "one_hot"
    LOWER_BOUND = "lower_bound"
    MIDPOINT = "midpoint" # >=12 would use 12 as representative
    MIDPOINT_WITH_INHALER = "midpoint_with_inhaler" # midpoint representation for >=12 would require checking inhaler data too  
    UPPER_BOUND = "upper_bound" # >=12 would still use 12 as representative
    UPPER_BOUND_WITH_INHALER = "upper_bound_with_inhaler" # upper bound representation for >=12 would require checking inhaler data too 

@dataclass
class DataLoaderConfig:
    # Data filtering options
    use_end_date: bool = True # Whether to use the daily_end_date alone rather than the duration (from daily_start_date and daily_end_date)
    drop_dates_less_than_0: bool = True # Whether to drop rows with start dates that are less than 0
    compare_inhaler_dates: bool = True # Whether to compare the inhaler dates with the daily dates to filter out enterings where the inhaler start/end date differs from questionnaire start/end date
    min_duration_threshold: int = 0
    filter_out_zero_usage_entries: bool = False # Whether to filter out entries that have both 0 inhaler_usage and 0 daily_relief_inhaler
    
    # Time window option
    timestamp_window_hours: int = 24
    
    # New time window options
    use_daily_max_windows: bool = False  # Whether to use 24-hour chunks
    use_calendar_days: bool = False      # Whether to use calendar day boundaries
    
    # Data categorization options
    categorization_method: CategorizationMethod = CategorizationMethod.MIDPOINT
    
    # Output options
    include_demographics: bool = False
    
    # Special options for Bland-Altman analysis
    skip_inhaler_filtering: bool = False  # Skip filtering based on inhaler_end_date for users with no inhaler data
    
    # Logging options
    log_level: int = logging.WARNING  # Default to WARNING level
    
class AsthmaDataLoader:
    def __init__(
        self,
        config: DataLoaderConfig,
        patient_info_path: Union[str, Path]='anonym_aamos00_patient_info.csv',
        daily_questionnaire_path: Union[str, Path]='anonym_aamos00_dailyquestionnaire_dt.csv',
        inhaler_data_path: Union[str, Path]='anonym_aamos00_smartinhaler_dt.csv',
    ):
        self.config = config
        # Set log level from config
        logger.setLevel(self.config.log_level)
        
        logger.info("Loading data files...")
        self.patient_info = pd.read_csv(patient_info_path)
        self.daily_questionnaire = pd.read_csv(daily_questionnaire_path)
        self.inhaler_data = pd.read_csv(inhaler_data_path)
        
        logger.debug(f"Initial dataframe sizes:")
        logger.debug(f"Patient info: {len(self.patient_info)} rows")
        logger.debug(f"Daily questionnaire: {len(self.daily_questionnaire)} rows")
        logger.debug(f"Inhaler data: {len(self.inhaler_data)} rows")
        
        logger.info("Removing duplicates...")
        self.patient_info.drop_duplicates(inplace=True)
        self.daily_questionnaire.drop_duplicates(inplace=True)
        self.inhaler_data.drop_duplicates(inplace=True)
        
        logger.debug("Columns in dataframes:")
        logger.debug(f"Patient info: {self.patient_info.columns.tolist()}")
        logger.debug(f"Daily questionnaire: {self.daily_questionnaire.columns.tolist()}")
        logger.debug(f"Inhaler data: {self.inhaler_data.columns.tolist()}")

    def _filter_patients(self) -> pd.DataFrame:
        """Filter patients based on configuration settings"""
        logger.info("Filtering patients...")
        initial_count = len(self.patient_info)
        
        filtered_patients = self.patient_info.copy()
        
        if self.config.drop_dates_less_than_0:
            logger.debug("Dropping negative dates...")
            filtered_patients = filtered_patients[filtered_patients['daily_start_date'] >= 0]
            filtered_patients = filtered_patients[filtered_patients['inhaler_start_date'] >= 0]
            
        if self.config.skip_inhaler_filtering:
            logger.debug("Skipping inhaler filtering...")
        else:
            if self.config.use_end_date:
                logger.debug(f"Filtering by end date threshold: {self.config.min_duration_threshold}")
                filtered_patients = filtered_patients[filtered_patients['daily_end_date'] >= self.config.min_duration_threshold]
                filtered_patients = filtered_patients[filtered_patients['inhaler_end_date'] >= self.config.min_duration_threshold]
            else:
                logger.debug(f"Filtering by duration threshold: {self.config.min_duration_threshold}")
                filtered_patients['daily_q_duration'] = filtered_patients['daily_end_date'] - filtered_patients['daily_start_date']
                filtered_patients['inhaler_duration'] = filtered_patients['inhaler_end_date'] - filtered_patients['inhaler_start_date']
                filtered_patients = filtered_patients[filtered_patients['daily_q_duration'] >= self.config.min_duration_threshold]
                filtered_patients = filtered_patients[filtered_patients['inhaler_duration'] >= self.config.min_duration_threshold]
            
        if self.config.compare_inhaler_dates:
            logger.debug("Comparing inhaler dates with questionnaire dates...")
            filtered_patients = filtered_patients[filtered_patients['inhaler_end_date'] <= filtered_patients['daily_end_date']]
            filtered_patients = filtered_patients[filtered_patients['inhaler_start_date'] >= filtered_patients['daily_start_date']]
        
        final_count = len(filtered_patients)
        logger.info(f"Filtered patients from {initial_count} to {final_count} ({final_count/initial_count:.1%} remaining)")
        return filtered_patients
    
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
    
    def load_data(self) -> pd.DataFrame:
        """Main method to load and process all data according to config"""
        logger.info("Starting data loading process...")
        
        # 1. Filter patients
        logger.info("Step 1: Filtering patients...")
        valid_patients = self._filter_patients()
        
        # Filter the questionnaire and inhaler data
        logger.info("Filtering questionnaire and inhaler data...")
        initial_questionnaire_count = len(self.daily_questionnaire)
        initial_inhaler_count = len(self.inhaler_data)
        
        filtered_questionnaire = self.daily_questionnaire[
            self.daily_questionnaire['user_key'].isin(valid_patients['user_key'])
        ]
        filtered_inhaler = self.inhaler_data[
            self.inhaler_data['user_key'].isin(valid_patients['user_key'])
        ]
        
        logger.debug(f"Questionnaire data filtered from {initial_questionnaire_count} to {len(filtered_questionnaire)} rows")
        logger.debug(f"Inhaler data filtered from {initial_inhaler_count} to {len(filtered_inhaler)} rows")
        
        self.daily_questionnaire = filtered_questionnaire
        self.inhaler_data = filtered_inhaler
        
        # 2. Join questionnaire with inhaler data
        logger.info("Step 2: Joining questionnaire with inhaler data...")
        merged_data = self._join_questionnaire_with_inhaler_data()
        logger.debug(f"Merged data size: {len(merged_data)} rows")
        
        # 3. Apply categorization
        logger.info(f"Step 3: Applying {self.config.categorization_method.value} categorization...")
        final_data = self._categorize_inhaler_usage(merged_data)
        
        # 4. Optionally filter out entries that have both 0 inhaler_usage and 0 daily_relief_inhaler
        if self.config.filter_out_zero_usage_entries:
            final_data = final_data[
                (final_data['inhaler_usage'] > 0) | (final_data['daily_relief_inhaler'] > 0)
            ]   

        # 5. Add demographics if requested
        if self.config.include_demographics:
            logger.info("Step 4: Adding demographic data...")
            final_data = pd.merge(
                final_data,
                valid_patients[['user_key', 'sex', 'age_range', 'bmi_range', 'smoker', 'race', 'severity', 'inhaler_end_date', 'region']],
                on='user_key'
            )
            logger.debug(f"Final data size with demographics: {len(final_data)} rows")
        
        logger.info("Data loading complete!")
        return final_data 