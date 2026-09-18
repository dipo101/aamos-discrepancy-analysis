import logging
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
import pandas as pd
from pathlib import Path

from aamos_concordance import (
    CategorizationMethod,
    categorize_inhaler_usage,
    filter_zero_usage,
    join_multi_patient,
    load_raw,
)
from aamos_concordance.data import sha256_of

# Set up logger
logger = logging.getLogger(__name__)

class DataJoiningMethod(Enum):
    TIMESTAMP_LEVEL = "timestamp_level"

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
        patient_info_path: Optional[Union[str, Path]] = None,
        daily_questionnaire_path: Optional[Union[str, Path]] = None,
        inhaler_data_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """Load the raw frames.

        With no explicit file paths the raw data is located via
        :func:`aamos_concordance.load_raw` (``$AAMOS_DATA_DIR``, then
        ``data/raw/``, then the repository root) and verified against the
        committed manifest. Explicit paths bypass the manifest but are still
        hashed so that ``self.data_provenance`` names the bytes that were read.
        """
        self.config = config
        # Set log level from config
        logger.setLevel(self.config.log_level)
        
        logger.info("Loading data files...")
        explicit = (patient_info_path, daily_questionnaire_path, inhaler_data_path)
        if any(p is not None for p in explicit):
            if not all(p is not None for p in explicit):
                raise ValueError("Pass all three raw file paths or none of them.")
            self.patient_info = pd.read_csv(patient_info_path)
            self.daily_questionnaire = pd.read_csv(daily_questionnaire_path)
            self.inhaler_data = pd.read_csv(inhaler_data_path)
            self.data_provenance = {
                "data_dir": None,
                "sha256": {Path(p).name: sha256_of(Path(p)) for p in explicit},
                "manifest_verified": False,
            }
        else:
            raw = load_raw(data_dir)
            self.patient_info = raw.patient_info
            self.daily_questionnaire = raw.questionnaire
            self.inhaler_data = raw.inhaler
            self.data_provenance = raw.provenance()
        
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
        """Join questionnaire data with inhaler data using the configured time window.

        Delegates to :func:`aamos_concordance.join_multi_patient`, which
        matches on ``user_key`` and counts inhaler records in each
        questionnaire row's window.
        """
        return join_multi_patient(
            self.daily_questionnaire,
            self.inhaler_data,
            self.config.timestamp_window_hours,
            self.config.use_daily_max_windows,
            self.config.use_calendar_days,
        )
    
    def _categorize_inhaler_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the selected categorization method to both usage columns.

        The data-driven ``*_WITH_INHALER`` methods derive their top-category
        value from the whole (multi-patient) frame passed in, as this loader
        always did. If that frame has no device usage >= 12 the top category
        falls back to 12 (see aamos_concordance.categorization); the
        pre-refactor loader produced NaN there, a case that cannot occur on
        the AAMOS-00 data.
        """
        return categorize_inhaler_usage(df, self.config.categorization_method)
    
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
            final_data = filter_zero_usage(final_data)

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