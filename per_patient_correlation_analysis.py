import logging
from itertools import product
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from aamos_concordance import (
    CategorizationMethod,
    categorize_inhaler_usage,
    dedupe_combinations,
    filter_zero_usage,
    join_questionnaire_with_inhaler,
    load_raw,
    write_sidecar,
)
from aamos_concordance.data import sha256_of

# Set up logger
logger = logging.getLogger(__name__)

# The pre-refactor copy of the categoriser in this script returned 12 for the
# data-driven top category when a patient had no usage >= 12, whereas
# data_loader.py and the job-worker returned NaN. The published per-patient
# results were produced with 12, so this script keeps that behaviour.
TOP_CATEGORY_FALLBACK = 12

@dataclass
class PerPatientCorrelationConfig:
    """Configuration for per-patient correlation analysis"""
    min_duration_threshold: int = 0
    use_end_date: bool = True  # Whether to use end_date or duration for patient filtering
    compare_inhaler_dates: bool = False
    
    # Parameters to vary per patient
    timestamp_windows: List[int] = (12, 24, 36, 48)  # hours
    use_daily_max_windows: List[bool] = (False, True)
    use_calendar_days: List[bool] = (False, True)
    filter_out_zero_usage_entries: List[bool] = (False, True)
    categorization_methods: List[CategorizationMethod] = (
        CategorizationMethod.ONE_HOT,
        CategorizationMethod.LOWER_BOUND,
        CategorizationMethod.MIDPOINT,
        CategorizationMethod.UPPER_BOUND,
        CategorizationMethod.MIDPOINT_WITH_INHALER,
        CategorizationMethod.UPPER_BOUND_WITH_INHALER
    )
    
    # Analysis settings
    log_level: int = logging.INFO
    min_samples_per_patient: int = 5  # Minimum data points needed per patient for correlation

@dataclass
class PatientCorrelationResult:
    """Stores correlation results for a single patient with specific parameters"""
    user_key: str
    timestamp_window: int
    use_daily_max_windows: bool
    use_calendar_days: bool
    filter_out_zero_usage_entries: bool
    categorization_method: CategorizationMethod
    spearman_corr: float
    spearman_p: float
    pearson_corr: float
    pearson_p: float
    spearman_z: float  # Fisher Z-transformed Spearman
    pearson_z: float   # Fisher Z-transformed Pearson
    sample_size: int

class PerPatientDataLoader:
    """Loads and processes data for individual patients"""
    
    def __init__(
        self,
        config: PerPatientCorrelationConfig,
        patient_info_path: Optional[str] = None,
        daily_questionnaire_path: Optional[str] = None,
        inhaler_data_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        """Load the raw frames (see AsthmaDataLoader for the lookup and manifest rules)."""
        self.config = config
        logger.setLevel(config.log_level)
        
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
        
        # Remove duplicates
        self.patient_info.drop_duplicates(inplace=True)
        self.daily_questionnaire.drop_duplicates(inplace=True)
        self.inhaler_data.drop_duplicates(inplace=True)
        
        logger.info(f"Loaded {len(self.patient_info)} patients, "
                   f"{len(self.daily_questionnaire)} questionnaire entries, "
                   f"{len(self.inhaler_data)} inhaler records")
    
    def get_eligible_patients(self) -> List[str]:
        """Get list of patients meeting the fixed selection criteria"""
        logger.info("Filtering patients based on fixed criteria...")
        filtered_patients = self.patient_info.copy()
        
        # Apply fixed filtering criteria
        initial_count = len(filtered_patients)
        
        # Note: We do NOT filter by negative dates (drop_dates_less_than_0 = False)
        # We include all patients regardless of start date values
        
        # Filter by duration/end date threshold based on use_end_date setting
        if self.config.use_end_date:
            # Use end_date directly
            filtered_patients = filtered_patients[
                (filtered_patients['daily_end_date'] >= self.config.min_duration_threshold)
            ]
        else:
            # Use duration (end_date - start_date)
            filtered_patients['daily_q_duration'] = (
                filtered_patients['daily_end_date'] - filtered_patients['daily_start_date']
            )
            filtered_patients = filtered_patients[
                filtered_patients['daily_q_duration'] >= self.config.min_duration_threshold
            ]
        
        # Compare inhaler dates with questionnaire dates if configured
        if self.config.compare_inhaler_dates:
            filtered_patients = filtered_patients[
                (filtered_patients['inhaler_end_date'] <= filtered_patients['daily_end_date']) &
                (filtered_patients['inhaler_start_date'] >= filtered_patients['daily_start_date'])
            ]
        
        final_count = len(filtered_patients)
        logger.info(f"Filtered patients from {initial_count} to {final_count} "
                   f"({final_count/initial_count:.1%} remaining)")
        
        return filtered_patients['user_key'].tolist()
    
    def load_patient_data(
        self,
        user_key: str,
        timestamp_window: int,
        use_daily_max_windows: bool,
        use_calendar_days: bool,
        filter_out_zero_usage: bool,
        categorization_method: CategorizationMethod
    ) -> Optional[pd.DataFrame]:
        """Load and process data for a specific patient with given parameters"""
        
        # Filter data for this patient
        patient_questionnaire = self.daily_questionnaire[
            self.daily_questionnaire['user_key'] == user_key
        ].copy()
        patient_inhaler = self.inhaler_data[
            self.inhaler_data['user_key'] == user_key
        ].copy()
        
        if len(patient_questionnaire) == 0:
            logger.warning(f"No questionnaire data for patient {user_key}")
            return None
        
        # Join questionnaire with inhaler data
        merged_data = self._join_questionnaire_with_inhaler(
            patient_questionnaire,
            patient_inhaler,
            timestamp_window,
            use_daily_max_windows,
            use_calendar_days
        )
        
        # Apply categorization
        categorized_data = self._categorize_inhaler_usage(
            merged_data,
            categorization_method
        )
        
        # Filter out zero usage entries if configured
        if filter_out_zero_usage:
            categorized_data = filter_zero_usage(categorized_data)
        
        return categorized_data
    
    def _join_questionnaire_with_inhaler(
        self,
        questionnaire_df: pd.DataFrame,
        inhaler_df: pd.DataFrame,
        timestamp_window: int,
        use_daily_max_windows: bool,
        use_calendar_days: bool
    ) -> pd.DataFrame:
        """Join one patient's questionnaire rows with their inhaler records (shared implementation)."""
        return join_questionnaire_with_inhaler(
            questionnaire_df, inhaler_df, timestamp_window, use_daily_max_windows, use_calendar_days
        )
    
    def _categorize_inhaler_usage(
        self,
        df: pd.DataFrame,
        categorization_method: CategorizationMethod
    ) -> pd.DataFrame:
        """Apply categorization to both usage columns (shared implementation)."""
        return categorize_inhaler_usage(
            df, categorization_method, top_category_fallback=TOP_CATEGORY_FALLBACK
        )

class PerPatientCorrelationAnalysis:
    """Analyzes correlations on a per-patient basis"""
    
    def __init__(self, config: PerPatientCorrelationConfig, data_dir: Optional[Path] = None):
        """``data_dir`` optionally points at the folder holding the three raw CSVs;
        by default they are located via aamos_concordance.load_raw."""
        self.config = config
        logger.setLevel(config.log_level)
        self.data_loader = PerPatientDataLoader(config, data_dir=data_dir)
    
    def run_analysis(self) -> pd.DataFrame:
        """Run correlation analysis for all patients with all parameter combinations"""
        logger.info("Starting per-patient correlation analysis...")
        
        # Get eligible patients
        eligible_patients = self.data_loader.get_eligible_patients()
        logger.info(f"Analyzing {len(eligible_patients)} eligible patients")
        
        # Generate UNIQUE parameter combinations (132 instead of 192)
        # Note: 60 combinations are redundant because:
        # - When use_calendar_days=True, use_daily_max_windows is ignored
        # - When use_calendar_days=True, timestamp_window 24 and 36 both map to 1 day
        param_combinations = dedupe_combinations(
            self.config.timestamp_windows,
            self.config.use_daily_max_windows,
            self.config.use_calendar_days,
            self.config.filter_out_zero_usage_entries,
            self.config.categorization_methods,
        )
        
        total_combinations = len(param_combinations)
        logger.info(f"Testing {total_combinations} UNIQUE parameter combinations per patient (de-duplicated from 192)")
        
        results = []
        
        # Analyze each patient
        for patient_idx, user_key in enumerate(eligible_patients, 1):
            logger.info(f"\nAnalyzing patient {patient_idx}/{len(eligible_patients)}: {user_key}")
            
            patient_results_count = 0
            
            # Try all parameter combinations for this patient
            for window, use_daily_max, use_calendar, filter_zeros, cat_method in param_combinations:
                try:
                    # Load patient data with these parameters
                    data = self.data_loader.load_patient_data(
                        user_key=user_key,
                        timestamp_window=window,
                        use_daily_max_windows=use_daily_max,
                        use_calendar_days=use_calendar,
                        filter_out_zero_usage=filter_zeros,
                        categorization_method=cat_method
                    )
                    
                    if data is None or len(data) < self.config.min_samples_per_patient:
                        continue
                    
                    # Calculate correlations
                    try:
                        spearman_corr, spearman_p = stats.spearmanr(
                            data['inhaler_usage'],
                            data['daily_relief_inhaler']
                        )
                        
                        pearson_corr, pearson_p = stats.pearsonr(
                            data['inhaler_usage'],
                            data['daily_relief_inhaler']
                        )
                        
                        # Apply Fisher Z-transformation
                        spearman_z = np.arctanh(spearman_corr)
                        pearson_z = np.arctanh(pearson_corr)
                        
                        # Store result
                        result = PatientCorrelationResult(
                            user_key=user_key,
                            timestamp_window=window,
                            use_daily_max_windows=use_daily_max,
                            use_calendar_days=use_calendar,
                            filter_out_zero_usage_entries=filter_zeros,
                            categorization_method=cat_method,
                            spearman_corr=spearman_corr,
                            spearman_p=spearman_p,
                            pearson_corr=pearson_corr,
                            pearson_p=pearson_p,
                            spearman_z=spearman_z,
                            pearson_z=pearson_z,
                            sample_size=len(data)
                        )
                        results.append(result)
                        patient_results_count += 1
                        
                    except (ValueError, Exception) as e:
                        logger.debug(f"Error calculating correlations for patient {user_key}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error processing patient {user_key} with parameters: {str(e)}")
                    continue
            
            logger.info(f"  Obtained {patient_results_count} valid results for patient {user_key}")
        
        logger.info(f"\nAnalysis complete! Total results: {len(results)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'user_key': r.user_key,
                'timestamp_window': r.timestamp_window,
                'use_daily_max_windows': r.use_daily_max_windows,
                'use_calendar_days': r.use_calendar_days,
                'filter_out_zero_usage_entries': r.filter_out_zero_usage_entries,
                'categorization_method': r.categorization_method.value,
                'spearman_corr': r.spearman_corr,
                'spearman_p': r.spearman_p,
                'spearman_z': r.spearman_z,
                'pearson_corr': r.pearson_corr,
                'pearson_p': r.pearson_p,
                'pearson_z': r.pearson_z,
                'sample_size': r.sample_size
            }
            for r in results
        ])
        
        return results_df
    
    def plot_per_patient_boxplots(
        self,
        results_df: pd.DataFrame,
        correlation_type: str = 'spearman',
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        use_fisher_z: bool = True
    ):
        """Create box plots showing distribution of correlations for each patient
        
        Args:
            results_df: DataFrame with correlation results
            correlation_type: 'spearman' or 'pearson'
            save_dir: Directory to save plots
            figsize: Figure size
            use_fisher_z: If True, plot Fisher Z-transformed values (recommended)
        """
        
        if correlation_type not in ['spearman', 'pearson']:
            raise ValueError("correlation_type must be 'spearman' or 'pearson'")
        
        if use_fisher_z:
            corr_col = f'{correlation_type}_z'
            value_label = 'Fisher Z'
        else:
            corr_col = f'{correlation_type}_corr'
            value_label = 'Correlation Coefficient'
        
        # Get unique patients
        all_patients = sorted(results_df['user_key'].unique())
        
        logger.info(f"Creating {correlation_type} box plots for {len(all_patients)} patients...")
        
        # Prepare data for each patient, filtering out those with all NaN/inf values
        patient_data = []
        patients = []
        skipped_patients = []
        
        for patient in all_patients:
            patient_results = results_df[results_df['user_key'] == patient][corr_col]
            # Remove NaN and inf values
            valid_results = patient_results[np.isfinite(patient_results)]
            
            if len(valid_results) > 0:
                patient_data.append(valid_results)
                patients.append(patient)
            else:
                skipped_patients.append(patient)
                logger.warning(f"Skipping patient {patient}: no valid {correlation_type} values")
        
        if len(patients) == 0:
            logger.error(f"No patients with valid {correlation_type} values to plot!")
            return
        
        if skipped_patients:
            logger.info(f"Skipped {len(skipped_patients)} patients with no valid values: {skipped_patients}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        bp = ax.boxplot(
            patient_data,
            labels=patients,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=5),
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )
        
        # Customize plot
        ax.set_xlabel('Patient ID', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{correlation_type.title()} {value_label}', fontsize=12, fontweight='bold')
        title_suffix = ' (Fisher Z-Transformed)' if use_fisher_z else ''
        ax.set_title(
            f'Distribution of {correlation_type.title()} Correlations Across Parameter Configurations\n'
            f'(Per Patient){title_suffix}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add summary statistics as text
        stats_text = []
        for i, patient in enumerate(patients, 1):
            patient_results = results_df[results_df['user_key'] == patient][corr_col]
            valid_results = patient_results[np.isfinite(patient_results)]
            n_configs = len(patient_results)
            n_valid = len(valid_results)
            mean_corr = valid_results.mean() if n_valid > 0 else np.nan
            median_corr = valid_results.median() if n_valid > 0 else np.nan
            std_corr = valid_results.std() if n_valid > 0 else np.nan
            
            stats_text.append(
                f"Patient {patient}: n={n_configs} ({n_valid} valid), "
                f"mean={mean_corr:.3f}, median={median_corr:.3f}, std={std_corr:.3f}"
            )
        
        # Add text box with summary (using only valid values)
        all_valid_values = results_df[corr_col][np.isfinite(results_df[corr_col])]
        textstr = f"Total patients: {len(patients)} (plotted)\n"
        if skipped_patients:
            textstr += f"Skipped: {len(skipped_patients)} (no valid values)\n"
        # Show intended number of unique configurations (132) per methodology
        textstr += f"Configurations per patient: 132\n"
        textstr += f"Overall mean {value_label.lower()}: {all_valid_values.mean():.3f}\n"
        textstr += f"Overall std: {all_valid_values.std():.3f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            suffix = '_fisher_z' if use_fisher_z else ''
            save_path = f"{save_dir}/per_patient_{correlation_type}{suffix}_boxplots.png"
            logger.info(f"Saving plot to {save_path}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # Also save detailed statistics
            stats_path = f"{save_dir}/per_patient_{correlation_type}{suffix}_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write('\n'.join(stats_text))
            logger.info(f"Saved statistics to {stats_path}")
        
        plt.show()
        logger.info("Plot creation complete")
    
    def create_summary_statistics(self, results_df: pd.DataFrame, exclude_invalid: bool = True) -> pd.DataFrame:
        """
        Create summary statistics for each patient using Fisher Z-transformed values.
        
        Args:
            results_df: DataFrame with correlation results
            exclude_invalid: If True, exclude patients with no valid correlations
        
        Returns:
            DataFrame with summary statistics per patient
        """
        
        summary_stats = []
        
        for patient in results_df['user_key'].unique():
            patient_results = results_df[results_df['user_key'] == patient]
            
            # Count valid (finite) correlations
            n_valid_spearman = patient_results['spearman_z'][
                np.isfinite(patient_results['spearman_z'])
            ].count()
            n_valid_pearson = patient_results['pearson_z'][
                np.isfinite(patient_results['pearson_z'])
            ].count()
            n_valid_correlations = min(n_valid_spearman, n_valid_pearson)
            
            # Skip patients with no valid correlations if requested
            if exclude_invalid and n_valid_correlations == 0:
                logger.warning(f"Excluding patient {patient} from summary: no valid correlations")
                continue
            
            # Get finite values only for statistics
            valid_spearman_z = patient_results['spearman_z'][
                np.isfinite(patient_results['spearman_z'])
            ]
            valid_pearson_z = patient_results['pearson_z'][
                np.isfinite(patient_results['pearson_z'])
            ]
            valid_spearman_corr = patient_results['spearman_corr'][
                np.isfinite(patient_results['spearman_corr'])
            ]
            valid_pearson_corr = patient_results['pearson_corr'][
                np.isfinite(patient_results['pearson_corr'])
            ]
            
            stats = {
                'user_key': patient,
                'n_configurations': len(patient_results),
                'n_valid_spearman': n_valid_spearman,
                'n_valid_pearson': n_valid_pearson,
                'n_valid_correlations': n_valid_correlations,
                'pct_valid': (n_valid_correlations / len(patient_results) * 100) if len(patient_results) > 0 else 0,
                # Fisher Z-transformed statistics (primary) - using valid values only
                'mean_spearman_z': valid_spearman_z.mean() if len(valid_spearman_z) > 0 else np.nan,
                'median_spearman_z': valid_spearman_z.median() if len(valid_spearman_z) > 0 else np.nan,
                'std_spearman_z': valid_spearman_z.std() if len(valid_spearman_z) > 0 else np.nan,
                'min_spearman_z': valid_spearman_z.min() if len(valid_spearman_z) > 0 else np.nan,
                'max_spearman_z': valid_spearman_z.max() if len(valid_spearman_z) > 0 else np.nan,
                'mean_pearson_z': valid_pearson_z.mean() if len(valid_pearson_z) > 0 else np.nan,
                'median_pearson_z': valid_pearson_z.median() if len(valid_pearson_z) > 0 else np.nan,
                'std_pearson_z': valid_pearson_z.std() if len(valid_pearson_z) > 0 else np.nan,
                'min_pearson_z': valid_pearson_z.min() if len(valid_pearson_z) > 0 else np.nan,
                'max_pearson_z': valid_pearson_z.max() if len(valid_pearson_z) > 0 else np.nan,
                # Raw correlation statistics (for reference) - using valid values only
                'mean_spearman': valid_spearman_corr.mean() if len(valid_spearman_corr) > 0 else np.nan,
                'median_spearman': valid_spearman_corr.median() if len(valid_spearman_corr) > 0 else np.nan,
                'std_spearman': valid_spearman_corr.std() if len(valid_spearman_corr) > 0 else np.nan,
                'min_spearman': valid_spearman_corr.min() if len(valid_spearman_corr) > 0 else np.nan,
                'max_spearman': valid_spearman_corr.max() if len(valid_spearman_corr) > 0 else np.nan,
                'mean_pearson': valid_pearson_corr.mean() if len(valid_pearson_corr) > 0 else np.nan,
                'median_pearson': valid_pearson_corr.median() if len(valid_pearson_corr) > 0 else np.nan,
                'std_pearson': valid_pearson_corr.std() if len(valid_pearson_corr) > 0 else np.nan,
                'min_pearson': valid_pearson_corr.min() if len(valid_pearson_corr) > 0 else np.nan,
                'max_pearson': valid_pearson_corr.max() if len(valid_pearson_corr) > 0 else np.nan,
                'mean_sample_size': patient_results['sample_size'].mean()
            }
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def create_patient_exclusion_report(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a report of patients excluded due to insufficient valid correlations.
        
        Args:
            results_df: DataFrame with correlation results
        
        Returns:
            DataFrame with exclusion details per patient
        """
        
        exclusion_report = []
        
        for patient in results_df['user_key'].unique():
            patient_results = results_df[results_df['user_key'] == patient]
            
            # Count valid correlations
            n_valid_spearman = patient_results['spearman_z'][
                np.isfinite(patient_results['spearman_z'])
            ].count()
            n_valid_pearson = patient_results['pearson_z'][
                np.isfinite(patient_results['pearson_z'])
            ].count()
            n_valid_correlations = min(n_valid_spearman, n_valid_pearson)
            
            # Count NaN values
            n_nan_spearman = patient_results['spearman_corr'].isna().sum()
            n_nan_pearson = patient_results['pearson_corr'].isna().sum()
            
            # Check for constant inputs (correlation = NaN but sample size > 0)
            has_data = patient_results['sample_size'].mean() > 0
            all_correlations_nan = (n_valid_correlations == 0)
            
            # Determine exclusion reason
            if all_correlations_nan:
                if has_data:
                    exclusion_reason = "Constant input (likely missing smart inhaler data - all values 0)"
                else:
                    exclusion_reason = "No data"
                exclusion_status = "EXCLUDED"
            elif n_valid_correlations < len(patient_results) * 0.5:
                exclusion_reason = f"Low valid correlation rate ({n_valid_correlations}/{len(patient_results)})"
                exclusion_status = "WARNING"
            else:
                exclusion_reason = "Valid"
                exclusion_status = "INCLUDED"
            
            report_entry = {
                'user_key': patient,
                'exclusion_status': exclusion_status,
                'n_configurations': len(patient_results),
                'n_valid_spearman': n_valid_spearman,
                'n_valid_pearson': n_valid_pearson,
                'n_valid_correlations': n_valid_correlations,
                'pct_valid': (n_valid_correlations / len(patient_results) * 100) if len(patient_results) > 0 else 0,
                'mean_sample_size': patient_results['sample_size'].mean(),
                'exclusion_reason': exclusion_reason
            }
            
            exclusion_report.append(report_entry)
        
        # Sort by exclusion status (EXCLUDED first, then WARNING, then INCLUDED)
        report_df = pd.DataFrame(exclusion_report)
        status_order = {'EXCLUDED': 0, 'WARNING': 1, 'INCLUDED': 2}
        report_df['_sort_order'] = report_df['exclusion_status'].map(status_order)
        report_df = report_df.sort_values('_sort_order').drop('_sort_order', axis=1)
        
        return report_df


def main():
    """Main execution function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = PerPatientCorrelationConfig(
        # Fixed patient selection criteria
        min_duration_threshold=0,
        use_end_date=True,
        compare_inhaler_dates=False,
        
        # Parameters to vary per patient
        timestamp_windows=[12, 24, 36, 48],
        use_daily_max_windows=[False, True],
        use_calendar_days=[False, True],
        filter_out_zero_usage_entries=[False, True],
        categorization_methods=[
            CategorizationMethod.ONE_HOT,
            CategorizationMethod.LOWER_BOUND,
            CategorizationMethod.MIDPOINT,
            CategorizationMethod.UPPER_BOUND,
            CategorizationMethod.MIDPOINT_WITH_INHALER,
            CategorizationMethod.UPPER_BOUND_WITH_INHALER
        ],
        
        log_level=logging.INFO,
        min_samples_per_patient=5
    )
    
    # Create analyzer
    analyzer = PerPatientCorrelationAnalysis(config)
    
    # Run analysis
    results_df = analyzer.run_analysis()
    
    # Create output directory
    output_dir = Path("./results/per_patient_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "per_patient_correlation_results.csv"
    results_df.to_csv(results_path, index=False)
    write_sidecar(results_path, config=config, data=analyzer.data_loader.data_provenance)
    logger.info(f"Saved detailed results to {results_path}")
    
    # Create patient exclusion report
    exclusion_df = analyzer.create_patient_exclusion_report(results_df)
    exclusion_path = output_dir / "per_patient_exclusion_report.csv"
    exclusion_df.to_csv(exclusion_path, index=False)
    write_sidecar(exclusion_path, config=config, data=analyzer.data_loader.data_provenance)
    logger.info(f"Saved patient exclusion report to {exclusion_path}")
    
    # Print exclusion summary
    print("\n" + "="*80)
    print("PATIENT EXCLUSION SUMMARY")
    print("="*80)
    excluded = exclusion_df[exclusion_df['exclusion_status'] == 'EXCLUDED']
    warnings = exclusion_df[exclusion_df['exclusion_status'] == 'WARNING']
    included = exclusion_df[exclusion_df['exclusion_status'] == 'INCLUDED']
    
    print(f"\nINCLUDED: {len(included)} patients with valid correlations")
    if len(warnings) > 0:
        print(f"WARNING: {len(warnings)} patients with low valid correlation rate")
        for _, row in warnings.iterrows():
            print(f"   Patient {row['user_key']}: {row['pct_valid']:.1f}% valid - {row['exclusion_reason']}")
    
    if len(excluded) > 0:
        print(f"\nEXCLUDED: {len(excluded)} patients with no valid correlations")
        for _, row in excluded.iterrows():
            print(f"   Patient {row['user_key']}: {row['exclusion_reason']}")
    
    print("\n" + "="*80)
    
    # Create summary statistics (excluding patients with no valid correlations)
    summary_df = analyzer.create_summary_statistics(results_df, exclude_invalid=True)
    summary_path = output_dir / "per_patient_summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    write_sidecar(summary_path, config=config, data=analyzer.data_loader.data_provenance)
    logger.info(f"Saved summary statistics to {summary_path} ({len(summary_df)} patients)")
    
    # Also create a summary with ALL patients (including invalid) for comparison
    summary_all_df = analyzer.create_summary_statistics(results_df, exclude_invalid=False)
    summary_all_path = output_dir / "per_patient_summary_statistics_all.csv"
    summary_all_df.to_csv(summary_all_path, index=False)
    write_sidecar(summary_all_path, config=config, data=analyzer.data_loader.data_provenance)
    logger.info(f"Saved summary statistics (all patients) to {summary_all_path} ({len(summary_all_df)} patients)")
    
    # Create box plots (Fisher Z-transformed)
    analyzer.plot_per_patient_boxplots(
        results_df,
        correlation_type='spearman',
        save_dir=str(output_dir),
        use_fisher_z=True
    )
    
    analyzer.plot_per_patient_boxplots(
        results_df,
        correlation_type='pearson',
        save_dir=str(output_dir),
        use_fisher_z=True
    )
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    
    return results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = main()

