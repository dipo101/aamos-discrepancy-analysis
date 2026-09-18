"""
Sensitivity Analysis Core Module

Provides generic, reusable functions for running concordance analysis
across different summary statistics (median, mean, q25, q75, min, max).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Maps measure names to column names in various data sources
MEASURE_CONFIG = {
    'median': {
        'observed_col': 'median_spearman_z',
        'null_col': 'null_median_z',
        'display_name': 'Median',
        'description': 'Median Fisher Z across multiverse configurations',
    },
    'mean': {
        'observed_col': 'mean_spearman_z',
        'null_col': 'null_mean_z',
        'display_name': 'Mean',
        'description': 'Mean Fisher Z across multiverse configurations',
    },
    'min': {
        'observed_col': 'min_spearman_z',
        'null_col': 'null_min_z',
        'display_name': 'Minimum',
        'description': 'Minimum Fisher Z across multiverse configurations',
    },
    'max': {
        'observed_col': 'max_spearman_z',
        'null_col': 'null_max_z',
        'display_name': 'Maximum',
        'description': 'Maximum Fisher Z across multiverse configurations',
    },
    'q25': {
        'observed_col': 'q25_spearman_z',  # May need to compute
        'null_col': 'null_q25_z',
        'display_name': '25th Percentile',
        'description': '25th percentile Fisher Z across multiverse configurations',
    },
    'q75': {
        'observed_col': 'q75_spearman_z',  # May need to compute
        'null_col': 'null_q75_z',
        'display_name': '75th Percentile',
        'description': '75th percentile Fisher Z across multiverse configurations',
    },
}

# Measures for sensitivity analysis
RECOMMENDED_MEASURES = ['median', 'mean', 'q25', 'q75']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_observed_statistics(
    summary_file: Path,
    detailed_file: Optional[Path] = None,
    measure: str = 'median'
) -> pd.DataFrame:
    """
    Load observed statistics for a given measure.
    
    For q25/q75, computes from detailed results if not in summary file.
    
    Args:
        summary_file: Path to per_patient_summary_statistics.csv
        detailed_file: Path to per_patient_correlation_results.csv (for q25/q75)
        measure: Statistical measure ('median', 'mean', 'min', 'max', 'q25', 'q75')
    
    Returns:
        DataFrame with columns: user_key, observed_{measure}_z
    """
    config = MEASURE_CONFIG.get(measure)
    if not config:
        raise ValueError(f"Unknown measure: {measure}. Choose from {list(MEASURE_CONFIG.keys())}")
    
    observed_col = config['observed_col']
    
    # Load summary statistics
    summary_df = pd.read_csv(summary_file)
    summary_df['user_key'] = summary_df['user_key'].astype(str)
    
    # Check if column exists
    if observed_col in summary_df.columns:
        result = summary_df[['user_key', observed_col]].copy()
        result = result.rename(columns={observed_col: f'observed_{measure}_z'})
        return result
    
    # For q25/q75, compute from detailed results
    if measure in ['q25', 'q75'] and detailed_file and detailed_file.exists():
        logger.info(f"Computing {measure} from detailed results...")
        detailed_df = pd.read_csv(detailed_file)
        detailed_df['user_key'] = detailed_df['user_key'].astype(str)
        
        if measure == 'q25':
            stats = detailed_df.groupby('user_key')['spearman_z'].quantile(0.25).reset_index()
        else:  # q75
            stats = detailed_df.groupby('user_key')['spearman_z'].quantile(0.75).reset_index()
        
        stats.columns = ['user_key', f'observed_{measure}_z']
        return stats
    
    raise ValueError(f"Could not find or compute observed values for measure '{measure}'")


def load_null_distributions(
    permutation_parquet: Path,
    measure: str = 'median'
) -> pd.DataFrame:
    """
    Load null distribution data for a given measure.
    
    Args:
        permutation_parquet: Path to all_permutations.parquet
        measure: Statistical measure
    
    Returns:
        DataFrame with columns: patient_id, permutation_idx, null_{measure}_z
    """
    config = MEASURE_CONFIG.get(measure)
    if not config:
        raise ValueError(f"Unknown measure: {measure}")
    
    null_col = config['null_col']
    
    # Load permutation data
    perm_df = pd.read_parquet(permutation_parquet)
    perm_df['patient_id'] = perm_df['patient_id'].astype(str)
    
    if null_col not in perm_df.columns:
        raise ValueError(f"Null column '{null_col}' not found in permutation data. "
                        f"Available: {list(perm_df.columns)}")
    
    # Select relevant columns
    cols = ['patient_id', 'permutation_idx', null_col]
    if 'n_valid_configs' in perm_df.columns:
        cols.append('n_valid_configs')
    
    result = perm_df[cols].copy()
    result = result.rename(columns={null_col: f'null_{measure}_z'})
    
    return result


# =============================================================================
# P-VALUE COMPUTATION
# =============================================================================

def compute_p_values_for_measure(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    measure: str = 'median'
) -> pd.DataFrame:
    """
    Compute permutation p-values for a specific measure.
    
    Args:
        observed_df: DataFrame with observed values (user_key, observed_{measure}_z)
        null_df: DataFrame with null distributions (patient_id, null_{measure}_z)
        measure: Statistical measure name
    
    Returns:
        DataFrame with p-values and significance flags
    """
    config = MEASURE_CONFIG[measure]
    observed_col = f'observed_{measure}_z'
    null_col = f'null_{measure}_z'
    
    n_patients = observed_df['user_key'].nunique()
    results = []
    
    for patient_id in observed_df['user_key'].unique():
        # Get observed value
        obs_row = observed_df[observed_df['user_key'] == patient_id]
        if len(obs_row) == 0:
            continue
        
        observed_z = obs_row[observed_col].values[0]
        if pd.isna(observed_z):
            continue
        
        # Get null distribution
        patient_nulls = null_df[null_df['patient_id'] == patient_id][null_col].dropna()
        if len(patient_nulls) == 0:
            continue
        
        null_values = patient_nulls.values
        n_perms = len(null_values)
        
        # Two-tailed p-value about the null's own centre (see aamos_concordance.summary):
        # the pipeline on permuted data is not centred on zero, so |null| >= |observed|
        # (the rule used for the published v1 tables) answers a different question
        # from the null-band figure. results/v1 tables were produced with the old rule.
        centre = np.mean(null_values)
        n_extreme = np.sum(np.abs(null_values - centre) >= np.abs(observed_z - centre))
        p_value = (n_extreme + 1) / (n_perms + 1)  # Continuity correction
        p_bonferroni = min(p_value * n_patients, 1.0)
        
        # Null distribution statistics
        null_mean = np.mean(null_values)
        null_std = np.std(null_values)
        null_q025 = np.percentile(null_values, 2.5)
        null_q975 = np.percentile(null_values, 97.5)
        
        # Get avg valid configs if available
        avg_configs = None
        if 'n_valid_configs' in null_df.columns:
            patient_configs = null_df[null_df['patient_id'] == patient_id]['n_valid_configs']
            avg_configs = patient_configs.mean() if len(patient_configs) > 0 else None
        
        results.append({
            'patient_id': patient_id,
            f'observed_{measure}_z': observed_z,
            'n_permutations': n_perms,
            'avg_valid_configs': avg_configs,
            'p_value': p_value,
            'p_bonferroni': p_bonferroni,
            'significant_uncorrected': p_value < 0.05,
            'significant_bonferroni': p_bonferroni < 0.05,
            f'null_{measure}_mean': null_mean,
            f'null_{measure}_std': null_std,
            f'null_{measure}_q025': null_q025,
            f'null_{measure}_q975': null_q975,
        })
    
    results_df = pd.DataFrame(results)
    
    # Log summary
    if len(results_df) > 0:
        n_sig_uncorrected = results_df['significant_uncorrected'].sum()
        n_sig_bonferroni = results_df['significant_bonferroni'].sum()
        logger.info(f"\n{config['display_name']} Analysis Summary:")
        logger.info(f"  Total patients: {len(results_df)}")
        logger.info(f"  Significant (nominal): {n_sig_uncorrected}")
        logger.info(f"  Significant (Bonferroni): {n_sig_bonferroni}")
    
    return results_df


# =============================================================================
# SIGNIFICANCE HELPERS
# =============================================================================

def get_significant_patients(
    results_df: pd.DataFrame,
    bonferroni: bool = True
) -> List[str]:
    """Get list of significant patient IDs."""
    col = 'significant_bonferroni' if bonferroni else 'significant_uncorrected'
    return results_df[results_df[col]]['patient_id'].astype(str).tolist()


def get_concordant_patients(
    results_df: pd.DataFrame,
    measure: str,
    z_threshold: float = 0.45,  # Rounds to 0.5 at 1dp
    require_significance: bool = True
) -> List[str]:
    """
    Get list of concordant patient IDs.

    Concordant = observed Z >= threshold AND (optionally) significant.
    """
    obs_col = f'observed_{measure}_z'
    
    mask = results_df[obs_col] >= z_threshold
    if require_significance:
        mask = mask & results_df['significant_bonferroni']
    
    return results_df[mask]['patient_id'].astype(str).tolist()


# =============================================================================
# DISTRIBUTION STATISTICS
# =============================================================================

def compute_distribution_stats(values: np.ndarray, prefix: str) -> Dict:
    """Compute summary statistics for an array."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_max': np.nan,
            f'{prefix}_q25': np.nan,
            f'{prefix}_q75': np.nan,
        }
    
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return compute_distribution_stats(np.array([]), prefix)
    
    return {
        f'{prefix}_mean': float(np.mean(valid)),
        f'{prefix}_std': float(np.std(valid)),
        f'{prefix}_min': float(np.min(valid)),
        f'{prefix}_max': float(np.max(valid)),
        f'{prefix}_q25': float(np.percentile(valid, 25)),
        f'{prefix}_q75': float(np.percentile(valid, 75)),
    }

