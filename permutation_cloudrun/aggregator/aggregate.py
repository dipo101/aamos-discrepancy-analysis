"""
Aggregator: Combine Cloud Run Results
Downloads all batch results from GCS and computes final p-values.
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import storage
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = '../config.yaml') -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_batch_results(config: dict, cache_file: str = 'batch_results_cache.json') -> list:
    """
    Download all batch results from GCS, with local caching.
    """
    # Check cache first
    cache_path = Path(cache_file)
    if cache_path.exists():
        logger.info(f" Found local cache: {cache_file}")
        try:
            with open(cache_path, 'r') as f:
                batch_results = json.load(f)
            logger.info(f" Loaded {len(batch_results)} batch results from cache")
            return batch_results
        except Exception as e:
            logger.warning(f"  Could not load cache: {e}. Redownloading...")

    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    logger.info(f" Downloading results from gs://{bucket_name}/results/")
    
    # List all result files
    blobs = list(bucket.list_blobs(prefix='results/'))
    
    logger.info(f"Found {len(blobs)} result files")
    
    batch_results = []
    for blob in tqdm(blobs, desc="Downloading"):
        if blob.name.endswith('.json'):
            try:
                content = blob.download_as_text()
                result = json.loads(content)
                batch_results.append(result)
            except Exception as e:
                logger.warning(f"  Failed to download/parse {blob.name}: {e}")
    
    logger.info(f" Downloaded {len(batch_results)} batch results")
    
    # Save to cache
    logger.info(f" Saving to cache: {cache_file}")
    with open(cache_path, 'w') as f:
        json.dump(batch_results, f)
    
    return batch_results


def load_observed_medians(config: dict) -> pd.DataFrame:
    """Load the observed median Z-scores for each patient."""
    # From the original per-patient analysis
    
    results_file = Path(config.get('observed_medians_file', 
                                    '../../results/per_patient_analysis/per_patient_summary_statistics.csv'))
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        logger.info(f" Loaded observed medians for {len(df)} patients")
        return df
    else:
        logger.warning(f"  Observed medians file not found: {results_file}")
        return None


def combine_permutation_results(batch_results: list) -> pd.DataFrame:
    """
    Combine all batch results into a single DataFrame.
    
    Returns:
        DataFrame with columns: patient_id, permutation_idx, and all summary stats
    """
    logger.info("🔨 Combining permutation results...")
    
    rows = []
    
    for batch in tqdm(batch_results, desc="Processing"):
        patient_id = batch['patient_id']
        
        for perm in batch['permutations']:
            # Get median (required)
            null_z = perm.get('median_fisher_z')
            perm_idx = perm.get('permutation_idx')
            
            # Check if None (JSON null) or NaN
            if null_z is None or (isinstance(null_z, float) and np.isnan(null_z)):
                continue
            
            row = {
                    'patient_id': patient_id,
                'permutation_idx': perm_idx,
                'n_valid_configs': perm.get('n_valid_configs'),
                # Basic stats
                'null_median_z': null_z,
                'null_mean_z': perm.get('mean_fisher_z'),
                'null_min_z': perm.get('min_fisher_z'),
                'null_max_z': perm.get('max_fisher_z'),
                'null_q25_z': perm.get('q25_fisher_z'),
                'null_q75_z': perm.get('q75_fisher_z'),
                'null_std_z': perm.get('std_fisher_z'),
                # Boxplot stats
                'null_iqr_z': perm.get('iqr_fisher_z'),
                'null_lower_fence': perm.get('lower_fence'),
                'null_upper_fence': perm.get('upper_fence'),
                'null_lower_whisker': perm.get('lower_whisker'),
                'null_upper_whisker': perm.get('upper_whisker'),
                'n_outliers': perm.get('n_outliers'),
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f" Combined {len(df)} permutation results across "
               f"{df['patient_id'].nunique()} patients")
    
    # Log available columns
    logger.info(f"   Available stats: {list(df.columns)}")
    
    return df


def compute_distribution_stats(values: np.ndarray, prefix: str) -> dict:
    """Compute distribution statistics for an array of values."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return {
            f'{prefix}_mean': None,
            f'{prefix}_std': None,
            f'{prefix}_min': None,
            f'{prefix}_max': None,
            f'{prefix}_q25': None,
            f'{prefix}_q75': None,
        }
    
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return {
            f'{prefix}_mean': None,
            f'{prefix}_std': None,
            f'{prefix}_min': None,
            f'{prefix}_max': None,
            f'{prefix}_q25': None,
            f'{prefix}_q75': None,
        }
    
    return {
        f'{prefix}_mean': float(np.mean(valid_values)),
        f'{prefix}_std': float(np.std(valid_values)),
        f'{prefix}_min': float(np.min(valid_values)),
        f'{prefix}_max': float(np.max(valid_values)),
        f'{prefix}_q25': float(np.percentile(valid_values, 25)),
        f'{prefix}_q75': float(np.percentile(valid_values, 75)),
    }


def compute_p_values(
    permutation_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    correlation_type: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute permutation p-values for each patient.
    
    Args:
        permutation_df: Combined permutation results
        observed_df: Observed median Z-scores per patient
        correlation_type: 'spearman' or 'pearson'
    
    Returns:
        DataFrame with p-values and significance flags
    """
    logger.info(" Computing p-values...")
    
    results = []
    
    z_col = f'median_{correlation_type}_z'
    n_patients = observed_df['user_key'].nunique()
    
    # Summary stats we want to compute distributions for
    null_stat_columns = [
        'null_median_z', 'null_mean_z', 'null_min_z', 'null_max_z',
        'null_q25_z', 'null_q75_z', 'null_std_z', 'null_iqr_z',
        'null_lower_whisker', 'null_upper_whisker'
    ]
    
    for patient_id in tqdm(observed_df['user_key'].unique(), desc="Computing p-values"):
        # Get observed median Z
        obs_row = observed_df[observed_df['user_key'] == patient_id]
        if len(obs_row) == 0:
            continue
        
        observed_z = obs_row[z_col].values[0]
        
        if np.isnan(observed_z):
            continue
        
        # Get patient's permutation data
        patient_perms = permutation_df[permutation_df['patient_id'] == patient_id]
        
        if len(patient_perms) == 0:
            continue
        
        # Get null distribution (median)
        null_median_dist = patient_perms['null_median_z'].values
        
        # Compute two-tailed p-value using median
        n_perms = len(null_median_dist)
        n_extreme = np.sum(np.abs(null_median_dist) >= np.abs(observed_z))
        p_value = (n_extreme + 1) / (n_perms + 1)  # +1 for continuity correction
        p_bonferroni = min(p_value * n_patients, 1.0)
        
        # Build result row
        row = {
            'patient_id': patient_id,
            'observed_median_z': observed_z,
            'n_permutations': n_perms,
            'avg_valid_configs': patient_perms['n_valid_configs'].mean() if 'n_valid_configs' in patient_perms.columns else None,
            # Primary significance (using median)
            'p_value': p_value,
            'p_bonferroni': p_bonferroni,
            'significant_uncorrected': p_value < 0.05,
            'significant_bonferroni': p_bonferroni < 0.05,
        }
        
        # Compute distribution stats for each null statistic
        for col in null_stat_columns:
            if col in patient_perms.columns:
                dist_stats = compute_distribution_stats(
                    patient_perms[col].values, 
                    prefix=f'dist_{col}'
                )
                row.update(dist_stats)
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    logger.info(f" Computed p-values for {len(results_df)} patients")
    
    # Summary
    n_sig_uncorrected = results_df['significant_uncorrected'].sum()
    n_sig_bonferroni = results_df['significant_bonferroni'].sum()
    
    logger.info(f"\n{'='*80}")
    logger.info("SIGNIFICANCE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total patients: {len(results_df)}")
    logger.info(f"Significant (p < 0.05, uncorrected): {n_sig_uncorrected}")
    logger.info(f"Significant (p < 0.05, Bonferroni): {n_sig_bonferroni}")
    logger.info(f"{'='*80}\n")
    
    return results_df


def create_visualizations(
    results_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    output_dir: Path
):
    """Create visualizations."""
    logger.info(" Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. P-value histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df['p_value'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Permutation P-values')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'p_value_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"   Saved: p_value_distribution.png")
    plt.close()
    
    # 2. Volcano plot (effect size vs p-value)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        results_df['observed_median_z'],
        -np.log10(results_df['p_value']),
        c=results_df['significant_bonferroni'],
        cmap='coolwarm',
        alpha=0.7,
        s=100
    )
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    ax.set_xlabel('Observed Median Fisher Z')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Volcano Plot: Effect Size vs Statistical Significance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'volcano_plot.png', dpi=300, bbox_inches='tight')
    logger.info(f"   Saved: volcano_plot.png")
    plt.close()
    
    # 3. Null distributions for significant patients
    sig_patients = results_df[
        results_df['significant_bonferroni']
    ]['patient_id'].values
    
    if len(sig_patients) > 0:
        n_sig = min(len(sig_patients), 6)  # Show up to 6
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, patient_id in enumerate(sig_patients[:n_sig]):
            # Get null distribution
            null_dist = permutation_df[
                permutation_df['patient_id'] == patient_id
            ]['null_median_z'].values
            
            # Get observed value
            obs = results_df[
                results_df['patient_id'] == patient_id
            ]['observed_median_z'].values[0]
            
            p_val = results_df[
                results_df['patient_id'] == patient_id
            ]['p_bonferroni'].values[0]
            
            # Plot
            axes[i].hist(null_dist, bins=30, alpha=0.7, edgecolor='black')
            axes[i].axvline(obs, color='red', linewidth=2, label='Observed')
            axes[i].set_xlabel('Median Fisher Z')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Patient {patient_id} (p = {p_val:.4f})')
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(n_sig, 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'null_distributions_significant.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"   Saved: null_distributions_significant.png")
        plt.close()
    
    logger.info(" All visualizations created")


def save_final_results(
    results_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    config: dict,
    output_dir: Path
):
    """Save final results to CSV and GCS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save locally
    results_file = output_dir / 'permutation_test_results.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f" Saved results to: {results_file}")
    
    # Save to GCS
    bucket_name = config['gcp']['bucket_name']
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Upload results CSV
    blob = bucket.blob(f"final/permutation_test_results_{timestamp}.csv")
    blob.upload_from_filename(results_file)
    logger.info(f" Uploaded to: gs://{bucket_name}/{blob.name}")
    
    # Upload full permutation data (Parquet for efficiency)
    permutation_file = output_dir / 'all_permutations.parquet'
    permutation_df.to_parquet(permutation_file, index=False)
    
    blob = bucket.blob(f"final/all_permutations_{timestamp}.parquet")
    blob.upload_from_filename(permutation_file)
    logger.info(f" Uploaded to: gs://{bucket_name}/{blob.name}")


def main():
    """Main aggregation function."""
    logger.info("="*80)
    logger.info("PERMUTATION ANALYSIS V3 - RESULT AGGREGATOR")
    logger.info("="*80 + "\n")
    
    # Load config
    config = load_config()
    correlation_type = config['analysis']['correlation_type']
    
    # Create output directory
    output_dir = Path(config['output']['local_results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download batch results from GCS
    batch_results = download_batch_results(config)
    
    # Load observed medians
    observed_df = load_observed_medians(config)
    
    if observed_df is None:
        logger.error(" Cannot proceed without observed medians")
        return
    
    # Combine permutation results
    permutation_df = combine_permutation_results(batch_results)
    
    # Compute p-values
    results_df = compute_p_values(
        permutation_df,
        observed_df,
        correlation_type
    )
    
    # Create visualizations
    if config['output'].get('visualizations', True):
        create_visualizations(results_df, permutation_df, output_dir)
    
    # Save final results
    save_final_results(results_df, permutation_df, config, output_dir)
    
    logger.info("\n AGGREGATION COMPLETE!")
    logger.info(f"   Results saved to: {output_dir}")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string())
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

