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
try:
    from google.cloud import storage
except ImportError:  # local mode does not need GCS
    storage = None
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from tqdm import tqdm

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for aamos_concordance
from aamos_concordance import write_sidecar
from aamos_concordance.provenance import git_state
from aamos_concordance.summary import build_world_summary, derive_concordant_sets, null_sources_for, observed_table, threshold_sweep, to_v1_measure_table
from aamos_concordance.worlds import (
    BASELINE, REPO_ROOT, NULL_PARQUET, NULL_PER_CONFIG_PARQUET, OBSERVED_CSV, PER_CONFIG_Z, SUMMARY_CSV, THRESHOLD_SWEEP_CSV, as_world, gcs_results_prefix,
    register_world, world_dir, write_concordant_sets, write_world_config,
)
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration file (permutation_cloudrun/config.yaml by default)."""
    path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / 'config.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def download_batch_results(config: dict, cache_file: str = 'batch_results_cache.json', prefix: str = 'results/') -> list:
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
    
    logger.info(f" Downloading results from gs://{bucket_name}/{prefix}")
    
    # List all result files
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name[len(prefix):].count('/') == 0]
    
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


def download_per_config_null(config: dict, prefix: str):
    """Download and concatenate <prefix>perconfig/*.parquet (per-configuration null Z). None if absent."""
    bucket_name = config['gcp']['bucket_name']
    bucket = storage.Client().bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=f'{prefix}perconfig/') if b.name.endswith('.parquet')]
    if not blobs:
        logger.warning(f" No per-config null parquets under gs://{bucket_name}/{prefix}perconfig/ "
                       "(batches produced by a worker predating per-config storage)")
        return None
    logger.info(f" Downloading {len(blobs)} per-config parquets from gs://{bucket_name}/{prefix}perconfig/")
    parts = [pd.read_parquet(io.BytesIO(b.download_as_bytes())) for b in tqdm(blobs, desc="per-config")]
    df = pd.concat(parts, ignore_index=True).sort_values(['patient_id', 'permutation_idx', 'config_idx']).reset_index(drop=True)
    logger.info(f" Per-config null: {len(df)} rows, {df['patient_id'].nunique()} patients")
    return df


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




def save_world_outputs(world, permutation_df: pd.DataFrame, config: dict, *, upload: bool = True,
                       per_config_null=None) -> pd.DataFrame:
    """Write null.parquet (+ null_per_config.parquet) and summary.csv into the world directory,
    derive the concordant sets, register the world."""
    wdir = world_dir(world, create=True)

    null_path = wdir / NULL_PARQUET
    permutation_df.to_parquet(null_path, index=False)
    write_sidecar(null_path, config=config, extra={'script': 'aggregate.py', 'world': str(world), 'n_rows': int(len(permutation_df))})
    logger.info(f" Saved null distribution to: {null_path}")

    if per_config_null is not None:
        pc_path = wdir / NULL_PER_CONFIG_PARQUET
        per_config_null.to_parquet(pc_path, index=False)
        write_sidecar(pc_path, config=config, extra={'script': 'aggregate.py', 'world': str(world), 'n_rows': int(len(per_config_null))})
        logger.info(f" Saved per-config null to: {pc_path}")

    per_config_path = wdir / PER_CONFIG_Z
    if not per_config_path.exists():
        raise FileNotFoundError(
            f"{per_config_path} not found. Run per_patient_correlation_analysis.py --world {world} first; "
            "the summary needs the observed per-config Z table.")
    per_config = pd.read_csv(per_config_path)
    summary = build_world_summary(per_config, null_sources=null_sources_for(legacy_null=permutation_df, per_config_null=per_config_null))
    summary_path = wdir / SUMMARY_CSV
    summary.to_csv(summary_path, index=False)
    observed_path = wdir / OBSERVED_CSV
    observed_table(per_config).to_csv(observed_path, index=False)
    write_sidecar(observed_path, config=config, extra={'script': 'aggregate.py', 'world': str(world)})
    sweep_path = wdir / THRESHOLD_SWEEP_CSV
    threshold_sweep(summary).to_csv(sweep_path, index=False)
    write_sidecar(sweep_path, config=config, extra={'script': 'aggregate.py', 'world': str(world)})
    write_sidecar(summary_path, config=config, extra={'script': 'aggregate.py', 'world': str(world),
                                                       'inputs': {'per_config_z': str(per_config_path), 'null': str(null_path)}})
    logger.info(f" Saved summary to: {summary_path}")

    sets = derive_concordant_sets(summary)
    write_concordant_sets(world, sets)
    write_world_config(world, {'null_source': str(null_path.relative_to(REPO_ROOT)), 'built_by': 'aggregate.py'})
    register_world(world, git_commit=git_state()['git_commit'], note='aggregate.py')
    logger.info(f" Concordant sets: {sets}")

    if upload:
        bucket_name = config['gcp']['bucket_name']
        bucket = storage.Client().bucket(bucket_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uploads = [summary_path, null_path] + ([wdir / NULL_PER_CONFIG_PARQUET] if per_config_null is not None else [])
        for local in uploads:
            blob = bucket.blob(f"final/{world}/{local.stem}_{timestamp}{local.suffix}")
            blob.upload_from_filename(local)
            logger.info(f" Uploaded to: gs://{bucket_name}/{blob.name}")
    return summary


def main(argv=None):
    """Aggregate one world's permutation batches into null.parquet + summary.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', default=str(BASELINE), help='world id, e.g. span=Q__case=A (default: baseline)')
    parser.add_argument('--no-visualizations', action='store_true')
    parser.add_argument('--no-upload', action='store_true', help='do not copy outputs back to GCS final/')
    parser.add_argument('--local', action='store_true',
                        help='summarise from null.parquet / null_per_config.parquet already in the world folder '
                             '(e.g. written by scripts/run_null_local.py) instead of downloading batches; implies --no-upload')
    args = parser.parse_args(argv)
    world = as_world(args.world)

    logger.info("="*80)
    logger.info(f"PERMUTATION RESULT AGGREGATOR - world {world}")
    logger.info("="*80 + "\n")

    config = load_config()
    if args.local:
        wdir = world_dir(world)
        null_path, pc_path = wdir / NULL_PARQUET, wdir / NULL_PER_CONFIG_PARQUET
        if not null_path.exists():
            raise SystemExit(f"--local: {null_path} not found; run scripts/run_null_local.py --world {world} first")
        permutation_df = pd.read_parquet(null_path)
        per_config_null = pd.read_parquet(pc_path) if pc_path.exists() else None
        config = {**config, 'source': 'local', 'null_path': str(null_path)}
        summary = save_world_outputs(world, permutation_df, config, upload=False, per_config_null=per_config_null)
    else:
        prefix = gcs_results_prefix(world)
        batch_results = download_batch_results(config, cache_file=f'batch_results_cache_{world}.json', prefix=prefix)
        permutation_df = combine_permutation_results(batch_results)
        per_config_null = download_per_config_null(config, prefix)
        summary = save_world_outputs(world, permutation_df, config, upload=not args.no_upload, per_config_null=per_config_null)

    if not args.no_visualizations and config['output'].get('visualizations', True):
        create_visualizations(to_v1_measure_table(summary, 'median'), permutation_df, world_dir(world))

    logger.info("\n AGGREGATION COMPLETE!")
    logger.info(f"   Results saved to: {world_dir(world)}")
    print("\n" + "="*80)
    print(f"SUMMARY - world {world}")
    print("="*80)
    print(summary.to_string())
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
