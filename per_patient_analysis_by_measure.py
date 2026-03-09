#!/usr/bin/env python3
"""
Per-Patient Analysis by Summary Measure

Generates per-patient analysis outputs for each summary statistic (median, mean, etc.)
including box plots with significance coloring based on each measure's permutation test results.

Outputs are saved to: results/per_patient_analysis/{measure}/...
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MEASURE_CONFIG = {
    'median': {
        'observed_col': 'median_spearman_z',
        'display_name': 'Median',
    },
    'mean': {
        'observed_col': 'mean_spearman_z',
        'display_name': 'Mean',
    },
    'min': {
        'observed_col': 'min_spearman_z',
        'display_name': 'Minimum',
    },
    'max': {
        'observed_col': 'max_spearman_z',
        'display_name': 'Maximum',
    },
    'q25': {
        'observed_col': 'q25_spearman_z',
        'display_name': '25th Percentile',
    },
    'q75': {
        'observed_col': 'q75_spearman_z',
        'display_name': '75th Percentile',
    },
}

MEASURES = ['median', 'mean', 'min', 'max', 'q25', 'q75']

# Paths
DETAILED_RESULTS_PATH = Path("results/per_patient_analysis/per_patient_correlation_results.csv")
SUMMARY_STATS_PATH = Path("results/per_patient_analysis/per_patient_summary_statistics.csv")
SENSITIVITY_RESULTS_DIR = Path("results/permutation_by_measure")
OUTPUT_BASE_DIR = Path("results/per_patient_analysis")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_detailed_results() -> pd.DataFrame:
    """Load the per-configuration correlation results."""
    df = pd.read_csv(DETAILED_RESULTS_PATH)
    df['user_key'] = df['user_key'].astype(str)
    return df


def load_significance_results(measure: str) -> Optional[pd.DataFrame]:
    """Load the significance results for a specific measure."""
    sig_path = SENSITIVITY_RESULTS_DIR / measure / "permutation_test_results.csv"
    if not sig_path.exists():
        logger.warning(f"Significance results not found for {measure}: {sig_path}")
        return None
    df = pd.read_csv(sig_path)
    df['patient_id'] = df['patient_id'].astype(str)
    return df


def compute_quartiles(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Q25 and Q75 for each patient."""
    stats = detailed_df.groupby('user_key')['spearman_z'].agg(
        q25_spearman_z=lambda x: x.quantile(0.25),
        q75_spearman_z=lambda x: x.quantile(0.75)
    ).reset_index()
    return stats


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_per_patient_boxplots_with_significance(
    detailed_df: pd.DataFrame,
    sig_df: Optional[pd.DataFrame],
    measure: str,
    output_dir: Path,
    correlation_type: str = 'spearman',
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Create box plots showing distribution of correlations for each patient,
    with significance coloring based on the specified measure.
    """
    config = MEASURE_CONFIG[measure]
    logger.info(f"Creating {correlation_type} box plots for {config['display_name']}...")
    
    corr_col = f'{correlation_type}_z'
    
    # Get unique patients
    all_patients = sorted(detailed_df['user_key'].unique())
    
    # Prepare data
    patient_data = []
    patients = []
    patient_colors = []
    skipped_patients = []
    
    for patient in all_patients:
        patient_results = detailed_df[detailed_df['user_key'] == patient][corr_col]
        valid_results = patient_results[np.isfinite(patient_results)]
        
        if len(valid_results) == 0:
            skipped_patients.append(patient)
            continue
        
        patient_data.append(valid_results)
        patients.append(patient)
        
        # Determine color based on significance
        if sig_df is not None and patient in sig_df['patient_id'].values:
            patient_sig = sig_df[sig_df['patient_id'] == patient].iloc[0]
            is_sig_bonf = patient_sig.get('significant_bonferroni', False)
            is_sig_nom = patient_sig.get('significant_uncorrected', False)
            median_z = valid_results.median()
            
            if is_sig_bonf:
                color = '#2E8B57' if median_z >= 0 else '#FF6B6B'  # Teal/Red
            elif is_sig_nom:
                color = '#DAA520'  # Gold
            else:
                color = '#808080'  # Grey
        else:
            color = '#808080'  # Grey (no significance data)
        
        patient_colors.append(color)
    
    if len(patients) == 0:
        logger.error(f"No patients with valid {correlation_type} values!")
        return
    
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
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )
    
    # Apply colors
    for patch, color in zip(bp['boxes'], patient_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add optimal threshold line
    ax.axhline(y=0.5, color='#FF6B00', linestyle='--', alpha=0.7, linewidth=1.5,
               label='Concordance Threshold (Z = 0.5)')
    
    # Labels
    ax.set_xlabel('Patient ID', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{correlation_type.title()} Fisher Z', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Distribution of {correlation_type.title()} Correlations Across Configurations\n'
        f'(Significance by {config["display_name"]})',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2E8B57', alpha=0.7, label='Bonferroni Significant (Concordance)'),
        mpatches.Patch(color='#FF6B6B', alpha=0.7, label='Bonferroni Significant (Discordance)'),
        mpatches.Patch(color='#DAA520', alpha=0.7, label='Nominally Significant (p < 0.05)'),
        mpatches.Patch(color='#808080', alpha=0.7, label='Non-Significant'),
        plt.Line2D([0], [0], color='#FF6B00', linestyle='--', linewidth=1.5, label='Concordance (Z >= 0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Summary text
    if sig_df is not None:
        n_sig_bonf = sig_df['significant_bonferroni'].sum()
        n_sig_nom = sig_df['significant_uncorrected'].sum()
        textstr = f"Bonferroni significant: {n_sig_bonf}\nNominally significant: {n_sig_nom}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"per_patient_{correlation_type}_fisher_z_boxplots.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {save_path}")
    plt.close()


def plot_overall_distribution(
    detailed_df: pd.DataFrame,
    sig_df: Optional[pd.DataFrame],
    measure: str,
    output_dir: Path
):
    """
    Create overall distribution plot with density curves per significance group.
    """
    config = MEASURE_CONFIG[measure]
    logger.info(f"Creating overall distribution for {config['display_name']}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, corr_type in enumerate(['spearman', 'pearson']):
        ax = axes[idx]
        corr_col = f'{corr_type}_z'
        
        # Filter valid values
        valid_df = detailed_df[np.isfinite(detailed_df[corr_col])].copy()
        
        if sig_df is not None:
            # Add significance info
            valid_df = valid_df.merge(
                sig_df[['patient_id', 'significant_bonferroni', 'significant_uncorrected']],
                left_on='user_key', right_on='patient_id', how='left'
            )
            valid_df['significant_bonferroni'] = valid_df['significant_bonferroni'].fillna(False)
            valid_df['significant_uncorrected'] = valid_df['significant_uncorrected'].fillna(False)
            
            # Plot by group
            sig_data = valid_df[valid_df['significant_bonferroni']][corr_col]
            nom_data = valid_df[valid_df['significant_uncorrected'] & ~valid_df['significant_bonferroni']][corr_col]
            non_sig_data = valid_df[~valid_df['significant_uncorrected']][corr_col]
            
            if len(sig_data) > 0:
                ax.hist(sig_data, bins=30, alpha=0.5, color='#2E8B57', label=f'Bonferroni Sig. (n={len(sig_data)})', density=True)
            if len(nom_data) > 0:
                ax.hist(nom_data, bins=30, alpha=0.5, color='#DAA520', label=f'Nominal Sig. (n={len(nom_data)})', density=True)
            if len(non_sig_data) > 0:
                ax.hist(non_sig_data, bins=30, alpha=0.5, color='#808080', label=f'Non-Sig. (n={len(non_sig_data)})', density=True)
        else:
            ax.hist(valid_df[corr_col], bins=30, alpha=0.7, color='steelblue', density=True)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='#FF6B00', linestyle='--', alpha=0.7, label='Concordance (Z=0.5)')
        ax.set_xlabel(f'{corr_type.title()} Fisher Z', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{corr_type.title()} Correlations', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Overall Distribution of Correlations\n(Significance by {config["display_name"]})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "overall_correlation_distributions_fisher_z.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {save_path}")
    plt.close()


def plot_parameter_sensitivity(
    detailed_df: pd.DataFrame,
    sig_df: Optional[pd.DataFrame],
    measure: str,
    output_dir: Path
):
    """
    Create parameter sensitivity analysis plots.
    """
    config = MEASURE_CONFIG[measure]
    logger.info(f"Creating parameter sensitivity for {config['display_name']}...")
    
    # Parameters to analyze
    params = [
        ('timestamp_window', 'Time Window (hours)'),
        ('use_calendar_days', 'Calendar Days'),
        ('filter_out_zero_usage_entries', 'Filter Zero Usage'),
        ('categorization_method', 'Categorization Method')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    valid_df = detailed_df[np.isfinite(detailed_df['spearman_z'])].copy()
    
    for idx, (param, label) in enumerate(params):
        ax = axes[idx]
        
        # Group by parameter
        grouped = valid_df.groupby(param)['spearman_z'].agg(['mean', 'std', 'count']).reset_index()
        
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=3, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped[param], rotation=45, ha='right')
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Mean Spearman Fisher Z', fontsize=11)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Parameter Sensitivity Analysis\n(Fisher Z-transformed Spearman)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "parameter_sensitivity_analysis_fisher_z.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {save_path}")
    plt.close()


def create_summary_statistics_by_measure(
    detailed_df: pd.DataFrame,
    sig_df: Optional[pd.DataFrame],
    measure: str,
    output_dir: Path
):
    """
    Create summary statistics CSV that includes significance info for this measure.
    """
    config = MEASURE_CONFIG[measure]
    logger.info(f"Creating summary statistics for {config['display_name']}...")
    
    summary_rows = []
    
    for patient in detailed_df['user_key'].unique():
        patient_results = detailed_df[detailed_df['user_key'] == patient]
        
        # Get valid values
        valid_spearman_z = patient_results['spearman_z'][np.isfinite(patient_results['spearman_z'])]
        valid_pearson_z = patient_results['pearson_z'][np.isfinite(patient_results['pearson_z'])]
        
        if len(valid_spearman_z) == 0:
            continue
        
        row = {
            'user_key': patient,
            'n_configurations': len(patient_results),
            'n_valid': len(valid_spearman_z),
            'median_spearman_z': valid_spearman_z.median(),
            'mean_spearman_z': valid_spearman_z.mean(),
            'std_spearman_z': valid_spearman_z.std(),
            'min_spearman_z': valid_spearman_z.min(),
            'max_spearman_z': valid_spearman_z.max(),
            'q25_spearman_z': valid_spearman_z.quantile(0.25),
            'q75_spearman_z': valid_spearman_z.quantile(0.75),
            'median_pearson_z': valid_pearson_z.median() if len(valid_pearson_z) > 0 else np.nan,
            'mean_pearson_z': valid_pearson_z.mean() if len(valid_pearson_z) > 0 else np.nan,
        }
        
        # Add significance info if available
        if sig_df is not None and patient in sig_df['patient_id'].values:
            patient_sig = sig_df[sig_df['patient_id'] == patient].iloc[0]
            row['p_value'] = patient_sig.get('p_value', np.nan)
            row['p_bonferroni'] = patient_sig.get('p_bonferroni', np.nan)
            row['significant_bonferroni'] = patient_sig.get('significant_bonferroni', False)
            row['significant_uncorrected'] = patient_sig.get('significant_uncorrected', False)
        else:
            row['p_value'] = np.nan
            row['p_bonferroni'] = np.nan
            row['significant_bonferroni'] = False
            row['significant_uncorrected'] = False
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by the measure's statistic
    obs_col = config['observed_col']
    if obs_col in summary_df.columns:
        summary_df = summary_df.sort_values(obs_col, ascending=False)
    
    save_path = output_dir / "per_patient_summary_statistics.csv"
    summary_df.to_csv(save_path, index=False)
    logger.info(f"Saved: {save_path}")
    
    return summary_df


def create_significance_summary(
    sig_df: Optional[pd.DataFrame],
    measure: str,
    output_dir: Path
):
    """
    Create a text summary of significance results.
    """
    config = MEASURE_CONFIG[measure]
    
    summary_path = output_dir / "significance_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"Significance Summary: {config['display_name']}\n")
        f.write("=" * 50 + "\n\n")
        
        if sig_df is None:
            f.write("No significance results available.\n")
        else:
            n_total = len(sig_df)
            n_sig_bonf = sig_df['significant_bonferroni'].sum()
            n_sig_nom = sig_df['significant_uncorrected'].sum()
            
            f.write(f"Total patients: {n_total}\n")
            f.write(f"Bonferroni significant: {n_sig_bonf}\n")
            f.write(f"Nominally significant: {n_sig_nom}\n\n")
            
            if n_sig_bonf > 0:
                f.write("Bonferroni Significant Patients:\n")
                sig_patients = sig_df[sig_df['significant_bonferroni']]
                for _, row in sig_patients.iterrows():
                    obs_col = f"observed_{measure}_z"
                    obs_val = row.get(obs_col, row.get('observed_median_z', np.nan))
                    f.write(f"  Patient {row['patient_id']}: Z = {obs_val:.3f}, p = {row['p_value']:.4f}\n")
    
    logger.info(f"Saved: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_analysis_for_measure(measure: str, detailed_df: pd.DataFrame):
    """Run all per-patient analysis outputs for a single measure."""
    config = MEASURE_CONFIG[measure]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATING PER-PATIENT ANALYSIS: {config['display_name']}")
    logger.info(f"{'='*60}")
    
    # Create output directory
    output_dir = OUTPUT_BASE_DIR / measure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load significance results
    sig_df = load_significance_results(measure)
    
    # Generate outputs
    plot_per_patient_boxplots_with_significance(
        detailed_df, sig_df, measure, output_dir, 'spearman'
    )
    plot_per_patient_boxplots_with_significance(
        detailed_df, sig_df, measure, output_dir, 'pearson'
    )
    plot_overall_distribution(detailed_df, sig_df, measure, output_dir)
    plot_parameter_sensitivity(detailed_df, sig_df, measure, output_dir)
    create_summary_statistics_by_measure(detailed_df, sig_df, measure, output_dir)
    create_significance_summary(sig_df, measure, output_dir)
    
    logger.info(f"✅ Completed analysis for {config['display_name']}")


def main():
    logger.info("="*60)
    logger.info("PER-PATIENT ANALYSIS BY SUMMARY MEASURE")
    logger.info("="*60)
    
    # Load detailed results
    logger.info("Loading detailed correlation results...")
    if not DETAILED_RESULTS_PATH.exists():
        logger.error(f"Detailed results not found: {DETAILED_RESULTS_PATH}")
        sys.exit(1)
    
    detailed_df = load_detailed_results()
    logger.info(f"Loaded {len(detailed_df)} rows for {detailed_df['user_key'].nunique()} patients")
    
    # Run analysis for each measure
    for measure in MEASURES:
        try:
            run_analysis_for_measure(measure, detailed_df)
        except Exception as e:
            logger.error(f"Failed for {measure}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("ALL PER-PATIENT ANALYSES COMPLETE")
    logger.info(f"Results saved to: {OUTPUT_BASE_DIR.absolute()}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

