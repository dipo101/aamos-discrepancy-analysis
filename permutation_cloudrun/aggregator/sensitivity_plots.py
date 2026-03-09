"""
Sensitivity Analysis Plotting Module

Generic plotting functions parameterized by statistical measure.
These functions are adapted from create_advanced_plots.py and create_refined_plots.py
but made generic to support different measures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging

from sensitivity_core import MEASURE_CONFIG, get_significant_patients, get_concordant_patients

logger = logging.getLogger(__name__)

# Concordance threshold for Fisher's Z (rounds to 0.5 at 1dp)
CONCORDANCE_Z_THRESHOLD = 0.45


# =============================================================================
# DATA LOADING (for sensitivity analysis)
# =============================================================================

def load_detailed_results(detailed_path: Path) -> pd.DataFrame:
    """Load the per-configuration correlation results."""
    df = pd.read_csv(detailed_path)
    df['user_key'] = df['user_key'].astype(str)
    return df


def load_significance_results(sig_path: Path) -> pd.DataFrame:
    """Load the p-value/significance results."""
    df = pd.read_csv(sig_path)
    df['patient_id'] = df['patient_id'].astype(str)
    return df


def load_null_parquet(null_path: Path) -> Optional[pd.DataFrame]:
    """Load the full null distribution data."""
    if not null_path.exists():
        logger.warning(f"Null distribution file not found: {null_path}")
        return None
    df = pd.read_parquet(null_path)
    df['patient_id'] = df['patient_id'].astype(str)
    return df


# =============================================================================
# BOXPLOT WITH NULL BANDS (from create_advanced_plots.py)
# =============================================================================

def plot_distribution_boxplot(
    detailed_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    null_df: Optional[pd.DataFrame],
    output_dir: Path,
    measure: str = 'median'
):
    """
    Create box plot of correlations per patient with significance coloring
    and patient-specific null bands.
    """
    config = MEASURE_CONFIG[measure]
    obs_col = f'observed_{measure}_z'
    null_col = f'null_{measure}_z'
    
    logger.info(f"Generating Box Plot for {config['display_name']}...")
    
    # Prepare data
    plot_df = detailed_df.merge(
        sig_df[['patient_id', 'significant_bonferroni', 'significant_uncorrected']], 
        left_on='user_key', 
        right_on='patient_id', 
        how='inner'
    )
    
    if len(plot_df) == 0:
        logger.error("No common patients found!")
        return
    
    # Calculate median Z per patient for sorting (always use median for ordering)
    patient_medians = plot_df.groupby('user_key')['spearman_z'].median().sort_values()
    sorted_patients = patient_medians.index.tolist()
    
    # Define colors based on significance
    palette = {}
    for pid in sorted_patients:
        patient_rows = plot_df[plot_df['user_key'] == pid]
        if len(patient_rows) == 0:
            continue
            
        is_sig_strict = patient_rows['significant_bonferroni'].iloc[0]
        is_sig_nominal = patient_rows['significant_uncorrected'].iloc[0]
        median_z = patient_medians[pid]
        
        if is_sig_strict:
            palette[pid] = '#2E8B57' if median_z >= 0 else '#FF6B6B'
        elif is_sig_nominal:
            palette[pid] = '#DAA520'
        else:
            palette[pid] = '#808080'
    
    # Setup plot
    plt.figure(figsize=(16, 8))
    
    # Draw patient-specific null intervals
    for i, pid in enumerate(sorted_patients):
        patient_sig = sig_df[sig_df['patient_id'] == pid]
        if len(patient_sig) == 0:
            continue
        patient_sig = patient_sig.iloc[0]
        
        # Get 95% CI from null distribution
        low, high = None, None
        
        if null_df is not None and not null_df.empty:
            patient_nulls = null_df[null_df['patient_id'] == pid][null_col].dropna()
            if len(patient_nulls) > 0:
                low = np.percentile(patient_nulls, 2.5)
                high = np.percentile(patient_nulls, 97.5)
        
        # Fallback to stored stats if available
        if low is None or high is None:
            mean_col = f'null_{measure}_mean'
            std_col = f'null_{measure}_std'
            if mean_col in patient_sig.index and std_col in patient_sig.index:
                mean_val = patient_sig[mean_col]
                std_val = patient_sig[std_col]
                if not pd.isna(mean_val) and not pd.isna(std_val):
                    low = mean_val - 1.96 * std_val
                    high = mean_val + 1.96 * std_val
        
        if low is not None and high is not None:
            plt.fill_between(
                [i - 0.4, i + 0.4], [low, low], [high, high],
                color='#4682B4', alpha=0.2, edgecolor=None, zorder=0
            )
    
    # Draw zero line
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    
    # Draw box plot
    with plt.rc_context({'legend.frameon': False}):
        sns.boxplot(
            data=plot_df,
            x='user_key',
            y='spearman_z',
            order=sorted_patients,
            palette=palette,
            hue='user_key',
            legend=False,
            fliersize=2,
            linewidth=1.2,
            zorder=10,
            boxprops=dict(alpha=0.6)
        )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Bonferroni Significant (Concordance)'),
        mpatches.Patch(color='#FF6B6B', label='Bonferroni Significant (Discordance)'),
        mpatches.Patch(color='#DAA520', label='Nominally Significant (p < 0.05)'),
        mpatches.Patch(color='#808080', label='Non-Significant'),
        mpatches.Patch(facecolor='#4682B4', alpha=0.2, label='Null 95% CI')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title(f'Distribution of Correlations - Significance by {config["display_name"]}', 
              fontsize=14, pad=20)
    plt.xlabel('Patient ID', fontsize=12)
    plt.ylabel("Fisher's Z-Score (Spearman)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    output_path = output_dir / 'concordance_distribution_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MEDIAN POINT PLOT WITH NULL BAND (from create_refined_plots.py)
# =============================================================================

def plot_point_with_null_band(
    detailed_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    null_df: Optional[pd.DataFrame],
    output_dir: Path,
    measure: str = 'median',
    filter_type: str = 'all',
    title_suffix: str = ''
):
    """
    Simplified point plot showing observed statistic with null band.
    """
    config = MEASURE_CONFIG[measure]
    obs_col = f'observed_{measure}_z'
    null_col = f'null_{measure}_z'
    
    logger.info(f"Generating Point Plot ({filter_type}) for {config['display_name']}...")
    
    # Merge data
    merge_cols = ['patient_id', 'significant_bonferroni', 'significant_uncorrected', obs_col]
    
    # Add null stats if available
    null_mean_col = f'null_{measure}_mean'
    null_std_col = f'null_{measure}_std'
    if null_mean_col in sig_df.columns:
        merge_cols.append(null_mean_col)
    if null_std_col in sig_df.columns:
        merge_cols.append(null_std_col)
    
    # Filter to columns that exist
    merge_cols = [c for c in merge_cols if c in sig_df.columns]
    
    plot_df = detailed_df.merge(
        sig_df[merge_cols], 
        left_on='user_key', 
        right_on='patient_id', 
        how='inner'
    )
    
    if len(plot_df) == 0:
        logger.warning("No common patients found!")
        return
    
    # Get observed values per patient
    patient_obs = plot_df.groupby('user_key')[obs_col].first() if obs_col in plot_df.columns else \
                  plot_df.groupby('user_key')['spearman_z'].median()
    
    # For sorting, we use the observed statistic for this measure
    if obs_col in sig_df.columns:
        # Use observed from sig_df
        obs_sorted = sig_df.set_index('patient_id')[obs_col].sort_values()
        sorted_patients = [p for p in obs_sorted.index if p in plot_df['user_key'].values]
    else:
        # Fallback to median
        patient_medians = plot_df.groupby('user_key')['spearman_z'].median().sort_values()
        sorted_patients = patient_medians.index.tolist()
    
    # Apply filter
    if filter_type == 'significant':
        sig_patients = get_significant_patients(sig_df, bonferroni=True)
        sorted_patients = [p for p in sorted_patients if p in sig_patients]
    elif filter_type == 'optimal':
        optimal_patients = get_concordant_patients(sig_df, measure, CONCORDANCE_Z_THRESHOLD, True)
        sorted_patients = [p for p in sorted_patients if p in optimal_patients]
    
    if len(sorted_patients) == 0:
        logger.warning(f"No patients match filter '{filter_type}'")
        return
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = []
    observed_vals = []
    null_lows = []
    null_highs = []
    
    for pid in sorted_patients:
        patient_sig = sig_df[sig_df['patient_id'] == pid].iloc[0]
        
        # Get observed value
        if obs_col in patient_sig.index:
            obs_val = patient_sig[obs_col]
        else:
            # Compute from detailed
            obs_val = plot_df[plot_df['user_key'] == pid]['spearman_z'].median()
        observed_vals.append(obs_val)
        
        # Get null band
        low, high = None, None
        if null_df is not None and not null_df.empty:
            patient_nulls = null_df[null_df['patient_id'] == pid][null_col].dropna()
            if len(patient_nulls) > 0:
                low = np.percentile(patient_nulls, 2.5)
                high = np.percentile(patient_nulls, 97.5)
        
        if low is None or high is None:
            if null_mean_col in patient_sig.index and null_std_col in patient_sig.index:
                mean_val = patient_sig[null_mean_col]
                std_val = patient_sig[null_std_col]
                if not pd.isna(mean_val) and not pd.isna(std_val):
                    low = mean_val - 1.96 * std_val
                    high = mean_val + 1.96 * std_val
        
        null_lows.append(low if low is not None else np.nan)
        null_highs.append(high if high is not None else np.nan)
        
        # Determine color
        is_sig_strict = patient_sig['significant_bonferroni']
        is_sig_nominal = patient_sig['significant_uncorrected']
        
        if is_sig_strict:
            colors.append('#2E8B57' if obs_val >= 0 else '#FF6B6B')
        elif is_sig_nominal:
            colors.append('#DAA520')
        else:
            colors.append('#808080')
    
    x_positions = np.arange(len(sorted_patients))
    
    # Draw null bands
    for i, (low, high) in enumerate(zip(null_lows, null_highs)):
        if not np.isnan(low) and not np.isnan(high):
            ax.fill_between([i - 0.3, i + 0.3], [low, low], [high, high],
                           color='#4682B4', alpha=0.25, edgecolor=None, zorder=0)
    
    # Draw reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    ax.axhline(0.5, color='#FF6B00', linestyle='--', linewidth=1.5, 
               alpha=0.7, zorder=2, label='Concordance Threshold (Z = 0.5)')
    
    # Plot points
    ax.scatter(x_positions, observed_vals, c=colors, s=120, zorder=10, 
               edgecolors='black', linewidth=0.5)
    
    # Add IQR error bars from multiverse
    for i, pid in enumerate(sorted_patients):
        patient_data = plot_df[plot_df['user_key'] == pid]['spearman_z']
        q25, q75 = patient_data.quantile([0.25, 0.75])
        ax.plot([i, i], [q25, q75], color=colors[i], linewidth=2, alpha=0.6, zorder=5)
    
    # Labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_patients, rotation=45)
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel(f"Fisher's Z-Score ({config['display_name']})", fontsize=12)
    
    title = f'Per-Patient {config["display_name"]} Correlation with Null Band{title_suffix}'
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Significant Concordance'),
        mpatches.Patch(color='#FF6B6B', label='Significant Discordance'),
        mpatches.Patch(color='#DAA520', label='Nominally Significant'),
        mpatches.Patch(color='#808080', label='Non-Significant'),
        mpatches.Patch(facecolor='#4682B4', alpha=0.25, label='Null 95% CI'),
        plt.Line2D([0], [0], color='#FF6B00', linestyle='--', linewidth=1.5, label='Concordance Threshold (Z ≥ 0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    filename = f'{measure}_null_band_{filter_type}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# CONCORDANCE DIRECTION PLOT (from create_refined_plots.py)
# =============================================================================

def plot_concordance_direction(
    detailed_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_dir: Path,
    measure: str = 'median',
    filter_type: str = 'all',
    title_suffix: str = ''
):
    """
    Bar chart showing concordance direction (positive/negative).
    """
    config = MEASURE_CONFIG[measure]
    obs_col = f'observed_{measure}_z'
    
    logger.info(f"Generating Concordance Direction Plot ({filter_type}) for {config['display_name']}...")
    
    # Get observed values
    if obs_col in sig_df.columns:
        patient_obs = sig_df.set_index('patient_id')[obs_col].sort_values()
    else:
        patient_obs = detailed_df.groupby('user_key')['spearman_z'].median().sort_values()
    
    sorted_patients = [str(p) for p in patient_obs.index if str(p) in sig_df['patient_id'].values]
    
    # Apply filter
    if filter_type == 'significant':
        sig_patients = get_significant_patients(sig_df, bonferroni=True)
        sorted_patients = [p for p in sorted_patients if p in sig_patients]
    elif filter_type == 'optimal':
        optimal_patients = get_concordant_patients(sig_df, measure, CONCORDANCE_Z_THRESHOLD, True)
        sorted_patients = [p for p in sorted_patients if p in optimal_patients]
    elif filter_type == 'nominal_and_significant':
        nom_or_sig = sig_df[sig_df['significant_uncorrected'] | sig_df['significant_bonferroni']]['patient_id'].tolist()
        sorted_patients = [p for p in sorted_patients if p in nom_or_sig]
    
    if len(sorted_patients) == 0:
        logger.warning(f"No patients match filter '{filter_type}'")
        return
    
    fig, ax = plt.subplots(figsize=(max(8, len(sorted_patients) * 0.8), 6))
    
    x_positions = np.arange(len(sorted_patients))
    obs_values = [float(patient_obs[p]) if p in patient_obs.index else 0 for p in sorted_patients]
    
    # Color by direction
    colors = ['#2E8B57' if m >= 0 else '#FF6B6B' for m in obs_values]
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
    ax.bar(x_positions, obs_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_patients, rotation=45)
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel(f"{config['display_name']} Fisher's Z-Score", fontsize=12)
    ax.set_title(f'Concordance Direction by Patient ({config["display_name"]}){title_suffix}', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='Concordance (Z ≥ 0)'),
        mpatches.Patch(color='#FF6B6B', label='Discordance (Z < 0)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    filename = f'{measure}_concordance_direction_{filter_type}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# SIGNIFICANCE PLOT (from create_refined_plots.py)
# =============================================================================

def plot_significance(
    sig_df: pd.DataFrame,
    output_dir: Path,
    measure: str = 'median'
):
    """
    Plot -log10(p-value) bar chart.
    """
    config = MEASURE_CONFIG[measure]
    
    logger.info(f"Generating Significance Plot for {config['display_name']}...")
    
    sig_df = sig_df.sort_values('p_value').copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_positions = np.arange(len(sig_df))
    neg_log_p = -np.log10(sig_df['p_value'].clip(lower=1e-10))
    
    # Colors based on significance
    colors = []
    for _, row in sig_df.iterrows():
        if row['significant_bonferroni']:
            colors.append('#2E8B57')
        elif row['significant_uncorrected']:
            colors.append('#DAA520')
        else:
            colors.append('#808080')
    
    ax.bar(x_positions, neg_log_p, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Threshold lines
    bonferroni_threshold = -np.log10(0.05 / len(sig_df))
    nominal_threshold = -np.log10(0.05)
    
    ax.axhline(bonferroni_threshold, color='#2E8B57', linestyle='--', linewidth=1.5, 
               label=f'Bonferroni (p < {0.05/len(sig_df):.4f})')
    ax.axhline(nominal_threshold, color='#DAA520', linestyle='--', linewidth=1.5,
               label='Nominal (p < 0.05)')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sig_df['patient_id'], rotation=45)
    ax.set_xlabel('Patient ID (sorted by p-value)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title(f'Statistical Significance by Patient ({config["display_name"]})', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    
    output_path = output_dir / f'{measure}_significance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# SIGNIFICANCE TABLE (from create_refined_plots.py)
# =============================================================================

def plot_significance_table(
    sig_df: pd.DataFrame,
    output_dir: Path,
    measure: str = 'median'
):
    """
    Create a table image showing p-values and significance.
    """
    config = MEASURE_CONFIG[measure]
    obs_col = f'observed_{measure}_z'
    
    logger.info(f"Generating Significance Table for {config['display_name']}...")
    
    sig_df = sig_df.sort_values('p_value').reset_index(drop=True).copy()
    
    table_data = []
    cell_colors = []
    
    for _, row in sig_df.iterrows():
        pid = row['patient_id']
        p = row['p_value']
        
        # Get observed value
        if obs_col in row.index and not pd.isna(row[obs_col]):
            obs_z = row[obs_col]
        else:
            obs_z = row.get('observed_median_z', np.nan)
        
        # Format p-value
        if p < 0.0001:
            p_str = '< 0.0001'
        elif p < 0.001:
            p_str = f'{p:.4f}'
        else:
            p_str = f'{p:.3f}'
        
        # Significance level
        if row['significant_bonferroni']:
            sig_level = '✓✓ Bonferroni'
            row_color = ['#C8E6C9'] * 4
        elif row['significant_uncorrected']:
            sig_level = '✓ Nominal'
            row_color = ['#FFF9C4'] * 4
        else:
            sig_level = '✗ Not Sig.'
            row_color = ['#F5F5F5'] * 4
        
        z_str = f'{obs_z:.3f}' if not pd.isna(obs_z) else 'N/A'
        
        table_data.append([pid, p_str, z_str, sig_level])
        cell_colors.append(row_color)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    columns = ['Patient ID', 'P-Value', f'{config["display_name"]} Z', 'Significance']
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=['#E3F2FD'] * 4,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight='bold')
    
    bonf_thresh = 0.05 / len(sig_df)
    ax.set_title(f'Permutation Test Results: {config["display_name"]}\n(Bonferroni threshold: p < {bonf_thresh:.4f})', 
                fontsize=14, fontweight='bold', pad=20)
    
    legend_text = '✓✓ Bonferroni Significant  |  ✓ Nominally Significant  |  ✗ Not Significant'
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')
    
    output_path = output_dir / f'{measure}_significance_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# SPECIFICATION CURVE (from create_refined_plots.py)
# =============================================================================

def plot_specification_curve(
    detailed_df: pd.DataFrame,
    patient_id: str,
    output_dir: Path,
    measure: str = 'median'
):
    """
    Generate specification curve for a single patient.
    """
    config = MEASURE_CONFIG[measure]
    
    logger.info(f"Generating Specification Curve for Patient {patient_id} ({config['display_name']})...")
    
    df = detailed_df[detailed_df['user_key'] == str(patient_id)].copy()
    
    if len(df) == 0:
        logger.warning(f"No data for patient {patient_id}")
        return
    
    df = df.sort_values('spearman_z').reset_index(drop=True)
    df['rank'] = df.index
    
    specs = {
        'Time Window': ['timestamp_window', [12, 24, 36, 48]],
        'Window Type': ['use_daily_max_windows', [False, True], ['Rolling', 'Daily Max']],
        'Day Type': ['use_calendar_days', [False, True], ['24h Period', 'Calendar Day']],
        'Zero Handling': ['filter_out_zero_usage_entries', [False, True], ['Keep Zeros', 'Drop Zeros']],
        'Categorization': ['categorization_method', [
            'one_hot', 'lower_bound', 'midpoint', 'upper_bound', 
            'midpoint_with_inhaler', 'upper_bound_with_inhaler'
        ]]
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.05)
    
    # Top: Curve
    ax_curve = fig.add_subplot(gs[0])
    ax_curve.scatter(df.index, df['spearman_z'], c='#4169E1', s=20, zorder=2, alpha=0.7)
    ax_curve.plot(df.index, df['spearman_z'], color='#4169E1', linewidth=0.8, zorder=1, alpha=0.5)
    
    # Add statistic line based on measure
    if measure == 'median':
        stat_val = df['spearman_z'].median()
    elif measure == 'mean':
        stat_val = df['spearman_z'].mean()
    elif measure == 'q25':
        stat_val = df['spearman_z'].quantile(0.25)
    elif measure == 'q75':
        stat_val = df['spearman_z'].quantile(0.75)
    elif measure == 'min':
        stat_val = df['spearman_z'].min()
    elif measure == 'max':
        stat_val = df['spearman_z'].max()
    else:
        stat_val = df['spearman_z'].median()
    
    ax_curve.axhline(stat_val, color='#FF6B6B', linestyle='--', linewidth=1.5, 
                     label=f'{config["display_name"]}: {stat_val:.3f}')
    ax_curve.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    
    ax_curve.set_ylabel("Fisher's Z-Score", fontsize=12)
    ax_curve.set_title(f'Specification Curve: Patient {patient_id} ({config["display_name"]})', 
                       fontsize=14, pad=15)
    ax_curve.grid(True, linestyle='--', alpha=0.3)
    ax_curve.set_xticklabels([])
    ax_curve.set_xlim(-2, len(df)+2)
    ax_curve.legend(loc='upper left')
    
    # Bottom: Barcode
    ax_bar = fig.add_subplot(gs[1], sharex=ax_curve)
    
    y_pos = 0
    y_labels = []
    
    for group_name, spec_info in specs.items():
        col = spec_info[0]
        values = spec_info[1]
        labels = spec_info[2] if len(spec_info) > 2 else [str(v) for v in values]
        
        ax_bar.text(-5, y_pos - (len(values)/2) + 0.5, group_name, 
                   fontsize=10, fontweight='bold', va='center', ha='right')
        
        for i, val in enumerate(values):
            matches = df[col] == val
            x_matches = df.index[matches]
            ax_bar.scatter(x_matches, [y_pos] * len(x_matches), 
                          marker='|', color='#333333', s=60, linewidth=0.8)
            
            label_text = labels[i] if isinstance(labels[i], str) else str(labels[i])
            label_text = label_text.replace('_', ' ').title()
            y_labels.append((y_pos, label_text))
            
            y_pos -= 1
        
        y_pos -= 0.5
    
    ax_bar.set_yticks([y[0] for y in y_labels])
    ax_bar.set_yticklabels([y[1] for y in y_labels], fontsize=9)
    ax_bar.set_ylim(y_pos + 0.5, 0.5)
    ax_bar.set_xlabel('Specifications (sorted by correlation)', fontsize=12)
    ax_bar.grid(axis='y', linestyle='-', alpha=0.2)
    
    save_path = output_dir / f'{measure}_specification_curve_{patient_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# NULL DISTRIBUTION HISTOGRAMS
# =============================================================================

def plot_null_distributions(
    null_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_dir: Path,
    measure: str = 'median',
    max_plots: int = 6
):
    """
    Plot null distribution histograms for significant patients.
    """
    config = MEASURE_CONFIG[measure]
    obs_col = f'observed_{measure}_z'
    null_col = f'null_{measure}_z'
    
    logger.info(f"Generating Null Distribution Plots for {config['display_name']}...")
    
    sig_patients = get_significant_patients(sig_df, bonferroni=True)
    
    if len(sig_patients) == 0:
        logger.warning("No significant patients to plot")
        return
    
    n_plots = min(len(sig_patients), max_plots)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, patient_id in enumerate(sig_patients[:n_plots]):
        ax = axes[i]
        
        # Get null distribution
        null_vals = null_df[null_df['patient_id'] == patient_id][null_col].dropna().values
        
        # Get observed value
        patient_sig = sig_df[sig_df['patient_id'] == patient_id].iloc[0]
        if obs_col in patient_sig.index:
            obs_val = patient_sig[obs_col]
        else:
            obs_val = patient_sig.get('observed_median_z', np.nan)
        
        p_val = patient_sig['p_bonferroni']
        
        # Plot
        ax.hist(null_vals, bins=30, alpha=0.7, edgecolor='black', color='#4682B4')
        ax.axvline(obs_val, color='red', linewidth=2, label=f'Observed: {obs_val:.3f}')
        ax.set_xlabel(f'{config["display_name"]} Fisher Z')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Patient {patient_id} (p = {p_val:.4f})')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Null Distributions - {config["display_name"]}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f'{measure}_null_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# GENERATE ALL PLOTS FOR A MEASURE
# =============================================================================

def generate_all_plots_for_measure(
    detailed_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    null_df: Optional[pd.DataFrame],
    output_dir: Path,
    measure: str = 'median'
):
    """
    Generate all standard plots for a given measure.
    """
    config = MEASURE_CONFIG[measure]
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating all plots for {config['display_name']}")
    logger.info(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Box plot with null bands
    plot_distribution_boxplot(detailed_df, sig_df, null_df, output_dir, measure)
    
    # 2. Point plots with null bands (all, significant, optimal)
    for filter_type, suffix in [
        ('all', ' (All Patients)'),
        ('significant', ' (Bonferroni Significant)'),
        ('optimal', ' (Concordant Patients)')
    ]:
        plot_point_with_null_band(detailed_df, sig_df, null_df, output_dir, 
                                  measure, filter_type, suffix)
    
    # 3. Concordance direction plots
    for filter_type, suffix in [
        ('all', ' (All Patients)'),
        ('significant', ' (Bonferroni Significant)'),
        ('optimal', ' (Concordant)'),
        ('nominal_and_significant', ' (Nominally + Bonferroni Significant)')
    ]:
        plot_concordance_direction(detailed_df, sig_df, output_dir, measure, filter_type, suffix)
    
    # 4. Significance plots
    plot_significance(sig_df, output_dir, measure)
    plot_significance_table(sig_df, output_dir, measure)
    
    # 5. Null distribution histograms (when available)
    if null_df is not None and not null_df.empty:
        plot_null_distributions(null_df, sig_df, output_dir, measure)
    
    # 6. Specification curves for significant patients
    sig_patients = get_significant_patients(sig_df, bonferroni=True)
    if sig_patients:
        logger.info(f"Generating specification curves for {len(sig_patients)} significant patients...")
        for patient_id in sig_patients:
            plot_specification_curve(detailed_df, patient_id, output_dir, measure)
    
    logger.info(f" All plots generated for {config['display_name']}")

