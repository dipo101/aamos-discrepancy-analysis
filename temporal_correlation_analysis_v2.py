"""
Temporal Correlation Analysis
==============================
Generates temporal correlation plots for concordant vs remaining patients.

Usage:
    python temporal_correlation_analysis_v2.py [--group concordant|median_concordant]

Groups are defined in config.py and selected via the --group argument.

Outputs:
1. Group comparison plot (main text)
2. Individual concordant user plots (supplementary)
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
from config import GROUPS, ASSESSED_PATIENTS, get_group_config, get_remaining_patients, get_output_dir, add_world_argument, set_active_world, record_run_provenance

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent

# Analysis parameters
WINDOW_SIZE = 30  # days

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare data for temporal correlation analysis."""
    from data_loader import AsthmaDataLoader, DataLoaderConfig, CategorizationMethod
    
    config = DataLoaderConfig(
        min_duration_threshold=0,
        timestamp_window_hours=24,
        use_end_date=True,
        drop_dates_less_than_0=False,
        use_daily_max_windows=True,
        filter_out_zero_usage_entries=False,
        use_calendar_days=False,
        compare_inhaler_dates=False,
        categorization_method=CategorizationMethod.LOWER_BOUND,
        skip_inhaler_filtering=True,
        log_level=40  # ERROR only
    )
    
    loader = AsthmaDataLoader(config=config)
    df = loader.load_data()
    
    # Filter to assessed patients only
    df = df[df['user_key'].isin(ASSESSED_PATIENTS)].copy()
    
    # The 'date' column from data_loader already contains numeric day values (0, 1, 2, ...)
    df['date_numeric'] = df['date'].astype(int)
    
    print(f"Days range: {df['date_numeric'].min()} to {df['date_numeric'].max()}")
    
    return df


# =============================================================================
# ROLLING CORRELATION FUNCTIONS
# =============================================================================

def fisher_z_transform(r: float) -> float:
    """Apply Fisher's Z transformation to a correlation coefficient."""
    # Clamp to avoid infinity
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Convert Fisher's Z back to correlation coefficient."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def calculate_rolling_correlation(
    df: pd.DataFrame, 
    window_size: int = 30
) -> Tuple[List, List, List, List, List]:
    """
    Calculate rolling Spearman correlation for a single user.
    
    Returns:
        Tuple of (dates, correlations, fisher_z_values, p_values, sample_sizes)
    """
    df = df.sort_values('date_numeric').reset_index(drop=True)
    
    dates = []
    correlations = []
    fisher_z_values = []
    p_values = []
    sample_sizes = []
    
    for i in range(window_size, len(df) + 1):
        window = df.iloc[i - window_size:i]
        n = len(window)
        
        if n >= 5:  # Minimum samples for correlation
            try:
                corr, p = stats.spearmanr(
                    window['inhaler_usage'], 
                    window['daily_relief_inhaler']
                )
                if np.isnan(corr):
                    corr = 0
                    p = 1
                fisher_z = fisher_z_transform(corr)
            except:
                corr = 0
                p = 1
                fisher_z = 0
        else:
            corr = np.nan
            p = np.nan
            fisher_z = np.nan
        
        dates.append(window['date_numeric'].iloc[-1])
        correlations.append(corr)
        fisher_z_values.append(fisher_z)
        p_values.append(p)
        sample_sizes.append(n)
    
    return dates, correlations, fisher_z_values, p_values, sample_sizes


def get_group_rolling_correlations(
    df: pd.DataFrame, 
    user_ids: List[int], 
    window_size: int = 30
) -> pd.DataFrame:
    """Calculate rolling correlations for all users in a group."""
    all_data = []
    
    for user_id in user_ids:
        user_data = df[df['user_key'] == user_id]
        if len(user_data) < window_size:
            continue
            
        dates, correlations, fisher_z_values, p_values, sample_sizes = calculate_rolling_correlation(
            user_data, window_size
        )
        
        user_df = pd.DataFrame({
            'date': dates,
            'correlation': correlations,
            'fisher_z': fisher_z_values,
            'p_value': p_values,
            'sample_size': sample_sizes,
            'user_key': user_id
        })
        all_data.append(user_df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def compute_weighted_group_summary(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted mean Fisher's Z per date, with proper weighting by (n-3).
    
    Returns DataFrame with columns: date, weighted_mean_z, weighted_mean_r, se_z, ci_lower_r, ci_upper_r
    """
    results = []
    
    for date in sorted(rolling_df['date'].unique()):
        day_data = rolling_df[rolling_df['date'] == date].copy()
        
        # Filter to valid observations (n > 3)
        day_data = day_data[day_data['sample_size'] > 3].copy()
        day_data = day_data[np.isfinite(day_data['fisher_z'])].copy()
        
        if len(day_data) == 0:
            continue
        
        # Weights = n - 3 (inverse variance weighting)
        day_data['weight'] = day_data['sample_size'] - 3
        
        # Weighted mean Z
        total_weight = day_data['weight'].sum()
        weighted_z = (day_data['weight'] * day_data['fisher_z']).sum() / total_weight
        
        # Standard error of weighted mean
        se_z = np.sqrt(1.0 / total_weight)
        
        # 95% CI in Z-space
        ci_lower_z = weighted_z - 1.96 * se_z
        ci_upper_z = weighted_z + 1.96 * se_z
        
        # Convert back to correlation space
        weighted_r = inverse_fisher_z(weighted_z)
        ci_lower_r = inverse_fisher_z(ci_lower_z)
        ci_upper_r = inverse_fisher_z(ci_upper_z)
        
        results.append({
            'date': date,
            'weighted_mean_z': weighted_z,
            'weighted_mean_r': weighted_r,
            'se_z': se_z,
            'ci_lower_z': ci_lower_z,
            'ci_upper_z': ci_upper_z,
            'ci_lower_r': ci_lower_r,
            'ci_upper_r': ci_upper_r,
            'n_patients': len(day_data),
            'total_weight': total_weight
        })
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_group_comparison(
    optimal_summary: pd.DataFrame,
    remaining_summary: pd.DataFrame,
    save_path: Path,
    group_label: str = "Concordant",
    n_concordant: int = 3,
    n_remaining: int = 12
):
    """
    Plot group comparison (Part a) - for main text.
    Uses weighted Fisher's Z means with proper 95% CI bands.
    Plots in Fisher's Z space for consistency with other plots.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(optimal_summary['date'], optimal_summary['weighted_mean_z'],
            color='#1f77b4', linewidth=2, label=f'{group_label} (n={n_concordant}, weighted mean)')
    ax.fill_between(
        optimal_summary['date'],
        optimal_summary['ci_lower_z'],
        optimal_summary['ci_upper_z'],
        color='#1f77b4', alpha=0.2, label=f'{group_label} (95% CI)'
    )

    ax.plot(remaining_summary['date'], remaining_summary['weighted_mean_z'],
            color='#d62728', linewidth=2, label=f'Remaining (n={n_remaining}, weighted mean)')
    ax.fill_between(
        remaining_summary['date'],
        remaining_summary['ci_lower_z'],
        remaining_summary['ci_upper_z'],
        color='#d62728', alpha=0.2, label='Remaining (95% CI)'
    )
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # Z = 0.5 threshold (corresponds to ρ ≈ 0.46, moderate-to-large effect)
    fisher_z_threshold = 0.5
    ax.axhline(y=fisher_z_threshold, color='orange', linestyle='--', alpha=0.7, 
               label=f'Z = {fisher_z_threshold:.1f} (ρ ≈ 0.46)')
    
    ax.set_xlabel('Days in Study', fontsize=12)
    ax.set_ylabel('Fisher Z-transformed Spearman Correlation', fontsize=12)
    ax.set_title('Temporal Correlation Analysis: 30-Day Rolling Spearman Correlation\n(Weighted by sample size, 95% CI)', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    
    # Dynamic y-axis limits based on data
    all_upper = pd.concat([optimal_summary['ci_upper_z'], remaining_summary['ci_upper_z']])
    all_lower = pd.concat([optimal_summary['ci_lower_z'], remaining_summary['ci_lower_z']])
    y_max = all_upper.max() + 0.3
    y_min = min(all_lower.min() - 0.3, -0.5)  # Ensure we show below zero
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved group comparison plot to: {save_path}")


def plot_individual_optimal_users(
    optimal_rolling: pd.DataFrame,
    remaining_summary: pd.DataFrame,
    save_path: Path
):
    """
    Plot individual optimal users vs remaining group (Part b) - for supplementary.
    Uses Fisher's Z space with weighted summary for remaining group.
    """
    optimal_users = sorted(optimal_rolling['user_key'].unique())
    n_users = len(optimal_users)
    
    fig, axes = plt.subplots(1, n_users, figsize=(5 * n_users, 5), sharey=True)
    if n_users == 1:
        axes = [axes]
    
    fisher_z_threshold = 0.5  # Z = 0.5 threshold (ρ ≈ 0.46)
    
    for i, user_id in enumerate(optimal_users):
        ax = axes[i]
        user_data = optimal_rolling[optimal_rolling['user_key'] == user_id].sort_values('date')
        
        # Remaining group background (weighted 95% CI in Z space)
        ax.fill_between(
            remaining_summary['date'],
            remaining_summary['ci_lower_z'],
            remaining_summary['ci_upper_z'],
            color='#d62728', alpha=0.2, label='Remaining (95% CI)'
        )
        ax.plot(remaining_summary['date'], remaining_summary['weighted_mean_z'],
                color='#d62728', linewidth=1, linestyle='--', alpha=0.7)
        
        # User line using Fisher's Z - solid for significant, dashed for not
        dates = user_data['date'].values
        fisher_z = user_data['fisher_z'].values
        p_vals = user_data['p_value'].values
        
        # Plot segments with different styles based on significance
        for j in range(len(dates) - 1):
            style = '-' if p_vals[j] < 0.05 else '--'
            ax.plot([dates[j], dates[j+1]], [fisher_z[j], fisher_z[j+1]], 
                    color='#1f77b4', linestyle=style, linewidth=1.5)
        
        # Add markers
        sig_mask = p_vals < 0.05
        ax.scatter(dates[sig_mask], fisher_z[sig_mask], 
                   color='#1f77b4', s=20, zorder=5, label='Significant')
        ax.scatter(dates[~sig_mask], fisher_z[~sig_mask], 
                   color='#1f77b4', s=20, alpha=0.4, zorder=5)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=fisher_z_threshold, color='orange', linestyle='--', alpha=0.5, 
                   label=f'Z = {fisher_z_threshold:.2f}')
        ax.set_xlabel('Days in Study')
        ax.set_title(f'Patient {user_id}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_ylabel('Fisher Z-transformed Correlation')
    
    # Dynamic shared y-axis based on all data
    all_z = optimal_rolling['fisher_z'].dropna()
    y_max = max(all_z.max(), remaining_summary['ci_upper_z'].max()) + 0.3
    y_min = min(all_z.min(), remaining_summary['ci_lower_z'].min(), -0.5) - 0.3
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    
    fig.suptitle('Individual Concordant Users vs Remaining Group (Fisher Z)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual users plot to: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal correlation analysis.")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="concordant",
                        help="Patient group definition to use (default: concordant)")
    add_world_argument(parser)
    args = parser.parse_args()
    set_active_world(args.world, args.summary)
    return args


def main():
    args = parse_args()
    group_cfg = get_group_config(args.group)
    CONCORDANT_PATIENTS = group_cfg["patients"]
    group_label = group_cfg["label"]
    REMAINING_PATIENTS = get_remaining_patients(args.group)
    OUTPUT_DIR = get_output_dir(args.group, "temporal_correlation")
    record_run_provenance(OUTPUT_DIR, group_name=args.group, script="temporal_correlation_analysis_v2.py")

    print("="*60)
    print("TEMPORAL CORRELATION ANALYSIS")
    print("="*60)
    print(f"{group_label} patients (n={len(CONCORDANT_PATIENTS)}): {CONCORDANT_PATIENTS}")
    print(f"Remaining patients (n={len(REMAINING_PATIENTS)}): {REMAINING_PATIENTS}")
    print()

    # Load data
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} observations from {df['user_key'].nunique()} patients")
    
    # Calculate rolling correlations
    print(f"\nCalculating {WINDOW_SIZE}-day rolling correlations...")
    
    concordant_rolling = get_group_rolling_correlations(df, CONCORDANT_PATIENTS, WINDOW_SIZE)
    remaining_rolling = get_group_rolling_correlations(df, REMAINING_PATIENTS, WINDOW_SIZE)

    print(f"{group_label} group: {len(concordant_rolling)} data points")
    print(f"Remaining group: {len(remaining_rolling)} data points")

    print("\nComputing weighted group summaries...")
    concordant_summary = compute_weighted_group_summary(concordant_rolling)
    remaining_summary = compute_weighted_group_summary(remaining_rolling)

    print(f"{group_label} summary: {len(concordant_summary)} time points")
    print(f"Remaining summary: {len(remaining_summary)} time points")

    print("\nGenerating plots...")

    plot_group_comparison(
        concordant_summary,
        remaining_summary,
        OUTPUT_DIR / "temporal_correlation_group_comparison.png"
    )

    plot_individual_optimal_users(
        concordant_rolling,
        remaining_summary,
        OUTPUT_DIR / "temporal_correlation_individual_users.png"
    )

    print("\nSummary Statistics (Fisher's Z):")
    print("-" * 40)
    conc_mean_z = concordant_rolling['fisher_z'].mean()
    conc_std_z = concordant_rolling['fisher_z'].std()
    rem_mean_z = remaining_rolling['fisher_z'].mean()
    rem_std_z = remaining_rolling['fisher_z'].std()

    print(f"{group_label} group mean Fisher's Z: {conc_mean_z:.3f} (approx rho = {inverse_fisher_z(conc_mean_z):.3f})")
    print(f"{group_label} group std Fisher's Z: {conc_std_z:.3f}")
    print(f"Remaining group mean Fisher's Z: {rem_mean_z:.3f} (approx rho = {inverse_fisher_z(rem_mean_z):.3f})")
    print(f"Remaining group std Fisher's Z: {rem_std_z:.3f}")

    concordant_rolling.to_csv(OUTPUT_DIR / "concordant_rolling_correlations.csv", index=False)
    remaining_rolling.to_csv(OUTPUT_DIR / "remaining_rolling_correlations.csv", index=False)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

