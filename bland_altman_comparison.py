import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from data_loader import DataLoaderConfig, AsthmaDataLoader, CategorizationMethod
from config import GROUPS, ASSESSED_PATIENTS, get_group_config, get_remaining_patients, get_output_dir, record_run_provenance

def create_bland_altman_plot(data, group_name, save_path=None, ylim=None):
    """
    Create a Bland-Altman plot for a specific user group
    
    Parameters:
    data: DataFrame with 'reported_usage' and 'actual_usage' columns
    group_name: String name for the group (e.g., 'Optimal Users', 'Non-Optimal Users')
    save_path: Optional path to save the plot
    ylim: Optional tuple for y-axis limits
    """
    
    # Calculate mean and difference for Bland-Altman plot
    mean_usage = (data['reported_usage'] + data['actual_usage']) / 2
    difference = data['actual_usage'] - data['reported_usage']
    
    # Calculate statistics
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(mean_usage, difference, alpha=0.6, s=30)
    
    # Reference lines
    plt.axhline(mean_diff, color='gray', linestyle='--', linewidth=2, 
                label=f'Mean difference: {mean_diff:.2f}')
    plt.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5, 
                label=f'Upper LoA: {loa_upper:.2f}')
    plt.axhline(loa_lower, color='blue', linestyle='--', linewidth=1.5, 
                label=f'Lower LoA: {loa_lower:.2f}')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Labels and title
    plt.xlabel('Mean of Reported and Actual Usage (puffs/day)', fontsize=12)
    plt.ylabel('Difference (Actual - Reported) (puffs/day)', fontsize=12)
    plt.title(f'Bland-Altman Plot: {group_name}\n(n={len(data)} observations)', fontsize=14, fontweight='bold')
    
    if ylim:
        plt.ylim(ylim)
        
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean difference: {mean_diff:.2f}\nStd of difference: {std_diff:.2f}\n95% LoA: [{loa_lower:.2f}, {loa_upper:.2f}]'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Bland-Altman plot to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'loa_upper': loa_upper,
        'loa_lower': loa_lower,
        'n_observations': len(data)
    }

def create_bland_altman_percentage_plot(data, group_name, save_path=None, ylim=None):
    """
    Create a Bland-Altman plot using percentage difference.
    
    Parameters:
    data: DataFrame with 'reported_usage' and 'actual_usage' columns
    group_name: String name for the group
    save_path: Optional path to save the plot
    ylim: Optional tuple for y-axis limits
    """
    
    # Calculate mean and difference
    mean_usage = (data['reported_usage'] + data['actual_usage']) / 2
    difference = data['actual_usage'] - data['reported_usage']
    
    # Calculate percentage difference.
    # When mean_usage is 0, both reported and actual usage are 0. The difference is 0,
    # so we can logically define the percentage difference as 0 in this case.
    # We replace mean_usage of 0 with NaN to avoid division by zero errors, then fill the resulting NaNs with 0.
    percentage_difference = (difference / mean_usage.replace(0, np.nan) * 100).fillna(0)

    # In the unlikely event of infinities (e.g., from floating point inaccuracies where mean is ~0 but not exactly 0),
    # we'll remove those points.
    percentage_difference = percentage_difference.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure mean_usage for plotting aligns with the data points in percentage_difference
    mean_usage = mean_usage.loc[percentage_difference.index]

    # Calculate statistics for percentage difference
    mean_diff = np.mean(percentage_difference)
    std_diff = np.std(percentage_difference)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(mean_usage, percentage_difference, alpha=0.6, s=30)
    
    # Reference lines
    plt.axhline(mean_diff, color='gray', linestyle='--', linewidth=2, 
                label=f'Mean difference: {mean_diff:.2f}%')
    plt.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5, 
                label=f'Upper LoA: {loa_upper:.2f}%')
    plt.axhline(loa_lower, color='blue', linestyle='--', linewidth=1.5, 
                label=f'Lower LoA: {loa_lower:.2f}%')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Labels and title
    plt.xlabel('Mean of Reported and Actual Usage (puffs/day)', fontsize=12)
    plt.ylabel('Percentage Difference ((Actual - Reported) / Mean) (%)', fontsize=12)
    plt.title(f'Bland-Altman Plot (Percentage Difference): {group_name}\n(n={len(percentage_difference)} observations)', fontsize=14, fontweight='bold')
    
    # Force symmetrical y-axis to avoid misleading auto-scaling
    if ylim is None:
        max_abs_val = np.max(np.abs(percentage_difference))
        plt.ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
    else:
        plt.ylim(ylim)
        
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean difference: {mean_diff:.2f}%\nStd of difference: {std_diff:.2f}%\n95% LoA: [{loa_lower:.2f}%, {loa_upper:.2f}%]'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Percentage Bland-Altman plot to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean_diff_percent': mean_diff,
        'std_diff_percent': std_diff,
        'loa_upper_percent': loa_upper,
        'loa_lower_percent': loa_lower,
        'n_observations': len(percentage_difference)
    }

def create_bland_altman_log_ratio_plot(data, group_name, save_path=None, ylim=None):
    """
    Create a Bland-Altman plot using log ratios.
    This is suitable for data where the error is proportional to the mean.
    
    Parameters:
    data: DataFrame with 'reported_usage' and 'actual_usage' columns
    group_name: String name for the group
    save_path: Optional path to save the plot
    ylim: Optional tuple for y-axis limits
    """
    # Use a small constant to handle zero values in log transformation
    constant = 0.5
    
    # X-axis: standard mean. Y-axis: log ratio of (actual+c)/(reported+c)
    mean_usage = (data['reported_usage'] + data['actual_usage']) / 2
    log_ratio = np.log((data['actual_usage'] + constant) / (data['reported_usage'] + constant))
    
    # Calculate statistics on the log-transformed data
    mean_log_ratio = np.mean(log_ratio)
    std_log_ratio = np.std(log_ratio)
    loa_upper_log = mean_log_ratio + 1.96 * std_log_ratio
    loa_lower_log = mean_log_ratio - 1.96 * std_log_ratio
    
    # Back-transform the mean and limits of agreement to interpret them as ratios
    mean_ratio = np.exp(mean_log_ratio)
    loa_upper_ratio = np.exp(loa_upper_log)
    loa_lower_ratio = np.exp(loa_lower_log)

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(mean_usage, log_ratio, alpha=0.6, s=30)
    
    # Reference lines (on the log scale)
    plt.axhline(mean_log_ratio, color='gray', linestyle='--', linewidth=2, 
                label=f'Mean Ratio: {mean_ratio:.2f}')
    plt.axhline(loa_upper_log, color='red', linestyle='--', linewidth=1.5, 
                label=f'Upper LoA (Ratio): {loa_upper_ratio:.2f}')
    plt.axhline(loa_lower_log, color='blue', linestyle='--', linewidth=1.5, 
                label=f'Lower LoA (Ratio): {loa_lower_ratio:.2f}')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1) # Perfect agreement line (log(1)=0)
    
    # Labels and title
    plt.xlabel('Mean of Reported and Recorded Usage (puffs/day)', fontsize=12)
    plt.ylabel('Log Ratio (log(Recorded / Reported))', fontsize=12)
    plt.title(f'Bland-Altman Plot (Log Ratio): {group_name}\n(n={len(data)} observations)', fontsize=14, fontweight='bold')
    
    if ylim:
        plt.ylim(ylim)
        
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean Ratio: {mean_ratio:.2f}\n'
                  f'95% LoA for Ratio: [{loa_lower_ratio:.2f}, {loa_upper_ratio:.2f}]')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Log Ratio Bland-Altman plot to {save_path}")
    
    plt.tight_layout()
    plt.show()

    return {
        'mean_ratio': mean_ratio,
        'loa_upper_ratio': loa_upper_ratio,
        'loa_lower_ratio': loa_lower_ratio,
        'n_observations': len(data)
    }

def load_data_for_bland_altman(concordant_users, remaining_users):
    
    # Configuration for optimal users (normal filtering)
    optimal_config = DataLoaderConfig(
        min_duration_threshold=0,    # No duration filtering
        timestamp_window_hours=24,   # 24-hour time window (optimal)
        use_end_date=True,           # Use end dates alone (optimal)
        drop_dates_less_than_0=False, # Keep zero entries (optimal)
        use_daily_max_windows=True,  # Use fixed 24-hour chunks (optimal)
        filter_out_zero_usage_entries=False, # Keep zero entries (optimal)
        use_calendar_days=False,     # Don't use calendar days (optimal)
        compare_inhaler_dates=False, # Don't compare inhaler dates (optimal)
        categorization_method=CategorizationMethod.LOWER_BOUND, # Best Spearman method
        skip_inhaler_filtering=False, # Normal filtering for optimal users
        log_level=1  # WARNING level
    )
    
    # Configuration for non-optimal users (skip inhaler filtering)
    nonoptimal_config = DataLoaderConfig(
        min_duration_threshold=0,    # No duration filtering
        timestamp_window_hours=24,   # 24-hour time window (optimal)
        use_end_date=True,           # Use end dates alone (optimal)
        drop_dates_less_than_0=False, # Keep zero entries (optimal)
        use_daily_max_windows=True,  # Use fixed 24-hour chunks (optimal)
        filter_out_zero_usage_entries=False, # Keep zero entries (optimal)
        use_calendar_days=False,     # Don't use calendar days (optimal)
        compare_inhaler_dates=False, # Don't compare inhaler dates (optimal)
        categorization_method=CategorizationMethod.LOWER_BOUND, # Best Spearman method
        skip_inhaler_filtering=True, # Skip inhaler filtering to include users with nan inhaler_end_date
        log_level=1  # WARNING level
    )
    
    print("Loading concordant users data...")
    concordant_loader = AsthmaDataLoader(config=optimal_config)
    concordant_data = concordant_loader.load_data()

    print("Loading remaining users data...")
    nonoptimal_loader = AsthmaDataLoader(config=nonoptimal_config)
    all_data = nonoptimal_loader.load_data()

    concordant_data_from_all = all_data[all_data['user_key'].isin(concordant_users)].copy()
    remaining_data = all_data[all_data['user_key'].isin(remaining_users)].copy()

    print(f"Concordant users data: {len(concordant_data_from_all)} observations from {len(concordant_users)} patients")
    print(f"Remaining assessed users data: {len(remaining_data)} observations from {len(remaining_users)} patients")

    combined_data = pd.concat([concordant_data_from_all, remaining_data], ignore_index=True)

    print(f"Combined data: {len(combined_data)} observations")
    print(f"Unique users: {sorted(combined_data['user_key'].unique())}")

    return combined_data, concordant_users, remaining_users

def parse_args():
    parser = argparse.ArgumentParser(description="Bland-Altman comparison analysis.")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="concordant",
                        help="Patient group definition to use (default: concordant)")
    return parser.parse_args()


def main():
    args = parse_args()
    group_cfg = get_group_config(args.group)
    concordant_users = group_cfg["patients"]
    group_label = group_cfg["label"]
    remaining_users = get_remaining_patients(args.group)

    print(f"Creating Bland-Altman plots for {group_label} vs remaining assessed users...")

    try:
        data, concordant_users, remaining_users = load_data_for_bland_altman(concordant_users, remaining_users)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have the required data files:")
        print("- anonym_aamos00_patient_info.csv")
        print("- anonym_aamos00_dailyquestionnaire_dt.csv")
        print("- anonym_aamos00_smartinhaler_dt.csv")
        return

    data['reported_usage'] = data['daily_relief_inhaler']
    data['actual_usage'] = data['inhaler_usage']

    concordant_data = data[data['user_key'].isin(concordant_users)].copy()
    remaining_data = data[data['user_key'].isin(remaining_users)].copy()

    print(f"{group_label} users: {len(concordant_data)} observations from {len(concordant_users)} patients")
    print(f"Remaining assessed users: {len(remaining_data)} observations from {len(remaining_users)} patients")

    output_dir = get_output_dir(args.group, "bland_altman")
    record_run_provenance(output_dir, group_name=args.group, script="bland_altman_comparison.py")
    st_dir = output_dir / 'supplementary'
    st_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("MAIN PLOTS (Log Ratio Bland-Altman)")
    print("="*60)

    print(f"\nCreating Log Ratio Bland-Altman plot for {group_label} Users...")
    optimal_log_stats = create_bland_altman_log_ratio_plot(
        concordant_data,
        f'{group_label} Users',
        save_path=output_dir / 'bland_altman_log_ratio_concordant.png',
        ylim=[-3, 3]
    )

    print("\nCreating Log Ratio Bland-Altman plot for Remaining Users...")
    remaining_log_stats = create_bland_altman_log_ratio_plot(
        remaining_data,
        'Remaining Users',
        save_path=output_dir / 'bland_altman_log_ratio_remaining.png',
        ylim=[-3, 3]
    )

    print("\n" + "="*60)
    print("SUPPLEMENTARY PLOTS (Standard Bland-Altman)")
    print("="*60)

    print(f"\nCreating Standard Bland-Altman plot for {group_label} Users...")
    optimal_stats = create_bland_altman_plot(
        concordant_data,
        f'{group_label} Users',
        save_path=st_dir / 'bland_altman_standard_concordant.png',
        ylim=[-10, 8]
    )

    print("\nCreating Standard Bland-Altman plot for Remaining Users...")
    remaining_stats = create_bland_altman_plot(
        remaining_data,
        'Remaining Users',
        save_path=st_dir / 'bland_altman_standard_remaining.png',
        ylim=[-10, 8]
    )

    print(f"\nCreating Percentage Bland-Altman plot for {group_label} Users...")
    optimal_percent_stats = create_bland_altman_percentage_plot(
        concordant_data,
        f'{group_label} Users',
        save_path=st_dir / 'bland_altman_percentage_concordant.png',
        ylim=[-300, 300]
    )

    print("\nCreating Percentage Bland-Altman plot for Remaining Users...")
    remaining_percent_stats = create_bland_altman_percentage_plot(
        remaining_data,
        'Remaining Users',
        save_path=st_dir / 'bland_altman_percentage_remaining.png',
        ylim=[-300, 300]
    )

    # Print comparison summary
    print("\n" + "="*60)
    print("LOG RATIO BLAND-ALTMAN ANALYSIS SUMMARY (MAIN)")
    print("="*60)

    print(f"\n{group_label} Users (n={optimal_log_stats['n_observations']} obs, {len(concordant_users)} patients):")
    print(f"  Mean Ratio (Recorded/Reported): {optimal_log_stats['mean_ratio']:.3f}")
    print(f"  95% LoA for Ratio: [{optimal_log_stats['loa_lower_ratio']:.3f}, {optimal_log_stats['loa_upper_ratio']:.3f}]")
    
    print(f"\nRemaining Users (n={remaining_log_stats['n_observations']} obs, {len(remaining_users)} patients):")
    print(f"  Mean Ratio (Recorded/Reported): {remaining_log_stats['mean_ratio']:.3f}")
    print(f"  95% LoA for Ratio: [{remaining_log_stats['loa_lower_ratio']:.3f}, {remaining_log_stats['loa_upper_ratio']:.3f}]")
    
    print(f"\n" + "="*60)
    print("STANDARD BLAND-ALTMAN ANALYSIS SUMMARY (SUPPLEMENTARY)")
    print("="*60)
    
    print(f"\n{group_label} Users (n={optimal_stats['n_observations']}):")
    print(f"  Mean difference: {optimal_stats['mean_diff']:.3f}")
    print(f"  Std of difference: {optimal_stats['std_diff']:.3f}")
    print(f"  95% LoA: [{optimal_stats['loa_lower']:.3f}, {optimal_stats['loa_upper']:.3f}]")
    
    print(f"\nRemaining Users (n={remaining_stats['n_observations']}):")
    print(f"  Mean difference: {remaining_stats['mean_diff']:.3f}")
    print(f"  Std of difference: {remaining_stats['std_diff']:.3f}")
    print(f"  95% LoA: [{remaining_stats['loa_lower']:.3f}, {remaining_stats['loa_upper']:.3f}]")
    
    print(f"\n" + "="*60)
    print("PERCENTAGE BLAND-ALTMAN ANALYSIS SUMMARY (SUPPLEMENTARY)")
    print("="*60)
    
    print(f"\n{group_label} Users (n={optimal_percent_stats['n_observations']}):")
    print(f"  Mean percentage difference: {optimal_percent_stats['mean_diff_percent']:.3f}%")
    print(f"  Std of percentage difference: {optimal_percent_stats['std_diff_percent']:.3f}%")
    print(f"  95% LoA: [{optimal_percent_stats['loa_lower_percent']:.3f}%, {optimal_percent_stats['loa_upper_percent']:.3f}%]")
    
    print(f"\nRemaining Users (n={remaining_percent_stats['n_observations']}):")
    print(f"  Mean percentage difference: {remaining_percent_stats['mean_diff_percent']:.3f}%")
    print(f"  Std of percentage difference: {remaining_percent_stats['std_diff_percent']:.3f}%")
    print(f"  95% LoA: [{remaining_percent_stats['loa_lower_percent']:.3f}%, {remaining_percent_stats['loa_upper_percent']:.3f}%]")
    
    print(f"\nComparison (Standard Difference):")
    print(f"  Difference in mean bias: {abs(optimal_stats['mean_diff'] - remaining_stats['mean_diff']):.3f}")
    print(f"  Difference in precision: {abs(optimal_stats['std_diff'] - remaining_stats['std_diff']):.3f}")
    
    # Interpret the results (commented out as obsolete)
    # print(f"\nInterpretation:")
    # if abs(optimal_stats['mean_diff']) < abs(remaining_stats['mean_diff']):
    #     print(f"  + {group_label} users show less systematic bias")
    # else:
    #     print(f"  - {group_label} users show more systematic bias")

    # if optimal_stats['std_diff'] < remaining_stats['std_diff']:
    #     print(f"  + {group_label} users show better precision (smaller variability)")
    # else:
    #     print(f"  - {group_label} users show worse precision (larger variability)")
    
    # Log ratio interpretation
    # print(f"\nLog Ratio Interpretation:")
    # print(f"  - Mean ratio = 1.0: No systematic bias (recorded ≈ reported)")
    # print(f"  - Mean ratio < 1.0: Systematic under-reporting (recorded < reported)")
    # print(f"  - Mean ratio > 1.0: Systematic over-reporting (recorded > reported)")
    # print(f"  - Narrower LoA range = more consistent reporting")
    
    print(f"\nLimitations:")
    print(f"  - The {group_label} user group, while showing better agreement than the remaining group,")
    print(f"    still exhibits considerable variability in reporting accuracy.")
    print(f"    The LoA for this group ([{optimal_log_stats['loa_lower_ratio']:.2f}, {optimal_log_stats['loa_upper_ratio']:.2f}])")
    print(f"    indicate that individual differences can still be substantial.")

    print(f"\nSample data verification:")
    print(f"{group_label} users sample:")
    print(concordant_data[['user_key', 'reported_usage', 'actual_usage']].head())

    print(f"\nRemaining users sample:")
    print(remaining_data[['user_key', 'reported_usage', 'actual_usage']].head())

    print(f"\n" + "="*60)
    print("USER COHORT VERIFICATION")
    print("="*60)

    concordant_user_ids = sorted(concordant_data['user_key'].unique())
    remaining_user_ids = sorted(remaining_data['user_key'].unique())

    print(f"{group_label} Users:")
    print(f"  Count: {len(concordant_user_ids)} patients")
    print(f"  Patient IDs: {concordant_user_ids}")
    print(f"  Observations: {len(concordant_data)}")

    print(f"\nRemaining Assessed Users:")
    print(f"  Count: {len(remaining_user_ids)} patients")
    print(f"  Patient IDs: {remaining_user_ids}")
    print(f"  Observations: {len(remaining_data)}")

    print(f"\nTotal assessed patients: {len(concordant_user_ids) + len(remaining_user_ids)}")
    print(f"Total observations: {len(concordant_data) + len(remaining_data)}")

if __name__ == "__main__":
    main()
